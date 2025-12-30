from collections import deque
import torch
import numpy as np
from torch.optim.optimizer import Optimizer
from copy import deepcopy

from kge.util.triton_fused import fused_embedding_optimizer_update


class DistAdagrad(Optimizer):
    """Implements Adagrad algorithm.

    It has been proposed in `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)

    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
    """

    def __init__(
        self,
        # model,
        params,
        lr=1e-2,
        lr_decay=0,
        weight_decay=0,
        initial_accumulator_value=0,
        eps=1e-10,
        parameter_client=None,
        lapse_indexes=None,
        lapse_optimizer_index_offset=0,
        async_write_back=[],
        is_row=False,
        use_lr_scheduler=False,
        min_rank=-1,
        max_pending_pushes=2,
        conflict_free_merge=False,
        causal_merge=False,
        row_causal_merge=False,
        record_replay=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError(
                "Invalid initial_accumulator_value value: {}".format(
                    initial_accumulator_value
                )
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        self.lapse_optimizer_index_offset = lapse_optimizer_index_offset
        self.lapse_indexes = lapse_indexes
        self.pulled_parameters = [None, None]
        self.async_write_back = async_write_back
        self._async_write_back_map = {
            "entity": bool(async_write_back[0]) if len(async_write_back) > 0 else True,
            "relation": bool(async_write_back[1]) if len(async_write_back) > 1 else True,
        }

        self.is_row = is_row

        self.parameter_client = parameter_client
        self._push_buffer_pool = {"entity": [], "relation": []}
        self.use_lr_scheduler = use_lr_scheduler
        self.min_rank = min_rank
        self.entity_async_wait_values = deque()
        self.relation_async_wait_values = deque()
        self.async_wait_values = {
            "entity": self.entity_async_wait_values,
            "relation": self.relation_async_wait_values,
        }
        self.max_pending_pushes = max(1, int(max_pending_pushes))
        self.conflict_free_merge = bool(conflict_free_merge)
        self.causal_merge = bool(causal_merge)
        self.row_causal_merge = bool(row_causal_merge)
        self._record_replay = bool(record_replay)
        self._partition_context = {
            "partition_id": None,
            "partition_version": None,
        }
        self._partition_context_map = None
        self._partition_context_offsets = {"entity": 0, "relation": 0}
        self._partition_context_maps = {"entity": None, "relation": None}
        self._window_replay_payloads = None
        self._current_replay_key = None
        self._current_replay_payloads = {"entity": [], "relation": []}
        self._replay_archive = {}
        self._replay_archive_order = deque()
        self._replay_archive_capacity = 8

        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
        )
        super(DistAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            if group["name"] != "default":
                if parameter_client.get_lr(group["name"]) == 0:
                    self.parameter_client.set_lr(group["name"], group["lr"])
            if "lapse_offset" in group:
                self._partition_context_offsets[group["name"]] = int(
                    group["lapse_offset"]
                )
            group["prev_lr"] = group["lr"]
            for i, p in enumerate(group["params"]):
                state = self.state[p]
                state["step"] = 0
                # state["sum"] = self.optimizer_values[i]

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # we need to wait here for the previous push to finish, otherwise we can not
        #  delete the push tensors
        self._process_wait_queue(
            self.entity_async_wait_values,
            self._async_write_back_map["entity"],
            "entity",
        )
        self._process_wait_queue(
            self.relation_async_wait_values,
            self._async_write_back_map["relation"],
            "relation",
        )
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if group["lr_decay"] > 0:
                    # we only need to synchronize steps between workers if we actually
                    #  use the step variable for something
                    self.parameter_client.step_optim(i)
                    state["step"] = self.parameter_client.get_step_optim(group["name"])
                else:
                    state["step"] += 1
                if self.use_lr_scheduler:
                    if self.parameter_client.rank == self.min_rank:
                        if group["prev_lr"] != group["lr"]:
                            self.parameter_client.set_lr(group["name"], group["lr"])
                    group["lr"] = self.parameter_client.get_lr(group["name"])

                if group["weight_decay"] != 0:
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "weight_decay option is not compatible with sparse gradients"
                        )
                    grad = grad.add(p, alpha=group["weight_decay"])

                clr = group["lr"] / (1 + (state["step"] - 1) * group["lr_decay"])

                if grad.is_sparse:
                    grad = (
                        grad.coalesce()
                    )  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()[0]
                    # grad_indices_flat = grad_indices.flatten()
                    grad_values = grad._values()
                    size = grad.size()

                    # pull the current internal optimizer parameters
                    state_sum = group["optimizer_values"][grad_indices]

                    if not self.is_row:
                        sum_update_values = grad_values.pow(2)
                    else:
                        sum_update_values = grad_values.pow(2).mean(1).view(-1, 1)
                    state_sum.add_(sum_update_values)
                    # std = state["sum"].sparse_mask(grad)
                    # std_values = std._values().sqrt_().add_(group["eps"])
                    std_values = state_sum.sqrt().add_(group["eps"])
                    update_value = (grad_values / std_values).mul_(-clr)
                    if group["sync_level"] == "batch":
                        update_indexes = grad_indices.cpu()
                        push_keys = group["local_to_lapse_mapper"][update_indexes]
                        payload_buffer = self._acquire_push_buffer(
                            group["name"], len(update_value)
                        )
                        payload = payload_buffer[: len(update_value)]
                        self._pack_push_payload(
                            payload, update_value, sum_update_values
                        )
                        wait_value = self._push_with_context(
                            group["name"],
                            push_keys,
                            payload,
                            asynchronous=self._async_write_back_map[group["name"]],
                        )
                        self._enqueue_wait(group["name"], wait_value, payload_buffer)
                        self._record_partition_push(
                            group["name"],
                            push_keys.clone(),
                            payload[: len(update_value)].clone(),
                        )
                    else:
                        fused_embedding_optimizer_update(
                            p.data,
                            group["optimizer_values"],
                            grad_indices,
                            update_value,
                            state_sum,
                        )
                    # p.add_(make_sparse(grad_values / std_values), alpha=-clr)
                else:
                    raise ValueError(
                        "Currently only sparse parameters supported with dist_adagrad and dist_rowadagrad"
                    )

        return loss

    def _push_with_context(self, group_name, keys, payload, asynchronous=False):
        if (
            self._partition_context_map is not None
            and group_name in self._partition_context_maps
            and hasattr(self.parameter_client, "push_versioned")
        ):
            return self._push_with_partition_map(
                group_name, keys, payload, asynchronous=asynchronous
            )
        if (
            (self.conflict_free_merge or self.causal_merge or self.row_causal_merge)
            and self._partition_context["partition_id"] is not None
            and self._partition_context["partition_version"] is not None
            and hasattr(self.parameter_client, "push_versioned")
        ):
            return self.parameter_client.push_versioned(
                keys,
                payload,
                self._partition_context["partition_id"],
                self._partition_context["partition_version"],
                asynchronous=asynchronous,
            )
        return self.parameter_client.push(
            keys, payload, asynchronous=asynchronous
        )

    def _push_with_partition_map(self, group_name, keys, payload, asynchronous=False):
        partition_map = self._partition_context_maps.get(group_name)
        if partition_map is None or keys.numel() == 0:
            return self.parameter_client.push(
                keys, payload, asynchronous=asynchronous
            )
        offset = self._partition_context_offsets.get(group_name, 0)
        raw_ids = keys - offset
        valid_mask = (raw_ids >= 0) & (raw_ids < partition_map.numel())
        if not torch.any(valid_mask):
            return self.parameter_client.push(
                keys, payload, asynchronous=asynchronous
            )
        raw_ids = raw_ids[valid_mask].long()
        keys = keys[valid_mask]
        payload = payload[valid_mask]
        partition_ids = partition_map[raw_ids]
        wait_value = None
        for pid in torch.unique(partition_ids).tolist():
            version = self._partition_context_map.get(pid)
            pid_mask = partition_ids == pid
            if not torch.any(pid_mask):
                continue
            keys_pid = keys[pid_mask]
            payload_pid = payload[pid_mask]
            self._record_window_push(
                int(pid), self._partition_context_map.get(pid), group_name, keys_pid, payload_pid
            )
            if version is None:
                wait_value = self.parameter_client.push(
                    keys_pid, payload_pid, asynchronous=asynchronous
                )
            else:
                wait_value = self.parameter_client.push_versioned(
                    keys_pid,
                    payload_pid,
                    int(pid),
                    int(version),
                    asynchronous=asynchronous,
                )
        return wait_value

    def _acquire_push_buffer(self, group_name: str, rows: int) -> torch.Tensor:
        pool = self._push_buffer_pool[group_name]
        for idx, buffer in enumerate(pool):
            if buffer.size(0) >= rows:
                return pool.pop(idx)
        cols = self.parameter_client.dim
        capacity = max(rows, 1024)
        buffer = torch.empty((capacity, cols), dtype=torch.float32)
        if torch.cuda.is_available():
            buffer = buffer.pin_memory()
        return buffer

    def _release_push_buffer(self, group_name: str, buffer: torch.Tensor):
        self._push_buffer_pool[group_name].append(buffer)

    def _pack_push_payload(
        self, dest: torch.Tensor, update_value: torch.Tensor, optimizer_values: torch.Tensor
    ):
        rows = update_value.size(0)
        dest_view = dest[:rows]
        update_cols = update_value.size(1)
        optimizer_cols = optimizer_values.size(1)
        dest_view[:, :update_cols].copy_(
            update_value.contiguous(), non_blocking=True
        )
        dest_view[:, update_cols : update_cols + optimizer_cols].copy_(
            optimizer_values.contiguous(), non_blocking=True
        )
        remaining = dest_view.size(1) - update_cols - optimizer_cols
        if remaining > 0:
            dest_view[:, update_cols + optimizer_cols :].zero_()

    def _enqueue_wait(self, group_name: str, wait_value, buffer: torch.Tensor):
        if wait_value is None:
            self._release_push_buffer(group_name, buffer)
        else:
            self.async_wait_values[group_name].append((wait_value, buffer))

    def _process_wait_queue(self, queue: deque, async_enabled: bool, group_name: str):
        if async_enabled:
            while len(queue) >= self.max_pending_pushes:
                wait_value, buffer = queue.popleft()
                if wait_value is not None:
                    self.parameter_client.wait(wait_value)
                self._release_push_buffer(group_name, buffer)
        else:
            self._drain_wait_queue(queue, group_name)

    def _drain_wait_queue(self, queue: deque, group_name: str):
        while queue:
            wait_value, buffer = queue.popleft()
            if wait_value is not None:
                self.parameter_client.wait(wait_value)
            self._release_push_buffer(group_name, buffer)

    def flush_pending_pushes(self):
        self._drain_wait_queue(self.entity_async_wait_values, "entity")
        self._drain_wait_queue(self.relation_async_wait_values, "relation")

    def wait_for_pending(self, group_name: str):
        if group_name not in self.async_wait_values:
            return
        self._drain_wait_queue(self.async_wait_values[group_name], group_name)

    def pull_all(self):
        """
        loads optimizer values stored in distributed lookup embedder to state[sum]
        used for checkpoint of complete model
        embedder.pull_all needs to be called before this function.
        """
        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                self.state[p]["sum"] = group["optimizer_values"]
                self.state[p]["step"] = self.parameter_client.get_step_optim(
                    group["name"]
                )
            if group["name"] == "default":
                continue
            group["lr"] = self.parameter_client.get_lr(group["name"])

    def state_dict(self) -> dict:
        """
        We are removing the optimizer values from state dict since stored separately
        """
        state_dict = super(DistAdagrad, self).state_dict()
        for i, group in enumerate(state_dict["param_groups"]):
            for key in ["optimizer_values", "local_to_lapse_mapper"]:
                state_dict["param_groups"][i].pop(key, None)
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        """
        We need to keep the created references to the opitmizer values in the embedder.
        super.load_state_dict removes the created references if not in state_dict.
        """

        saved_references = list()
        for group in self.param_groups:
            ref = dict()
            if "optimizer_values" in group:
                ref["optimizer_values"] = group["optimizer_values"]
            if "local_to_lapse_mapper" in group:
                ref["local_to_lapse_mapper"] = group["local_to_lapse_mapper"]
            saved_references.append(ref)
        super(DistAdagrad, self).load_state_dict(state_dict)
        for ref, group in zip(saved_references, self.param_groups):
            group.update(ref)
            if group["name"] != "default":
                if self.parameter_client.get_lr(group["name"]) == 0:
                    self.parameter_client.set_lr(group["name"], group["lr"])

    def set_partition_context(self, partition_id: int, partition_version: int):
        self._partition_context["partition_id"] = partition_id
        self._partition_context["partition_version"] = partition_version
        if not self._record_replay:
            return
        self._current_replay_key = (partition_id, partition_version)
        self._current_replay_payloads = {"entity": [], "relation": []}

    def set_partition_context_map(
        self,
        partition_versions: dict,
        entity_partition_map: torch.Tensor,
        relation_partition_map: torch.Tensor,
    ):
        self._partition_context_map = {
            int(pid): int(version) for pid, version in partition_versions.items()
        }
        self._partition_context_maps["entity"] = entity_partition_map
        self._partition_context_maps["relation"] = relation_partition_map
        if not self._record_replay:
            return
        self._window_replay_payloads = {}

    def clear_partition_context_map(self):
        if self._record_replay and self._window_replay_payloads:
            for key, payloads in self._window_replay_payloads.items():
                self._store_replay_entry(key, payloads)
        self._partition_context_map = None
        self._partition_context_maps["entity"] = None
        self._partition_context_maps["relation"] = None
        self._window_replay_payloads = None

    def finalize_partition_context(self):
        if self._record_replay and self._current_replay_key is not None:
            self._store_replay_entry(
                self._current_replay_key, self._current_replay_payloads
            )
        self._current_replay_key = None
        self._current_replay_payloads = {"entity": [], "relation": []}
        self._partition_context["partition_id"] = None
        self._partition_context["partition_version"] = None

    def get_partition_context(self):
        return dict(self._partition_context)

    def _store_replay_entry(self, key, payloads):
        entry = {
            "entity": [self._clone_record(rec) for rec in payloads.get("entity", [])],
            "relation": [
                self._clone_record(rec) for rec in payloads.get("relation", [])
            ],
        }
        self._replay_archive[key] = entry
        self._replay_archive_order.append(key)
        while len(self._replay_archive_order) > self._replay_archive_capacity:
            old_key = self._replay_archive_order.popleft()
            self._replay_archive.pop(old_key, None)

    @staticmethod
    def _clone_record(record):
        keys, payload = record
        return (keys.clone(), payload.clone())

    def _record_partition_push(self, group_name, keys, payload):
        if not self._record_replay:
            return
        if self._current_replay_key is None:
            return
        if keys.numel() == 0 or payload.numel() == 0:
            return
        self._current_replay_payloads[group_name].append(
            (keys.cpu(), payload.cpu())
        )

    def _record_window_push(self, partition_id, version, group_name, keys, payload):
        if not self._record_replay:
            return
        if self._window_replay_payloads is None or version is None:
            return
        if keys.numel() == 0 or payload.numel() == 0:
            return
        key = (int(partition_id), int(version))
        entry = self._window_replay_payloads.setdefault(
            key, {"entity": [], "relation": []}
        )
        entry[group_name].append((keys.cpu(), payload.cpu()))

    def replay_partition_updates(
        self, partition_id: int, original_version: int, replay_version: int
    ):
        key = (partition_id, original_version)
        entry = self._replay_archive.get(key)
        if entry is None:
            return False
        for group_name in ("entity", "relation"):
            for keys, payload in entry[group_name]:
                if keys.numel() == 0:
                    continue
                self.parameter_client.push(keys, payload, asynchronous=False)
        new_key = (
            (partition_id, replay_version)
            if replay_version is not None and replay_version >= 0
            else key
        )
        self._store_replay_entry(new_key, entry)
        return True
