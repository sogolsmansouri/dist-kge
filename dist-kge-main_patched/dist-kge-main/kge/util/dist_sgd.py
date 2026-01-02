import torch
import numpy as np
from collections import deque
from torch.optim.optimizer import Optimizer, required


class DistSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(
        self,
        model,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        parameter_client=None,
        lapse_indexes=None,
        local_index_mappers=None,
        conflict_free_merge=False,
        causal_merge=False,
        row_causal_merge=False,
    ):
        params = [p for p in model.parameters() if p.requires_grad]
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        self.lapse_indexes = lapse_indexes
        # self.local_index_mappers = local_index_mappers
        self.local_index_mappers = [
            model._entity_embedder.local_index_mapper,
            model._relation_embedder.local_index_mapper,
        ]
        self.local_to_lapse_mappers = [
            model._entity_embedder.local_to_lapse_mapper,
            model._relation_embedder.local_to_lapse_mapper,
        ]
        self.lapse_offsets = [
            model._entity_embedder.lapse_offset,
            model._relation_embedder.lapse_offset,
        ]
        self.parameter_client = parameter_client
        self.conflict_free_merge = bool(conflict_free_merge)
        self.causal_merge = bool(causal_merge)
        self.row_causal_merge = bool(row_causal_merge)
        self._partition_context = {
            "partition_id": None,
            "partition_version": None,
        }
        self._partition_context_map = None
        self._partition_context_offsets = {
            "entity": int(self.lapse_offsets[0]),
            "relation": int(self.lapse_offsets[1]),
        }
        self._partition_context_maps = {"entity": None, "relation": None}
        self._current_replay_key = None
        self._current_replay_payloads = []
        self._replay_archive = {}
        self._replay_archive_order = deque()
        self._replay_archive_capacity = 8
        self._window_replay_payloads = None
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(DistSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DistSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                if d_p.is_sparse:
                    d_p = (
                        d_p.coalesce()
                    )  # the update is non-linear so indices must be unique
                    push_tensor = d_p._values().mul_(-group["lr"]).cpu()
                    update_indexes = d_p._indices().cpu()
                    push_keys = self.local_to_lapse_mappers[i][update_indexes].view(-1)
                    group_name = "relation" if i == 1 else "entity"
                    self._push_with_context(group_name, push_keys, push_tensor)
                else:
                    indexes_to_push_mask = self.local_to_lapse_mappers[i] != -1
                    self._push_with_context(
                        "relation" if i == 1 else "entity",
                        self.local_to_lapse_mappers[i][indexes_to_push_mask],
                        (-group["lr"] * d_p).cpu()[indexes_to_push_mask],
                    )
                # self.lapse_worker.push(self.lapse_indexes[i], (-group['lr']*d_p).cpu().to_dense().numpy())
                # p.add_(d_p, alpha=-group['lr'])

        return loss

    def _push_with_context(self, group_name, keys, payload):
        if (
            self._partition_context_map is not None
            and group_name in self._partition_context_maps
            and hasattr(self.parameter_client, "push_versioned")
        ):
            result = self._push_with_partition_map(group_name, keys, payload)
            self._record_partition_push(keys, payload)
            return result
        if (
            (self.conflict_free_merge or self.causal_merge or self.row_causal_merge)
            and self._partition_context["partition_id"] is not None
            and self._partition_context["partition_version"] is not None
            and hasattr(self.parameter_client, "push_versioned")
        ):
            result = self.parameter_client.push_versioned(
                keys,
                payload,
                self._partition_context["partition_id"],
                self._partition_context["partition_version"],
            )
            self._record_partition_push(keys, payload)
            return result
        result = self.parameter_client.push(keys, payload)
        self._record_partition_push(keys, payload)
        return result

    def _push_with_partition_map(self, group_name, keys, payload):
        partition_map = self._partition_context_maps.get(group_name)
        if partition_map is None or keys.numel() == 0:
            return self.parameter_client.push(keys, payload)
        offset = self._partition_context_offsets.get(group_name, 0)
        raw_ids = keys - offset
        valid_mask = (raw_ids >= 0) & (raw_ids < partition_map.numel())
        if not torch.any(valid_mask):
            return self.parameter_client.push(keys, payload)
        raw_ids = raw_ids[valid_mask].long()
        keys = keys[valid_mask]
        payload = payload[valid_mask]
        partition_ids = partition_map[raw_ids]
        for pid in torch.unique(partition_ids).tolist():
            version = self._partition_context_map.get(pid)
            pid_mask = partition_ids == pid
            if not torch.any(pid_mask):
                continue
            keys_pid = keys[pid_mask]
            payload_pid = payload[pid_mask]
            self._record_window_push(
                int(pid), self._partition_context_map.get(pid), keys_pid, payload_pid
            )
            if version is None:
                self.parameter_client.push(keys_pid, payload_pid)
            else:
                self.parameter_client.push_versioned(
                    keys_pid,
                    payload_pid,
                    int(pid),
                    int(version),
                )
        return None

    def pull_entities(self, entity_ids):
        pass

    def pull_relations(self, relation_ids):
        pass

    def set_entities(self):
        pass

    def set_relations(self):
        pass

    def pull_all(self):
        pass

    def push_all(self):
        pass

    def set_partition_context(self, partition_id: int, partition_version: int):
        self._partition_context["partition_id"] = partition_id
        self._partition_context["partition_version"] = partition_version
        self._current_replay_key = (partition_id, partition_version)
        self._current_replay_payloads = []

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
        self._window_replay_payloads = {}

    def clear_partition_context_map(self):
        if self._window_replay_payloads:
            for key, payloads in self._window_replay_payloads.items():
                self._store_replay_entry(key, payloads)
        self._partition_context_map = None
        self._partition_context_maps["entity"] = None
        self._partition_context_maps["relation"] = None
        self._window_replay_payloads = None

    def finalize_partition_context(self):
        if self._current_replay_key is not None:
            self._store_replay_entry(
                self._current_replay_key, self._current_replay_payloads
            )
        self._current_replay_key = None
        self._current_replay_payloads = []
        self._partition_context["partition_id"] = None
        self._partition_context["partition_version"] = None

    def get_partition_context(self):
        return dict(self._partition_context)

    def _store_replay_entry(self, key, payloads):
        entry = [self._clone_record(rec) for rec in payloads]
        self._replay_archive[key] = entry
        self._replay_archive_order.append(key)
        while len(self._replay_archive_order) > self._replay_archive_capacity:
            old_key = self._replay_archive_order.popleft()
            self._replay_archive.pop(old_key, None)

    @staticmethod
    def _clone_record(record):
        keys, payload = record
        return (keys.clone(), payload.clone())

    def _record_partition_push(self, keys, payload):
        if self._current_replay_key is None:
            return
        if keys.numel() == 0 or payload.numel() == 0:
            return
        self._current_replay_payloads.append((keys.cpu(), payload.cpu()))

    def _record_window_push(self, partition_id, version, keys, payload):
        if self._window_replay_payloads is None or version is None:
            return
        if keys.numel() == 0 or payload.numel() == 0:
            return
        key = (int(partition_id), int(version))
        entry = self._window_replay_payloads.setdefault(key, [])
        entry.append((keys.cpu(), payload.cpu()))

    def replay_partition_updates(
        self, partition_id: int, original_version: int, replay_version: int
    ):
        key = (partition_id, original_version)
        entry = self._replay_archive.get(key)
        if entry is None:
            return False
        for keys, payload in entry:
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
