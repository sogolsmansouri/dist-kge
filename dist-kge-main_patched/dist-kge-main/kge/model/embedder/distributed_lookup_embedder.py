import math
import time
from pathlib import Path
from torch import Tensor
import torch.nn
import torch.nn.functional

import torch
import numpy as np
from collections import deque

from kge import Config, Dataset
from kge.model import LookupEmbedder, KgeEmbedder
from kge.distributed.misc import get_optimizer_dim

from typing import List, Optional, Tuple


class DistributedLookupEmbedder(LookupEmbedder):
    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        vocab_size: int,
        parameter_client: "KgeParameterClient",
        complete_vocab_size,
        lapse_offset=0,
        init_for_load_only=False,
    ):
        super().__init__(
            config,
            dataset,
            configuration_key,
            vocab_size,
            init_for_load_only=init_for_load_only,
        )
        self.optimizer_dim = get_optimizer_dim(config, self.dim)
        self.optimizer_values = torch.zeros(
            (self.vocab_size, self.optimizer_dim),
            dtype=torch.float32,
            requires_grad=False,
        )

        self.complete_vocab_size = complete_vocab_size
        self.parameter_client = parameter_client
        self.lapse_offset = lapse_offset
        self.pulled_ids = None
        # global to local mapper only used in sync level partition
        self.global_to_local_mapper = torch.full(
            (self.dataset.num_entities(),), -1, dtype=torch.long, device="cpu"
        )

        # maps the local embeddings to the embeddings in lapse
        # used in optimizer
        self.local_to_lapse_mapper = torch.full(
            (vocab_size,), -1, dtype=torch.long, requires_grad=False
        )
        self.pull_dim = self.dim + self.optimizer_dim
        self.unnecessary_dim = self.parameter_client.dim - self.pull_dim

        # 3 pull tensors to pre-pull up to 3 batches
        # first boolean denotes if the tensor is free
        number_of_pre_pulls = 0
        if "entity" in self.configuration_key:
            number_of_pre_pulls = self.config.get("job.distributed.entity_pre_pull")
        elif "relation" in self.configuration_key:
            number_of_pre_pulls = self.config.get("job.distributed.relation_pre_pull")
        self.pull_tensors = []
        for i in range(number_of_pre_pulls + 1):
            self.pull_tensors.append(
                [
                    True,
                    torch.empty(
                        (self.vocab_size, self.parameter_client.dim),
                        # (self.vocab_size, self.dim + self.optimizer_dim),
                        dtype=torch.float32,
                        device="cpu",
                        requires_grad=False,
                    ),
                ]
            )

        if "cuda" in config.get("job.device"):
            # only pin tensors if we are using gpu
            # otherwise gpu memory will be allocated for no reason
            with torch.cuda.device(config.get("job.device")):
                for i in range(len(self.pull_tensors)):
                    self.pull_tensors[i][1] = self.pull_tensors[i][1].pin_memory()

        self.num_pulled = 0
        self.mapping_time = 0.0
        # self.pre_pulled = None
        self.pre_pulled = deque()
        self.copy_stream = None
        if "cuda" in config.get("job.device"):
            with torch.cuda.device(config.get("job.device")):
                self.copy_stream = torch.cuda.Stream()
        self.locality_rank = self._load_locality_rank()
        self._hot_cache_ids: Optional[torch.Tensor] = None
        self._hot_cache_id_to_slot: Optional[torch.Tensor] = None
        self._hot_cache_embeddings: Optional[torch.Tensor] = None
        self._hot_cache_optimizer: Optional[torch.Tensor] = None
        self._last_hot_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._active_partition_id: Optional[int] = None
        self._active_partition_version: Optional[int] = None

    def to_device(self, move_optim_data=True):
        """Needs to be called after model.to(self.device)"""
        if move_optim_data:
            self.optimizer_values = self.optimizer_values.to(
                self._embeddings.weight.device
            )

    @torch.no_grad()
    def init_pretrained(self, pretrained_embedder: KgeEmbedder) -> None:
        (
            self_intersect_ind,
            pretrained_intersect_ind,
        ) = self._intersect_ids_with_pretrained_embedder(pretrained_embedder)
        # process in chunks to reduce memory footprint
        chunk_size = 1000000
        num_objects = len(pretrained_intersect_ind)
        for chunk_number in range(math.ceil(num_objects / chunk_size)):
            chunk_start = chunk_size * chunk_number
            chunk_end = min(chunk_size * (chunk_number + 1), num_objects)
            current_chunk_size = chunk_end - chunk_start
            pretrained_embeddings = pretrained_embedder.embed(
                torch.from_numpy(pretrained_intersect_ind[chunk_start:chunk_end])
            )
            self.parameter_client.push(
                torch.from_numpy(self_intersect_ind[chunk_start:chunk_end]) + self.lapse_offset,
                torch.cat(
                    (
                        pretrained_embeddings,
                        torch.zeros(
                            (current_chunk_size, self.optimizer_dim + self.unnecessary_dim),
                            dtype=pretrained_embeddings.dtype,
                        ),
                    ),
                    dim=1,
                ),
            )

    def push_all(self):
        if self.unnecessary_dim > 0:
            # todo: this is currently just a workaround until we support parameter
            #  of different lengths
            push_tensor = torch.cat(
                (
                    self._embeddings.weight.detach().cpu(),
                    self.optimizer_values.cpu(),
                    torch.empty(
                        [len(self.optimizer_values), self.unnecessary_dim],
                        device="cpu",
                        dtype=self.optimizer_values.dtype,
                    ),
                ),
                dim=1,
            )
        else:
            push_tensor = torch.cat(
                (self._embeddings.weight.detach().cpu(), self.optimizer_values.cpu()),
                dim=1,
            )
        self.parameter_client.push(
            torch.arange(self.vocab_size) + self.lapse_offset, push_tensor
        )

    def pull_all(self):
        self._pull_embeddings(torch.arange(self.complete_vocab_size))

    def set_embeddings(self):
        # storing set_indexes and set_tensors in self to keep them alive until async
        #  set is finished
        self.set_indexes = self.pulled_ids + self.lapse_offset
        num_pulled = len(self.set_indexes)
        # move tensors to cpu before cat to reduce gpu memory usage
        if self.unnecessary_dim > 0:
            self.set_tensor = torch.cat(
                (
                    self._embeddings.weight[:num_pulled].detach().cpu(),
                    self.optimizer_values[:num_pulled].cpu(),
                    torch.empty((num_pulled, self.unnecessary_dim), device="cpu"),
                ),
                dim=1,
            )
        else:
            self.set_tensor = torch.cat(
                (
                    self._embeddings.weight[:num_pulled].detach().cpu(),
                    self.optimizer_values[:num_pulled].cpu(),
                ),
                dim=1,
            )
        self.parameter_client.set(self.set_indexes, self.set_tensor, asynchronous=True)

    def _get_free_pull_tensor(self):
        for i, (free, pull_tensor) in enumerate(self.pull_tensors):
            if free:
                self.pull_tensors[i][0] = False
                return i, pull_tensor
        return None

    def _async_copy_to_device(self, tensor):
        device = self._embeddings.weight.device
        if tensor.device == device:
            return tensor, None
        if self.copy_stream is None:
            return tensor.to(device, non_blocking=True), None
        target = torch.empty_like(tensor, device=device)
        with torch.cuda.stream(self.copy_stream):
            target.copy_(tensor, non_blocking=True)
        event = torch.cuda.Event()
        self.copy_stream.record_event(event)
        return target, event

    @torch.no_grad()
    def pre_pull(self, indexes):
        pull_indexes = (indexes + self.lapse_offset).cpu()
        result = self._get_free_pull_tensor()
        if result is None:
            return
        pull_tensor_index, pull_tensor = result
        num_indexes = len(indexes)
        pull_tensor = pull_tensor[:num_indexes]
        pull_future = self.parameter_client.pull(
            pull_indexes, pull_tensor, asynchronous=True
        )
        self.pre_pulled.append(
            {
                "indexes": indexes,
                "num_indexes": num_indexes,
                "pull_indexes": pull_indexes,
                "pull_tensor": pull_tensor,
                "pull_future": pull_future,
                "pull_tensor_index": pull_tensor_index,
            }
        )

    def pre_pulled_to_device(self):
        if len(self.pre_pulled) > 2:
            # id 0 is from the batch currently processed
            # last one is the one pulled from ps
            # we are moving the second last
            entry = self.pre_pulled[-2]
            if entry.get("device_tensor") is None:
                self.parameter_client.wait(entry["pull_future"])
                device_tensor, event = self._async_copy_to_device(entry["pull_tensor"])
                entry["device_tensor"] = device_tensor
                entry["copy_event"] = event

    @torch.no_grad()
    def _pull_embeddings(self, indexes):
        cpu_gpu_time = 0.0
        pull_time = 0.0
        device = self._embeddings.weight.device
        use_prefetch = len(self.pre_pulled) > 0
        if use_prefetch:
            # todo: add workaround for relations here as well
            # todo: clean up this method
            pre_pulled = self.pre_pulled.popleft()
            indexes = pre_pulled["indexes"]
            len_indexes = pre_pulled.get("num_indexes", len(indexes))
            pull_indexes = pre_pulled["pull_indexes"][:len_indexes]
            self.pulled_ids = indexes
            self.parameter_client.wait(pre_pulled["pull_future"])
            self.local_to_lapse_mapper[:len_indexes] = pull_indexes
            tensor = pre_pulled.get("device_tensor")
            event = pre_pulled.get("copy_event")
            if tensor is None:
                cpu_gpu_time -= time.time()
                tensor = pre_pulled["pull_tensor"].to(device, non_blocking=True)
                cpu_gpu_time += time.time()
            else:
                if event is not None and device.type == "cuda":
                    torch.cuda.current_stream(device).wait_event(event)
            pre_pulled_tensor = tensor
            pulled_embeddings, pulled_optim_values = torch.split(
                pre_pulled_tensor, [self.dim, self.optimizer_dim], dim=1
            )
            self._embeddings.weight[:len_indexes] = pulled_embeddings
            self.optimizer_values[:len_indexes] = pulled_optim_values
            self.pull_tensors[pre_pulled["pull_tensor_index"]][0] = True
            return pull_time, cpu_gpu_time

        len_indexes = len(indexes)
        self.pulled_ids = indexes
        output_rows = torch.arange(len_indexes, dtype=torch.long)
        indexes_cpu = indexes.cpu().long()
        device = self._embeddings.weight.device
        hot_mask = None
        if self._hot_cache_id_to_slot is not None:
            slots = self._hot_cache_id_to_slot[indexes_cpu]
            hot_mask = slots >= 0
            if hot_mask.any():
                hot_rows = output_rows[hot_mask]
                cache_rows = slots[hot_mask]
                hot_rows_device = hot_rows.to(device)
                cache_rows_device = cache_rows.to(device)
                self._embeddings.weight.data[hot_rows_device] = self._hot_cache_embeddings[
                    cache_rows_device
                ]
                self.optimizer_values[hot_rows_device] = self._hot_cache_optimizer[
                    cache_rows_device
                ]
                self._last_hot_batch = (
                    hot_rows.clone(),
                    cache_rows.clone(),
                )
            else:
                self._last_hot_batch = None
        else:
            slots = None
            self._last_hot_batch = None

        if hot_mask is None:
            cold_mask = torch.ones(len_indexes, dtype=torch.bool)
        else:
            cold_mask = ~hot_mask

        cold_rows = output_rows[cold_mask]
        num_cold = cold_rows.numel()
        if num_cold > 0:
            cold_indexes = indexes_cpu[cold_mask]
            pull_tensor = self.pull_tensors[0][1][:num_cold]
            pull_time -= time.time()
            self.parameter_client.pull(
                cold_indexes + self.lapse_offset, pull_tensor
            )
            pull_time += time.time()
            cpu_gpu_time -= time.time()
            pulled_embeddings, pulled_optim_values, _ = torch.split(
                pull_tensor, [self.dim, self.optimizer_dim, self.unnecessary_dim], dim=1
            )
            cold_rows_device = cold_rows.to(device)
            self._embeddings.weight.data[cold_rows_device] = pulled_embeddings.to(device)
            self.optimizer_values[cold_rows_device] = pulled_optim_values.to(
                self.optimizer_values.device
            )
            cpu_gpu_time += time.time()

        self.local_to_lapse_mapper[output_rows] = indexes_cpu + self.lapse_offset
        return pull_time, cpu_gpu_time

    def localize(self, indexes: Tensor, asynchronous=False, make_unique=False):
        if make_unique:
            indexes = torch.unique(indexes)
        self.parameter_client.localize(
            (indexes + self.lapse_offset).cpu(), asynchronous
        )

    def _embed(self, indexes: Tensor) -> Tensor:
        long_indexes = indexes.long()
        return self._embeddings(long_indexes)

    def embed(self, indexes: Tensor) -> Tensor:
        long_indexes = indexes.long()
        return self._postprocess(self._embeddings(long_indexes))

    def embed_all(self) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def push_back(self):
        if (
            self._hot_cache_embeddings is not None
            and self._last_hot_batch is not None
        ):
            batch_rows, cache_rows = self._last_hot_batch
            if batch_rows.numel() > 0:
                batch_rows_device = batch_rows.to(self._embeddings.weight.device)
                cache_rows_device = cache_rows.to(self._hot_cache_embeddings.device)
                self._hot_cache_embeddings[cache_rows_device] = self._embeddings.weight[
                    batch_rows_device
                ]
                self._hot_cache_optimizer[cache_rows_device] = self.optimizer_values[
                    batch_rows_device
                ]
        self._last_hot_batch = None
        self.local_to_lapse_mapper[:] = -1
        self.num_pulled = 0

    def _embeddings_all(self) -> Tensor:
        # TODO: this should not be possible in the distributed lookup embedder
        raise NotImplementedError

    def penalty(self, **kwargs) -> List[Tensor]:
        # TODO factor out to a utility method
        # Avoid calling lookup embedder penalty and instead call KgeEmbedder penalty
        result = KgeEmbedder.penalty(self, **kwargs)
        if self.regularize == "" or self.get_option("regularize_weight") == 0.0:
            pass
        elif self.regularize == "lp":
            p = (
                self.get_option("regularize_args.p")
                if self.has_option("regularize_args.p")
                else 2
            )
            regularize_weight = self._get_regularize_weight()
            if not self.get_option("regularize_args.weighted"):
                # unweighted Lp regularization
                parameters = self._embeddings_all()
                result += [
                    (
                        f"{self.configuration_key}.L{p}_penalty",
                        (regularize_weight / p * parameters.norm(p=p) ** p).sum(),
                    )
                ]
            else:
                # weighted Lp regularization
                unique_indexes, counts = torch.unique(
                    kwargs["indexes"], return_counts=True
                )
                parameters = self._embed(unique_indexes)
                if p % 2 == 1:
                    parameters = torch.abs(parameters)
                result += [
                    (
                        f"{self.configuration_key}.L{p}_penalty",
                        (
                            regularize_weight
                            / p
                            * (parameters ** p * counts.float().view(-1, 1))
                        ).sum()
                        # In contrast to unweighted Lp regularization, rescaling by
                        # number of triples/indexes is necessary here so that penalty
                        # term is correct in expectation
                        / len(kwargs["indexes"]),
                    )
                ]
        else:  # unknown regularization
            raise ValueError(f"Invalid value regularize={self.regularize}")

        return result

    def enable_hot_cache(self, hot_ids: torch.Tensor):
        if hot_ids is None or len(hot_ids) == 0:
            return
        hot_ids = torch.unique(hot_ids.long())
        device = self._embeddings.weight.device
        optimizer_device = self.optimizer_values.device
        num_hot = len(hot_ids)
        mapping_size = self.complete_vocab_size or self.dataset.num_entities()
        mapping_size = max(mapping_size, int(hot_ids.max().item()) + 1)
        mapping = torch.full(
            (mapping_size,), -1, dtype=torch.long
        )
        mapping[hot_ids.cpu()] = torch.arange(num_hot, dtype=torch.long)
        self._hot_cache_id_to_slot = mapping
        self._hot_cache_embeddings = torch.empty(
            (num_hot, self.dim), device=device
        )
        self._hot_cache_optimizer = torch.empty(
            (num_hot, self.optimizer_dim), device=optimizer_device
        )
        pull_tensor = torch.empty(
            (num_hot, self.dim + self.optimizer_dim + self.unnecessary_dim),
            dtype=torch.float32,
            device="cpu",
        )
        self.parameter_client.pull(
            (hot_ids + self.lapse_offset).cpu(), pull_tensor
        )
        embeddings, optim_values, _ = torch.split(
            pull_tensor, [self.dim, self.optimizer_dim, self.unnecessary_dim], dim=1
        )
        self._hot_cache_embeddings.copy_(embeddings.to(device))
        self._hot_cache_optimizer.copy_(optim_values.to(optimizer_device))
        self._hot_cache_ids = hot_ids
        self._last_hot_batch = None

    def _load_locality_rank(self):
        cfg = self.config.get("job.distributed.locality_ordering") or {}
        if not cfg.get("enable", False):
            return None
        if "entity" in self.configuration_key:
            filename = cfg.get("entity_rank_file") or "analysis_entity_locality_rank.npy"
            expected = self.dataset.num_entities()
        elif "relation" in self.configuration_key:
            filename = cfg.get("relation_rank_file") or "analysis_relation_locality_rank.npy"
            expected = self.dataset.num_relations()
        else:
            return None
        rank_path = Path(self.dataset.folder) / filename
        if not rank_path.is_file():
            return None
        try:
            data = np.load(rank_path)
            if len(data) != expected:
                self.config.log(
                    f"Ignoring locality rank {rank_path} (expected {expected} entries, found {len(data)})."
                )
                return None
            tensor = torch.from_numpy(data.astype(np.int64))
            self.config.log(f"Loaded locality rank from {rank_path}.")
            return tensor
        except Exception as exc:
            self.config.log(f"Failed to load locality rank from {rank_path}: {exc}")
            return None

    def apply_locality_order(self, indexes: Tensor) -> Tensor:
        if self.locality_rank is None or indexes.numel() <= 1:
            return indexes
        if indexes.device.type != "cpu":
            cpu_idx = indexes.detach().cpu()
            ranks = self.locality_rank[cpu_idx]
            order = torch.argsort(ranks)
            return cpu_idx[order].to(indexes.device)
        ranks = self.locality_rank[indexes]
        order = torch.argsort(ranks)
        return indexes[order]

    def set_partition_context(self, partition_id: int, partition_version: int):
        self._active_partition_id = partition_id
        self._active_partition_version = partition_version

    def clear_partition_context(self):
        self._active_partition_id = None
        self._active_partition_version = None

    def get_partition_context(self):
        return {
            "partition_id": self._active_partition_id,
            "partition_version": self._active_partition_version,
        }
