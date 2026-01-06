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
        if "relation" in configuration_key:
            mapper_size = self.dataset.num_relations()
        else:
            mapper_size = self.dataset.num_entities()
        self.global_to_local_mapper = torch.full(
            (mapper_size,), -1, dtype=torch.long, device="cpu"
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
        self._gpu_cache_enabled = False
        self._gpu_cache_max_entries = 0
        self._gpu_cache_max_insert_per_step = 50000
        self._gpu_cache_eviction_policy = "fifo"
        self._gpu_cache_lru_max_scan = 0
        self._gpu_cache_access_clock = 0
        self._gpu_cache_last_access: Optional[torch.Tensor] = None
        self._gpu_cache_pinned_persist = False
        self._gpu_cache_pinned_max_entries = 0
        self._gpu_cache_pinned_capacity = 0
        self._gpu_cache_pinned_next_slot = 0
        self._gpu_cache_prefetch_stream = None
        self._gpu_cache_prefetch_event = None
        self._gpu_cache_prefetch_pending = None
        self._gpu_cache_last_pinned_ids = None
        self._gpu_cache_ids: Optional[torch.Tensor] = None
        self._gpu_cache_id_to_slot: Optional[torch.Tensor] = None
        self._gpu_cache_embeddings: Optional[torch.Tensor] = None
        self._gpu_cache_optimizer: Optional[torch.Tensor] = None
        self._gpu_cache_next_slot = 0
        self._gpu_cache_hits = 0
        self._gpu_cache_misses = 0
        self._active_partition_id: Optional[int] = None
        self._active_partition_version: Optional[int] = None
        self._init_gpu_cache_config()
        # Optional: reserve a pinned cache segment for GLOW window prefetches.
        # (Configured via job.distributed.gpu_cache.window_pinned_max[_entity|_relation])

    def to_device(self, move_optim_data=True):
        if move_optim_data:
            self.optimizer_values = self.optimizer_values.to(self._embeddings.weight.device)

        device = self._embeddings.weight.device
        if device.type == "cuda":
            # pin pull buffers (CPU tensors)
            for i in range(len(self.pull_tensors)):
                self.pull_tensors[i][1] = self.pull_tensors[i][1].pin_memory()

            # create copy stream
            self.copy_stream = torch.cuda.Stream(device=device)

        self._setup_gpu_cache()


    def _init_gpu_cache_config(self):
        enabled = bool(self.config.get("job.distributed.gpu_cache.enable", False))
        if not enabled:
            return

        if "entity" in self.configuration_key:
            max_entries = int(self.config.get("job.distributed.gpu_cache.max_entries_entity", 0))
            pinned_max  = int(self.config.get("job.distributed.gpu_cache.window_pinned_max_entity", 0))
        elif "relation" in self.configuration_key:
            max_entries = int(self.config.get("job.distributed.gpu_cache.max_entries_relation", 0))
            pinned_max  = int(self.config.get("job.distributed.gpu_cache.window_pinned_max_relation", 0))
        else:
            max_entries = int(self.config.get("job.distributed.gpu_cache.max_entries", 0))
            pinned_max  = int(self.config.get("job.distributed.gpu_cache.window_pinned_max", 0))

        self._gpu_cache_max_insert_per_step = int(
            self.config.get("job.distributed.gpu_cache.max_insert_per_step", self._gpu_cache_max_insert_per_step)
        )
        eviction_policy = str(self.config.get("job.distributed.gpu_cache.eviction_policy", "fifo")).lower()
        if eviction_policy not in ("fifo", "lru"):
            eviction_policy = "fifo"
        self._gpu_cache_eviction_policy = eviction_policy
        self._gpu_cache_lru_max_scan = int(self.config.get("job.distributed.gpu_cache.lru_max_scan", 0) or 0)
        self._gpu_cache_pinned_persist = bool(self.config.get("job.distributed.gpu_cache.window_pinned_persist", False))

        if max_entries <= 0:
            return
        self._gpu_cache_enabled = True
        self._gpu_cache_max_entries = max_entries
        self._gpu_cache_pinned_max_entries = max(0, pinned_max)


    def _setup_gpu_cache(self):
        if not self._gpu_cache_enabled:
            return
        device = self._embeddings.weight.device
        if device.type != "cuda":
            return
        if (
            self._gpu_cache_embeddings is not None
            and self._gpu_cache_embeddings.device == device
        ):
            return
        cache_size = self._gpu_cache_max_entries
        if cache_size <= 0:
            return
        if "relation" in self.configuration_key:
            mapping_size = self.dataset.num_relations()
        else:
            mapping_size = self.complete_vocab_size or self.dataset.num_entities()
        mapping_size = max(mapping_size, cache_size)
        self._gpu_cache_id_to_slot = torch.full(
            (mapping_size,), -1, dtype=torch.long, device="cpu"
        )
        self._gpu_cache_ids = torch.full(
            (cache_size,), -1, dtype=torch.long, device="cpu"
        )
        self._gpu_cache_last_access = torch.full(
            (cache_size,), -1, dtype=torch.long, device="cpu"
        )
        self._gpu_cache_access_clock = 0
        self._gpu_cache_embeddings = torch.empty(
            (cache_size, self.dim), device=device
        )
        self._gpu_cache_optimizer = torch.empty(
            (cache_size, self.optimizer_dim), device=self.optimizer_values.device
        )
        pinned_cap = int(getattr(self, "_gpu_cache_pinned_max_entries", 0) or 0)
        self._gpu_cache_pinned_capacity = max(0, min(pinned_cap, cache_size))
        self._gpu_cache_pinned_next_slot = 0
        self._gpu_cache_next_slot = self._gpu_cache_pinned_capacity
        self._gpu_cache_hits = 0
        self._gpu_cache_misses = 0
        if device.type == "cuda":
            with torch.cuda.device(device):
                self._gpu_cache_prefetch_stream = torch.cuda.Stream()
                self._gpu_cache_prefetch_event = torch.cuda.Event()

    def _gpu_cache_lookup(
        self, indexes_cpu: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self._gpu_cache_enabled or self._gpu_cache_id_to_slot is None:
            return None, None
        slots = self._gpu_cache_id_to_slot[indexes_cpu]
        mask = slots >= 0
        if (
            self._gpu_cache_eviction_policy == "lru"
            and self._gpu_cache_last_access is not None
            and mask is not None
            and torch.any(mask)
        ):
            self._gpu_cache_access_clock += 1
            self._gpu_cache_last_access[slots[mask]] = self._gpu_cache_access_clock
        return slots, mask

    def _gpu_cache_select_lru_slots(self, n, start, end):
        if self._gpu_cache_ids is None or self._gpu_cache_last_access is None:
            return None
        if n <= 0 or start >= end:
            return None
        slots = torch.arange(start, end, dtype=torch.long, device="cpu")
        ids = self._gpu_cache_ids[slots]
        empty_mask = ids < 0
        selected = []
        if empty_mask.any():
            empty_slots = slots[empty_mask]
            take = min(int(empty_slots.numel()), n)
            if take > 0:
                selected.append(empty_slots[:take])
                n -= take
        if n <= 0:
            return torch.cat(selected) if selected else None
        candidates = slots[~empty_mask] if empty_mask.any() else slots
        if candidates.numel() == 0:
            return torch.cat(selected) if selected else None
        last_access = self._gpu_cache_last_access[candidates]
        if (
            self._gpu_cache_lru_max_scan
            and candidates.numel() > self._gpu_cache_lru_max_scan
        ):
            perm = torch.randperm(candidates.numel())[: self._gpu_cache_lru_max_scan]
            candidates = candidates[perm]
            last_access = last_access[perm]
        _, idx = torch.topk(last_access, k=min(n, candidates.numel()), largest=False)
        selected.append(candidates[idx])
        return torch.cat(selected) if selected else None

    def _dedupe_cache_inputs(
        self,
        indexes_cpu: torch.Tensor,
        emb_gpu: torch.Tensor,
        opt_gpu: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if indexes_cpu is None or indexes_cpu.numel() <= 1:
            return indexes_cpu, emb_gpu, opt_gpu
        unique_ids, inverse = torch.unique(indexes_cpu, return_inverse=True)
        if unique_ids.numel() == indexes_cpu.numel():
            return indexes_cpu, emb_gpu, opt_gpu
        n = int(indexes_cpu.numel())
        idx = torch.arange(n, dtype=torch.long, device="cpu")
        first = torch.full((unique_ids.numel(),), n, dtype=torch.long, device="cpu")
        try:
            first.scatter_reduce_(0, inverse, idx, reduce="amin", include_self=True)
        except Exception:
            inv_np = inverse.cpu().numpy()
            first_np = np.full(unique_ids.numel(), n, dtype=np.int64)
            for i, g in enumerate(inv_np):
                if i < first_np[g]:
                    first_np[g] = i
            first = torch.from_numpy(first_np)
        first = first[first < n]
        indexes_cpu = indexes_cpu[first]
        emb_gpu = emb_gpu[first.to(emb_gpu.device)]
        if opt_gpu is not None:
            opt_gpu = opt_gpu[first.to(opt_gpu.device)]
        return indexes_cpu, emb_gpu, opt_gpu

    def _gpu_cache_insert(
        self,
        indexes_cpu: torch.Tensor,
        emb_gpu: torch.Tensor,
        opt_gpu: Optional[torch.Tensor],
    ) -> None:
        if (
            not self._gpu_cache_enabled
            or self._gpu_cache_id_to_slot is None
            or self._gpu_cache_ids is None
            or self._gpu_cache_embeddings is None
        ):
            return
        if indexes_cpu.numel() == 0:
            return
        if opt_gpu is None and self.optimizer_dim > 0:
            return
        cache_size = int(self._gpu_cache_max_entries or 0)
        if cache_size <= 0:
            return
        pinned = int(getattr(self, "_gpu_cache_pinned_capacity", 0) or 0)
        usable = cache_size - pinned
        if usable <= 0:
            return
        indexes_cpu, emb_gpu, opt_gpu = self._dedupe_cache_inputs(
            indexes_cpu, emb_gpu, opt_gpu
        )
        if indexes_cpu.numel() == 0:
            return
        if indexes_cpu.numel() > usable:
            indexes_cpu = indexes_cpu[-usable:]
            emb_gpu = emb_gpu[-usable:]
            if opt_gpu is not None:
                opt_gpu = opt_gpu[-usable:]
        n = int(indexes_cpu.numel())
        if self._gpu_cache_eviction_policy == "lru" and self._gpu_cache_last_access is not None:
            slots = self._gpu_cache_select_lru_slots(n, pinned, cache_size)
            if slots is None or slots.numel() == 0:
                return
            slots = slots[:n]
            rel_start = None
        else:
            start = int(self._gpu_cache_next_slot or pinned)
            if start < pinned:
                start = pinned
            rel_start = start - pinned
            slots = (torch.arange(n, device="cpu") + rel_start) % usable + pinned
        evict_ids = self._gpu_cache_ids[slots]
        if evict_ids.numel() > 0:
            valid_evict = evict_ids >= 0
            if valid_evict.any():
                self._gpu_cache_id_to_slot[evict_ids[valid_evict]] = -1
        self._gpu_cache_ids[slots] = indexes_cpu
        self._gpu_cache_id_to_slot[indexes_cpu] = slots
        slots_device = slots.to(self._gpu_cache_embeddings.device, non_blocking=True)
        self._gpu_cache_embeddings[slots_device] = emb_gpu
        if (
            self._gpu_cache_optimizer is not None
            and self.optimizer_dim > 0
            and opt_gpu is not None
        ):
            slots_opt = slots.to(self._gpu_cache_optimizer.device, non_blocking=True)
            self._gpu_cache_optimizer[slots_opt] = opt_gpu
        if self._gpu_cache_eviction_policy == "lru" and self._gpu_cache_last_access is not None:
            self._gpu_cache_access_clock += 1
            self._gpu_cache_last_access[slots] = self._gpu_cache_access_clock
        else:
            self._gpu_cache_next_slot = int(((rel_start + n) % usable) + pinned)

    def _gpu_cache_clear_slots(self, slots_cpu: torch.Tensor) -> None:
        if (
            not self._gpu_cache_enabled
            or self._gpu_cache_id_to_slot is None
            or self._gpu_cache_ids is None
        ):
            return
        if slots_cpu is None or slots_cpu.numel() == 0:
            return
        evict_ids = self._gpu_cache_ids[slots_cpu]
        if evict_ids.numel() > 0:
            valid_evict = evict_ids >= 0
            if valid_evict.any():
                self._gpu_cache_id_to_slot[evict_ids[valid_evict]] = -1
        self._gpu_cache_ids[slots_cpu] = -1
        if self._gpu_cache_last_access is not None:
            self._gpu_cache_last_access[slots_cpu] = -1

    def _gpu_cache_clear_pinned(self) -> None:
        # Clear only what we pinned last time, not the whole pinned region.
        pinned = int(getattr(self, "_gpu_cache_pinned_capacity", 0) or 0)
        if pinned <= 0:
            return
        last_ids = getattr(self, "_gpu_cache_last_pinned_ids", None)
        if last_ids is None or last_ids.numel() == 0:
            return
        n = int(last_ids.numel())
        slots = torch.arange(n, dtype=torch.long, device="cpu")
        try:
            self._gpu_cache_id_to_slot[last_ids] = -1
        except Exception:
            pass
        self._gpu_cache_ids[slots] = -1
        if self._gpu_cache_last_access is not None:
            self._gpu_cache_last_access[slots] = -1
        self._gpu_cache_last_pinned_ids = None

    def _gpu_cache_insert_pinned(
        self,
        indexes_cpu: torch.Tensor,
        emb_gpu: torch.Tensor,
        opt_gpu: Optional[torch.Tensor],
    ) -> None:
        pinned = int(getattr(self, "_gpu_cache_pinned_capacity", 0) or 0)
        if pinned <= 0 or indexes_cpu is None:
            return
        n = int(indexes_cpu.numel())
        if n <= 0:
            return
        if n > pinned:
            indexes_cpu = indexes_cpu[:pinned]
            emb_gpu = emb_gpu[:pinned]
            if opt_gpu is not None:
                opt_gpu = opt_gpu[:pinned]
            n = int(indexes_cpu.numel())
        if n <= 0:
            return
        if opt_gpu is None and self.optimizer_dim > 0:
            return
        indexes_cpu, emb_gpu, opt_gpu = self._dedupe_cache_inputs(
            indexes_cpu, emb_gpu, opt_gpu
        )
        if indexes_cpu.numel() == 0:
            return
        n = int(indexes_cpu.numel())
        if not self._gpu_cache_pinned_persist:
            self._gpu_cache_clear_pinned()
            slots = torch.arange(n, dtype=torch.long, device="cpu")
        else:
            if self._gpu_cache_eviction_policy == "lru" and self._gpu_cache_last_access is not None:
                slots = self._gpu_cache_select_lru_slots(n, 0, pinned)
                if slots is None or slots.numel() == 0:
                    return
                slots = slots[:n]
            else:
                start = int(self._gpu_cache_pinned_next_slot or 0)
                slots = (torch.arange(n, device="cpu") + start) % pinned
                self._gpu_cache_pinned_next_slot = int((start + n) % pinned)
        evict_ids = self._gpu_cache_ids[slots]
        if evict_ids.numel() > 0:
            valid_evict = evict_ids >= 0
            if valid_evict.any():
                self._gpu_cache_id_to_slot[evict_ids[valid_evict]] = -1
        self._gpu_cache_ids[slots] = indexes_cpu
        self._gpu_cache_id_to_slot[indexes_cpu] = slots
        slots_device = slots.to(self._gpu_cache_embeddings.device, non_blocking=True)
        self._gpu_cache_embeddings[slots_device] = emb_gpu
        if opt_gpu is not None and self._gpu_cache_optimizer is not None:
            slots_opt = slots.to(self._gpu_cache_optimizer.device, non_blocking=True)
            self._gpu_cache_optimizer[slots_opt] = opt_gpu
        if self._gpu_cache_eviction_policy == "lru" and self._gpu_cache_last_access is not None:
            self._gpu_cache_access_clock += 1
            self._gpu_cache_last_access[slots] = self._gpu_cache_access_clock
        if not self._gpu_cache_pinned_persist:
            self._gpu_cache_last_pinned_ids = indexes_cpu.clone()

    def _gpu_cache_complete_prefetch(self):
        entry = getattr(self, "_gpu_cache_prefetch_pending", None)
        if entry is None:
            return
        self._gpu_cache_prefetch_pending = None
        self.parameter_client.wait(entry.get("pull_future"))
        pull_tensor = entry.get("pull_tensor")
        pull_tensor_index = entry.get("pull_tensor_index")
        idx_cpu = entry.get("indexes_cpu")
        if pull_tensor is None or idx_cpu is None or idx_cpu.numel() == 0:
            if pull_tensor_index is not None:
                self.pull_tensors[pull_tensor_index][0] = True
            return
        pulled_embeddings = pull_tensor[:, : self.dim]
        pulled_optim_values = pull_tensor[:, self.dim : (self.dim + self.optimizer_dim)]
        device = self._embeddings.weight.device
        if device.type == "cuda" and self._gpu_cache_prefetch_stream is not None:
            with torch.cuda.stream(self._gpu_cache_prefetch_stream):
                emb_gpu = pulled_embeddings.to(device, non_blocking=True)
                opt_gpu = pulled_optim_values.to(
                    self.optimizer_values.device, non_blocking=True
                )
                self._gpu_cache_insert_pinned(idx_cpu, emb_gpu, opt_gpu)
            if self._gpu_cache_prefetch_event is not None:
                self._gpu_cache_prefetch_stream.record_event(
                    self._gpu_cache_prefetch_event
                )
        else:
            emb_gpu = pulled_embeddings.to(device, non_blocking=True)
            opt_gpu = pulled_optim_values.to(
                self.optimizer_values.device, non_blocking=True
            )
            self._gpu_cache_insert_pinned(idx_cpu, emb_gpu, opt_gpu)
        if pull_tensor_index is not None:
            self.pull_tensors[pull_tensor_index][0] = True

    def prefetch_window_pinned(
        self,
        indexes: Tensor,
        make_unique: bool = True,
    ) -> None:
        """Prefetch (and pin) a window's ids into the GPU cache."""
        if not self._gpu_cache_enabled:
            return
        if int(getattr(self, "_gpu_cache_pinned_max_entries", 0) or 0) <= 0:
            # Avoid cache setup/prefetch overhead when pinned cache is disabled.
            return
        self._setup_gpu_cache()
        pinned = int(getattr(self, "_gpu_cache_pinned_capacity", 0) or 0)
        if pinned <= 0:
            return
        # Complete the previous async prefetch to keep the pipeline moving.
        self._gpu_cache_complete_prefetch()
        if indexes is None:
            return
        idx_cpu = indexes.detach()
        if idx_cpu.device.type != "cpu":
            idx_cpu = idx_cpu.cpu()
        idx_cpu = idx_cpu.long()
        if idx_cpu.numel() == 0:
            return
        if make_unique:
            idx_cpu = torch.unique(idx_cpu)
        if idx_cpu.numel() == 0:
            return
        try:
            _, hit_mask = self._gpu_cache_lookup(idx_cpu)
            if hit_mask is not None:
                miss_mask = ~hit_mask
                if torch.any(miss_mask):
                    idx_cpu = idx_cpu[miss_mask]
                else:
                    return
        except Exception:
            pass
        if idx_cpu.numel() > pinned:
            idx_cpu = idx_cpu[:pinned]
        n = int(idx_cpu.numel())
        if n <= 0:
            return

        pull = self._get_free_pull_tensor(allow_none=True)
        if pull is None:
            # Best-effort: if no free pull tensor is available, skip prefetch.
            # Allocating + pinning huge CPU tensors here causes major slowdowns.
            return
        pull_tensor_index, pull_tensor_full = pull
        pull_tensor = pull_tensor_full[:n]
        pull_future = self.parameter_client.pull(
            idx_cpu + self.lapse_offset, pull_tensor, asynchronous=True
        )
        self._gpu_cache_prefetch_pending = {
            "indexes_cpu": idx_cpu,
            "pull_tensor": pull_tensor,
            "pull_future": pull_future,
            "pull_tensor_index": pull_tensor_index,
        }

    def get_and_reset_gpu_cache_stats(self):
        if not self._gpu_cache_enabled:
            return None
        stats = {
            "hits": int(self._gpu_cache_hits),
            "misses": int(self._gpu_cache_misses),
        }
        self._gpu_cache_hits = 0
        self._gpu_cache_misses = 0
        return stats

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
                    torch.zeros(
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
        self._pull_embeddings(torch.arange(self.vocab_size))

    def set_embeddings(self):
        # storing set_indexes and set_tensors in self to keep them alive until async
        #  set is finished
        self.set_indexes = (self.pulled_ids + self.lapse_offset).cpu()
        num_pulled = len(self.set_indexes)
        # move tensors to cpu before cat to reduce gpu memory usage
        if self.unnecessary_dim > 0:
            self.set_tensor = torch.cat(
                (
                    self._embeddings.weight[:num_pulled].detach().cpu(),
                    self.optimizer_values[:num_pulled].cpu(),
                    torch.zeros(
                        (num_pulled, self.unnecessary_dim),
                        dtype=self.optimizer_values.dtype,
                        device="cpu",
                    ),
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

    def _get_free_pull_tensor(self, allow_none: bool = False):
        for i, (free, pull_tensor) in enumerate(self.pull_tensors):
            if free:
                self.pull_tensors[i][0] = False
                return i, pull_tensor
        if allow_none:
            return None
        raise RuntimeError(
            "No free pull tensor; increase pool size or reduce prefetch depth."
        )

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
        indexes_cpu = indexes.detach().cpu().long()
        if indexes_cpu.numel() == 0:
            return
        cold_mask = None
        cold_rows = None
        cold_indexes_cpu = None
        if getattr(self, "_gpu_cache_enabled", False):
            # Skip/trim pre-pull when ids are already cached to avoid redundant PS pulls.
            self._setup_gpu_cache()
            if self._gpu_cache_id_to_slot is not None:
                try:
                    _, cache_mask = self._gpu_cache_lookup(indexes_cpu)
                    if cache_mask is not None:
                        if torch.all(cache_mask):
                            return
                        cold_mask = ~cache_mask
                        if torch.any(cold_mask):
                            cold_rows = torch.arange(
                                indexes_cpu.numel(), dtype=torch.long, device="cpu"
                            )[cold_mask]
                            cold_indexes_cpu = indexes_cpu[cold_mask]
                except Exception:
                    cold_mask = None
                    cold_rows = None
                    cold_indexes_cpu = None
        full_len = int(indexes_cpu.numel())
        if cold_indexes_cpu is not None:
            pull_indexes = (cold_indexes_cpu + self.lapse_offset).cpu()
            pull_len = int(cold_indexes_cpu.numel())
        else:
            pull_indexes = (indexes_cpu + self.lapse_offset).cpu()
            pull_len = full_len
        pull = self._get_free_pull_tensor(allow_none=True)
        if pull is None:
            # Best-effort prefetch: skip if no free pull tensor is available.
            return
        pull_tensor_index, pull_tensor = pull
        pull_tensor = pull_tensor[:pull_len]
        pull_future = self.parameter_client.pull(
            pull_indexes, pull_tensor, asynchronous=True
        )
        self.pre_pulled.append(
            {
                "indexes": indexes,
                "num_indexes": full_len,
                "pull_len": pull_len,
                "pull_indexes": pull_indexes,
                "pull_tensor": pull_tensor,
                "pull_future": pull_future,
                "pull_tensor_index": pull_tensor_index,
                "cold_mask": cold_mask,
                "cold_rows": cold_rows,
                "cold_indexes_cpu": cold_indexes_cpu,
            }
        )

    def pre_pulled_to_device(self):
        # Intentional no-op: keep pre-pulled tensors in pinned CPU memory and
        # copy directly into destination buffers on consumption.
        return

    @torch.no_grad()
    def _pull_embeddings(self, indexes):
        cpu_gpu_time = 0.0
        pull_time = 0.0
        device = self._embeddings.weight.device
        if self._gpu_cache_enabled:
            self._setup_gpu_cache()
        use_prefetch = len(self.pre_pulled) > 0
        if use_prefetch:
            # todo: add workaround for relations here as well
            # todo: clean up this method
            pre_pulled = self.pre_pulled.popleft()
            indexes = pre_pulled["indexes"]
            len_indexes = pre_pulled.get("num_indexes", len(indexes))
            self.pulled_ids = indexes
            output_rows = torch.arange(len_indexes, dtype=torch.long)
            indexes_cpu = indexes.detach().cpu().long()
            self.parameter_client.wait(pre_pulled["pull_future"])
            self.local_to_lapse_mapper[:len_indexes] = indexes_cpu + self.lapse_offset
            pre_pulled_tensor = pre_pulled["pull_tensor"]
            cold_rows = pre_pulled.get("cold_rows")
            cold_indexes_cpu = pre_pulled.get("cold_indexes_cpu")

            pulled_embeddings, pulled_optim_values, _ = torch.split(
                pre_pulled_tensor,
                [self.dim, self.optimizer_dim, self.unnecessary_dim],
                dim=1,
            )

            cpu_gpu_time -= time.time()
            if cold_rows is None:
                self._embeddings.weight.data[:len_indexes].copy_(
                    pulled_embeddings, non_blocking=True
                )
                if self.optimizer_dim > 0:
                    self.optimizer_values[:len_indexes].copy_(
                        pulled_optim_values, non_blocking=True
                    )
                cpu_gpu_time += time.time()
            else:
                emb_device = self._embeddings.weight.device
                opt_device = self.optimizer_values.device
                cold_rows_emb = cold_rows.to(emb_device)
                cold_rows_opt = cold_rows.to(opt_device)
                if emb_device.type == "cuda":
                    pulled_embeddings_gpu = pulled_embeddings.to(
                        emb_device, non_blocking=True
                    )
                else:
                    pulled_embeddings_gpu = pulled_embeddings
                if self.optimizer_dim > 0:
                    if opt_device.type == "cuda":
                        pulled_optim_gpu = pulled_optim_values.to(
                            opt_device, non_blocking=True
                        )
                    else:
                        pulled_optim_gpu = pulled_optim_values
                else:
                    pulled_optim_gpu = None
                self._embeddings.weight.data.index_copy_(
                    0, cold_rows_emb, pulled_embeddings_gpu
                )
                if self.optimizer_dim > 0:
                    self.optimizer_values.index_copy_(
                        0, cold_rows_opt, pulled_optim_gpu
                    )
                cpu_gpu_time += time.time()

                # Fill cached rows from GPU cache (hot IDs) if enabled
                if getattr(self, "_gpu_cache_enabled", False):
                    self._setup_gpu_cache()
                    if self._gpu_cache_id_to_slot is not None:
                        cache_slots, cache_mask = self._gpu_cache_lookup(indexes_cpu)
                        if cache_mask is not None and cache_mask.any():
                            cold_mask = pre_pulled.get("cold_mask")
                            if cold_mask is None:
                                cold_mask = torch.zeros_like(
                                    cache_mask, dtype=torch.bool
                                )

                            cached_mask = cache_mask & ~cold_mask
                            cached_rows = output_rows[cached_mask]
                            if cached_rows.numel() > 0:
                                if (
                                    device.type == "cuda"
                                    and self._gpu_cache_prefetch_event is not None
                                ):
                                    torch.cuda.current_stream(device).wait_event(
                                        self._gpu_cache_prefetch_event
                                    )

                                cached_rows_device = cached_rows.to(device)
                                slots_device = cache_slots[cached_mask].to(device)
                                self._embeddings.weight.data[cached_rows_device] = (
                                    self._gpu_cache_embeddings[slots_device]
                                )
                                self.optimizer_values[cached_rows_device] = (
                                    self._gpu_cache_optimizer[slots_device]
                                )
                                self._gpu_cache_hits += int(cached_rows.numel())

                        # Optionally insert cold pulls into GPU cache
                        if cold_indexes_cpu is not None and cold_indexes_cpu.numel() > 0:
                            self._gpu_cache_misses += int(cold_rows.numel())
                            cold_n = int(cold_rows.numel())
                            max_insert = int(self._gpu_cache_max_insert_per_step or 0)
                            insert_n = cold_n
                            if max_insert > 0 and cold_n > max_insert:
                                insert_n = max_insert
                            if insert_n > 0:
                                if insert_n < cold_n:
                                    cold_indexes_cpu = cold_indexes_cpu[:insert_n]
                                    cold_rows_emb = cold_rows_emb[:insert_n]
                                    cold_rows_opt = cold_rows_opt[:insert_n]
                                emb_cache = (
                                    pulled_embeddings_gpu[:insert_n]
                                    if device.type == "cuda"
                                    else self._embeddings.weight.data[cold_rows_emb]
                                )
                                if self.optimizer_dim > 0:
                                    opt_cache = (
                                        pulled_optim_gpu[:insert_n]
                                        if self.optimizer_values.device.type == "cuda"
                                        else self.optimizer_values[cold_rows_opt]
                                    )
                                else:
                                    opt_cache = None
                                self._gpu_cache_insert(
                                    cold_indexes_cpu,
                                    emb_cache,
                                    opt_cache,
                                )
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

        cache_mask = None
        if self._gpu_cache_enabled and self._gpu_cache_id_to_slot is not None:
            cache_slots, cache_mask = self._gpu_cache_lookup(indexes_cpu)
            if cache_mask is not None and cache_mask.any():
                if (
                    device.type == "cuda"
                    and self._gpu_cache_prefetch_event is not None
                ):
                    torch.cuda.current_stream(device).wait_event(
                        self._gpu_cache_prefetch_event
                    )
                if hot_mask is not None:
                    cache_mask = cache_mask & cold_mask
                cache_rows = output_rows[cache_mask]
                if cache_rows.numel() > 0:
                    cache_rows_device = cache_rows.to(device)
                    slots_device = cache_slots[cache_mask].to(device)
                    self._embeddings.weight.data[cache_rows_device] = (
                        self._gpu_cache_embeddings[slots_device]
                    )
                    self.optimizer_values[cache_rows_device] = (
                        self._gpu_cache_optimizer[slots_device]
                    )
                    self._gpu_cache_hits += int(cache_rows.numel())
        if cache_mask is None:
            cache_mask = torch.zeros(len_indexes, dtype=torch.bool)

        cold_mask = cold_mask & ~cache_mask
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
            pulled_embeddings_gpu = pulled_embeddings.to(device)
            pulled_optim_gpu = pulled_optim_values.to(self.optimizer_values.device)
            self._embeddings.weight.data[cold_rows_device] = pulled_embeddings_gpu
            self.optimizer_values[cold_rows_device] = pulled_optim_gpu
            cpu_gpu_time += time.time()
            if self._gpu_cache_enabled:
                self._gpu_cache_misses += int(num_cold)
                cold_n = int(num_cold)
                max_insert = int(self._gpu_cache_max_insert_per_step or 0)
                insert_n = cold_n
                if max_insert > 0 and cold_n > max_insert:
                    insert_n = max_insert
                if insert_n > 0:
                    if insert_n < cold_n:
                        cold_indexes = cold_indexes[:insert_n]
                        pulled_embeddings_gpu = pulled_embeddings_gpu[:insert_n]
                        pulled_optim_gpu = pulled_optim_gpu[:insert_n]
                    self._gpu_cache_insert(
                        cold_indexes, pulled_embeddings_gpu, pulled_optim_gpu
                    )

        self.local_to_lapse_mapper[output_rows] = indexes_cpu + self.lapse_offset
        return pull_time, cpu_gpu_time

    def localize(self, indexes: Tensor, asynchronous=False, make_unique=False):
        """
        Hint/prefetch rows that are likely needed soon.

        NOTE:
        - In shared/torch parameter server modes, ParameterClient.localize() is a no-op.
        - To make Glow window prefetch effective on a single machine, we warm the optional
          GPU cache here by explicitly pulling rows and inserting them into the cache.
        """
        if make_unique:
            indexes = torch.unique(indexes)
        if indexes is None or indexes.numel() == 0:
            return

        # Best-effort call (may be a no-op depending on PS backend).
        try:
            self.parameter_client.localize((indexes + self.lapse_offset).cpu(), asynchronous)
        except Exception:
            pass

        # If GPU cache isn't enabled, nothing more to do.
        if not getattr(self, "_gpu_cache_enabled", False):
            return
        # Ensure cache tables are initialized before attempting inserts.
        self._setup_gpu_cache()
        if self._gpu_cache_id_to_slot is None:
            return

        # Work on CPU ids (RAW ids without lapse_offset) for cache bookkeeping.
        raw_ids_cpu = indexes.detach().to("cpu", dtype=torch.long)

        # Avoid refetching rows already in cache.
        try:
            _, cache_mask = self._gpu_cache_lookup(raw_ids_cpu)
            if cache_mask is not None and torch.any(cache_mask):
                raw_ids_cpu = raw_ids_cpu[~cache_mask]
            if raw_ids_cpu.numel() == 0:
                return
        except Exception:
            # If anything goes wrong, fall back to prefetching all provided ids.
            pass

        device = self._embeddings.weight.device
        embed_cols = int(self.dim)
        opt_cols = int(self.optimizer_dim)

        # Pull in chunks to keep temporary tensors bounded.
        chunk_size = 65536
        for start in range(0, raw_ids_cpu.numel(), chunk_size):
            chunk_raw = raw_ids_cpu[start : start + chunk_size]
            if chunk_raw.numel() == 0:
                continue

            # Keys into the PS include the lapse offset.
            chunk_keys = (chunk_raw + self.lapse_offset).to(dtype=torch.long)

            # Reuse the pinned CPU pull tensor.
            pull_tensor = self.pull_tensors[0][1][: chunk_keys.numel()]

            # Pull is CPU-side; we keep it synchronous here because the warmup is typically
            # called at window boundaries. (Async would need completion tracking.)
            wait_value = self.parameter_client.pull(chunk_keys, pull_tensor, asynchronous=False)
            if hasattr(wait_value, "wait"):
                wait_value.wait()

            # Split pull payload into update + optimizer state.
            emb_cpu = pull_tensor[:, :embed_cols].contiguous()
            if opt_cols > 0:
                opt_cpu = pull_tensor[
                    :, embed_cols : embed_cols + opt_cols
                ].contiguous()
            else:
                opt_cpu = torch.empty(
                    (chunk_keys.numel(), 0), dtype=emb_cpu.dtype
                )

            # Move to GPU and insert into cache (use prefetch stream if available).
            if device.type == "cuda" and self._gpu_cache_prefetch_stream is not None:
                with torch.cuda.stream(self._gpu_cache_prefetch_stream):
                    emb_gpu = emb_cpu.to(device, non_blocking=True)
                    opt_gpu = opt_cpu.to(
                        self.optimizer_values.device, non_blocking=True
                    )
                    self._gpu_cache_insert(chunk_raw, emb_gpu, opt_gpu)
                if self._gpu_cache_prefetch_event is not None:
                    self._gpu_cache_prefetch_stream.record_event(
                        self._gpu_cache_prefetch_event
                    )
            else:
                emb_gpu = emb_cpu.to(device, non_blocking=True)
                opt_gpu = opt_cpu.to(
                    self.optimizer_values.device, non_blocking=True
                )
                self._gpu_cache_insert(chunk_raw, emb_gpu, opt_gpu)

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
