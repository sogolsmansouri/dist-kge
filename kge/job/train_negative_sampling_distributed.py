import time
import shutil
import torch
import torch.utils.data
import numpy as np
import math
import gc
import os
import itertools
from pathlib import Path

from collections import defaultdict, deque
from typing import Dict, Any, Optional

from kge import Config
from kge.job import Job
from kge.job.train import TrainingJob, _generate_worker_init_fn
from kge.job.train_negative_sampling import TrainingJobNegativeSampling
from kge.model import KgeModel
from kge.util import KgeOptimizer
from kge.util.dist_adagrad import DistAdagrad
from kge.job.trace import format_trace_entry
from kge.distributed.work_scheduler import SchedulerClient
from kge.distributed.misc import get_min_rank
from kge.distributed.partition_stager import PartitionStager

SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]


class NumberDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, dataset):
        self.samples = list(range(num_samples))
        self.dataset = dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return idx
        # return self.dataset[self.samples[idx], :].long()

    def set_samples(self, samples):
        self.samples = samples


class InfiniteSequentialSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        super(InfiniteSequentialSampler, self).__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return itertools.count(start=0, step=1)

    def __len__(self):
        return len(self.data_source)


class BatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        triples,
        batch_size,
        shuffle=True,
        materialize=False,
        materialize_device=None,
    ):
        self.triples = triples
        # shared buffers used when we need to reshuffle samples
        self._samples_buffer = (
            torch.empty([len(triples)], dtype=torch.long, requires_grad=False)
            .share_memory_()
        )
        self.samples = self._samples_buffer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.materialize = materialize
        self.materialize_device = materialize_device
        self.materialized_triples_device = None
        self.materialized_triples = None
        self.materialized_size = 0
        self.materialization_events = (
            torch.zeros([1], dtype=torch.long, requires_grad=False).share_memory_()
        )
        self.num_samples = (
            torch.full([1], -1, dtype=torch.int, requires_grad=False).share_memory_()
        )
        self.epoch = (
            torch.full([1], -1, dtype=torch.int, requires_grad=False).share_memory_()
        )
        self.partition_id = (
            torch.full([1], -1, dtype=torch.int, requires_grad=False).share_memory_()
        )
        self._materialize_device = None
        self.partition_stager = None
        self._staged_partition = None
        self._staged_version = None
        self._staged_local_ids = False
        if self.materialize and self.materialize_device is not None:
            try:
                device_obj = (
                    self.materialize_device
                    if isinstance(self.materialize_device, torch.device)
                    else torch.device(self.materialize_device)
                )
            except (TypeError, RuntimeError, ValueError):
                device_obj = None
            if device_obj is not None and device_obj.type == "cuda":
                self.partition_stager = PartitionStager(device=device_obj)
                self._materialize_device = device_obj
        self._debug_stats = defaultdict(int)

    def __len__(self):
        if self.num_samples.item() <= 0:
            return 0
        return self.get_real_len()

    def get_real_len(self):
        return math.ceil(self.num_samples.item() / self.batch_size)

    def __getitem__(self, idx):
        """Gets a complete batch based on an idx"""
        # we are iterating with a infinite sampler. Get the actual batch index
        # with modulo
        actual_idx = idx % len(self)
        start = actual_idx * self.batch_size
        stop = min((actual_idx + 1) * (self.batch_size), self.num_samples.item())
        if start >= stop:
            print(idx, self.num_samples.item(), start, stop, len(self))
            return None
        return (
            self.samples[start:stop].clone().long(),
            self.epoch.item(),
            self.partition_id.item(),
        )

    def _ensure_materialized_buffer(self, size: int):
        if self.partition_stager is not None:
            return
        if (
            self.materialized_triples is None
            or self.materialized_triples.size(0) < size
        ):
            self.materialized_triples = (
                torch.empty(
                    (size, self.triples.size(1)),
                    dtype=self.triples.dtype,
                    requires_grad=False,
                )
                .share_memory_()
            )
        if (
            self.materialize_device
            and self.materialize_device.startswith("cuda")
            and (
                self.materialized_triples_device is None
                or self.materialized_triples_device.size(0) < size
            )
        ):
            self.materialized_triples_device = torch.empty(
                (size, self.triples.size(1)),
                dtype=self.triples.dtype,
                device=self.materialize_device,
                requires_grad=False,
            )

    def set_samples(
        self,
        samples: torch.Tensor,
        epoch,
        partition_id,
        partition_version=None,
        entity_mapper=None,
        relation_mapper=None,
        stage_local_ids: bool = False,
    ):
        samples = samples.to(
            dtype=torch.long, device=self._samples_buffer.device, non_blocking=True
        )
        if self.shuffle:
            perm = torch.randperm(samples.size(0), device=samples.device)
            samples = samples.index_select(0, perm)
        self._staged_partition = None
        self._staged_version = None
        self._staged_local_ids = bool(stage_local_ids)
        if self.materialize:
            self._ensure_materialized_buffer(len(samples))
            dataset_indices = samples.to(self.triples.device)
            partition_triples = self.triples.index_select(0, dataset_indices)
            if partition_triples.dtype != torch.long:
                partition_triples = partition_triples.to(dtype=torch.long)
            if stage_local_ids and entity_mapper is not None:
                partition_triples[:, S] = entity_mapper[partition_triples[:, S]]
                partition_triples[:, O] = entity_mapper[partition_triples[:, O]]
            if stage_local_ids and relation_mapper is not None:
                partition_triples[:, P] = relation_mapper[partition_triples[:, P]]
            if self.partition_stager is not None:
                version = (
                    int(partition_version)
                    if partition_version is not None
                    else 0
                )
                stage = self.partition_stager.stage(
                    partition_id=partition_id, version=version, triples=partition_triples
                )
                self._staged_partition = stage
                self._staged_version = version
                self.materialized_triples = stage.host_view
                self.materialized_triples_device = stage.device_view
            else:
                self.materialized_triples[: len(samples)] = partition_triples.cpu()
                if self.materialized_triples_device is not None:
                    self.materialized_triples_device[: len(samples)] = partition_triples.to(
                        self.materialized_triples_device.device, non_blocking=True
                    )
            self.materialized_size = len(samples)
            self.materialization_events[0] += 1
            self.samples = self._samples_buffer
            self.samples[: len(samples)] = torch.arange(
                len(samples), dtype=torch.long, device=self.samples.device
            )
            partition_triples = None
        elif self.shuffle:
            self.samples = self._samples_buffer
            self.samples[: len(samples)] = samples
            self.materialized_triples = None
            self.materialized_size = 0
            self.materialized_triples_device = None
            self._staged_partition = None
        else:
            if samples.device.type != "cpu":
                samples = samples.cpu()
            if not samples.is_shared():
                samples = samples.contiguous().share_memory_()
            self.samples = samples
            self.materialized_triples = None
            self.materialized_size = 0
            self.materialized_triples_device = None
            self._staged_partition = None
        self.num_samples[0] = len(samples)
        self.epoch[0] = epoch
        self.partition_id[0] = partition_id
        self._debug_stats.clear()

    def fetch_triples(self, sample_indices: torch.Tensor) -> torch.Tensor:
        sample_indices = sample_indices.long()
        staged_host = self._slice_staged_view(sample_indices, on_device=False)
        if staged_host is not None:
            self._debug_stats["materialized_fetches"] += 1
            return staged_host
        if (
            self.materialize
            and self.materialized_triples is not None
            and self.materialized_size >= sample_indices.max().item() + 1
        ):
            self._debug_stats["materialized_fetches"] += 1
            return self.materialized_triples[sample_indices, :]
        self._debug_stats["corpus_fetches"] += 1
        return self.triples[sample_indices, :]

    def fetch_triples_device(self, sample_indices: torch.Tensor):
        staged_device = self._slice_staged_view(sample_indices, on_device=True)
        if staged_device is not None:
            return staged_device
        return None

    def write_triples_to_device(
        self, sample_indices: torch.Tensor, mapped_triples: torch.Tensor, device
    ):
        if (
            self.materialized_triples_device is None
            or self.materialized_size <= 0
            or sample_indices.numel() == 0
        ):
            return None
        start = int(sample_indices[0].item())
        expected = torch.arange(
            start, start + sample_indices.numel(), device=sample_indices.device
        )
        if not torch.equal(sample_indices, expected):
            return None
        stop = start + mapped_triples.size(0)
        if stop > self.materialized_size:
            return None
        if (
            self._staged_local_ids
            and self.partition_stager is not None
            and self._staged_partition is not None
        ):
            device_view = self.materialized_triples_device[start:stop]
        else:
            device_view = self.materialized_triples_device[start:stop]
            device_view.copy_(mapped_triples.to(device, non_blocking=True))
        self._debug_stats["device_writes"] += 1
        return device_view

    def materialization_count(self) -> int:
        return int(self.materialization_events[0].item())

    def collect_debug_stats(self):
        stats = dict(self._debug_stats)
        self._debug_stats.clear()
        return stats

    def _slice_staged_view(
        self, sample_indices: torch.Tensor, on_device: bool
    ) -> Optional[torch.Tensor]:
        stage = self._staged_partition
        if stage is None:
            return None
        if sample_indices.numel() == 0:
            return None
        indices = sample_indices
        if indices.device.type != "cpu":
            indices = indices.to("cpu")
        start = int(indices[0].item())
        expected = torch.arange(
            start, start + indices.numel(), device=indices.device, dtype=indices.dtype
        )
        if not torch.equal(indices, expected):
            return None
        stop = start + indices.numel()
        if stop > stage.num_triples:
            return None
        base = stage.device_view if on_device else stage.host_view
        if base is None:
            return None
        return base[start:stop]


class MaterializedBatchIterator:
    def __init__(self, dataset: BatchDataset, collate_fn):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.position = 0

    def reset(self):
        self.position = 0

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        if self.dataset.num_samples.item() <= 0:
            raise StopIteration
        total_batches = len(self.dataset)
        if total_batches <= 0 or self.position >= total_batches:
            raise StopIteration
        entry = self.dataset[self.position]
        self.position += 1
        return self.collate_fn([entry])


class TrainingJobNegativeSamplingDistributed(TrainingJobNegativeSampling):
    def __init__(
        self,
        config,
        dataset,
        parent_job=None,
        model=None,
        optimizer=None,
        forward_only=False,
        parameter_client=None,
        work_scheduler_client=None,
        init_for_load_only=False,
    ):
        self._non_blocking_transfer = False
        self.config = config
        self.parameter_client = parameter_client
        self.min_rank = get_min_rank(config)

        job_device = config.get("job.device")
        enable_prefetch = (
            isinstance(job_device, str) and job_device.startswith("cuda")
        )
        # NOTE: Auto-prefetch/auto-num_workers disabled for correctness.
        # Set job.distributed.entity_pre_pull / relation_pre_pull / train.num_workers explicitly if needed.
        configured_workers = int(self.config.get("train.num_workers"))
        self._effective_num_workers = configured_workers
        if self.config.get("job.distributed.materialize_partition_batches"):
            if self._effective_num_workers != 0:
                self._effective_num_workers = 0
                self.config.log(
                    "Disabled train.num_workers because materialized partitions are streamed directly."
                )

        if work_scheduler_client is None:
            if self.config.get("job.distributed.single_process"):
                from kge.distributed.work_scheduler import LocalSchedulerClient

                self.work_scheduler_client = LocalSchedulerClient(config, dataset)
            else:
                self.work_scheduler_client = SchedulerClient(config)
        else:
            self.work_scheduler_client = work_scheduler_client
        (
            max_partition_entities,
            max_partition_relations,
        ) = self.work_scheduler_client.get_init_info()
        if model is None:
            model: KgeModel = KgeModel.create(
                config,
                dataset,
                parameter_client=parameter_client,
                max_partition_entities=max_partition_entities,
            )
        model.get_s_embedder().to_device()
        model.get_p_embedder().to_device()
        lapse_indexes = [
            torch.arange(dataset.num_entities(), dtype=torch.int),
            torch.arange(dataset.num_relations(), dtype=torch.int)
            + dataset.num_entities(),
        ]
        if optimizer is None:
            optimizer = KgeOptimizer.create(
                config,
                model,
                parameter_client=parameter_client,
                lapse_indexes=lapse_indexes,
            )
        # barrier to wait for loading of pretrained embeddings
        self.parameter_client.barrier()
        super().__init__(
            config,
            dataset,
            parent_job,
            model=model,
            optimizer=optimizer,
            forward_only=forward_only,
            parameter_client=parameter_client,
            work_scheduler_client=work_scheduler_client,
        )
        self.type_str = "negative_sampling"
        self._map_ids_on_gpu = bool(
            self.config.get("job.distributed.map_ids_on_gpu")
        )
        self._map_ids_device = self._resolve_map_ids_device()
        self._unique_on_gpu = bool(
            self.config.get("job.distributed.unique_on_gpu")
        )
        self._entity_partition_mapper_device = None
        self._relation_partition_mapper_device = None
        self._gpu_sampling_logged = False
        self.entity_localize = self.config.get("job.distributed.entity_localize")
        self.relation_localize = self.config.get("job.distributed.relation_localize")
        self.entity_partition_localized = False
        self.relation_partition_localized = False
        self.local_entities = None
        self.entity_async_write_back = self.config.get(
            "job.distributed.entity_async_write_back"
        )
        self.relation_async_write_back = self.config.get(
            "job.distributed.relation_async_write_back"
        )
        self.entity_sync_level = self.config.get("job.distributed.entity_sync_level")
        self.relation_sync_level = self.config.get(
            "job.distributed.relation_sync_level"
        )
        if (
            self.entity_sync_level == "partition"
            and self.relation_sync_level != "partition"
        ):
            self.config.log(
                "Relation sync level is not partition while entity sync is "
                "partition; relation updates will use the batch/PS path. "
                "If you want relation partitioning, set "
                "job.distributed.relation_sync_level=partition before starting "
                "the scheduler."
            )
        # DistAdagrad pushes sparse updates per step only for batch-level sync.
        # When using partition-level sync, we must write back the pulled embeddings.
        has_partition_sync = (
            self.entity_sync_level == "partition"
            or self.relation_sync_level == "partition"
        )
        self._skip_partition_set = (
            (not has_partition_sync)
            and (
                isinstance(self.optimizer, DistAdagrad)
                or bool(self.config.get("job.distributed.causal_merge"))
                or bool(self.config.get("job.distributed.causal_merge_row"))
                or bool(self._window_work)
            )
        )
        self._entity_vocab_size = self.dataset.num_entities()
        self._relation_vocab_size = self.dataset.num_relations()
        self._debug_id_bounds = bool(
            self.config.get("job.distributed.debug_id_bounds")
        )
        self._current_partition_version = None
        self._current_partition_id = None
        self._partition_grad_sum = 0.0
        self._partition_grad_sum_t = None
        self._partition_grad_samples = 0
        self._relation_gradient_trace = bool(
            self.config.get("job.distributed.relation_gradient_trace")
        )
        self._gradient_log_interval = 0
        self._gradient_log_top_relations = 0
        try:
            self._gradient_log_interval = int(
                self.config.get("job.distributed.gradient_trace_log_interval")
            )
        except KeyError:
            self._gradient_log_interval = 0
        if 0 < self._gradient_log_interval < 5:
            self.config.log(
                "gradient_trace_log_interval is very small; "
                "clamping to 5 to reduce logging overhead."
            )
            self._gradient_log_interval = 5
        try:
            self._gradient_log_top_relations = int(
                self.config.get("job.distributed.gradient_trace_top_relations")
            )
        except KeyError:
            self._gradient_log_top_relations = 0
        self._last_partition_grad_summary = None
        self._relation_grad_sum = None
        self._relation_grad_count = None
        if self._relation_gradient_trace:
            self._relation_grad_sum = torch.zeros(
                self._relation_vocab_size, dtype=torch.float32, device="cpu"
            )
            self._relation_grad_count = torch.zeros(
                self._relation_vocab_size, dtype=torch.long, device="cpu"
            )
        partition_type = str(self.config.get("job.distributed.partition_type") or "")
        self._use_glow_features = partition_type == "glow"
        glow_cfg = self.config.get("job.distributed.glow") or {}
        track_cfg = glow_cfg.get("track_partition_gradients")
        if track_cfg is None:
            self._track_partition_gradients = bool(
                self._use_glow_features
                or self._relation_gradient_trace
                or self._gradient_log_interval > 0
                or self._gradient_log_top_relations > 0
            )
        else:
            self._track_partition_gradients = (
                bool(track_cfg) or self._relation_gradient_trace
            )
        self._last_finished_partition = None
        self._current_window_members = None
        self._current_window_entities = None
        self._current_window_versions = None
        self._current_partition_size = 0
        if self._use_glow_features:
            self._prefetch_window_entities = bool(
                glow_cfg.get("prefetch_window_entities", False)
            )
            self._lookahead_negatives = bool(
                glow_cfg.get("lookahead_negatives", False)
            )
            self._overlap_negative_sampling = bool(
                glow_cfg.get("overlap_negative_sampling", False)
            )
            self._force_overlap_pool = bool(
                glow_cfg.get("force_pooled_negatives", False)
            )
        else:
            self._prefetch_window_entities = False
            self._lookahead_negatives = False
            self._overlap_negative_sampling = False
            self._force_overlap_pool = False
        self._overlap_sampling_logged = False
        self._window_prefetch_key = None
        self._window_work = (
            bool(glow_cfg.get("window_work", False)) if self._use_glow_features else False
        )
        self._glow_debug_trace = (
            bool(glow_cfg.get("debug_trace", False)) if self._use_glow_features else False
        )
        self._glow_trace_stats = None
        try:
            self._debug_window_versions_remaining = int(
                glow_cfg.get("debug_window_versions", 0)
            )
        except (TypeError, ValueError):
            self._debug_window_versions_remaining = 0
        self._partition_maps_ready = False
        self._entity_partition_map = None
        self._relation_partition_map = None
        self._materialize_partitions = False
        self._materialization_notice_logged = False
        self._stage_local_ids = False
        self._materialized_iterator = None
        self._loader_debug_stats = defaultdict(int)
        self._need_unique_entities = self.entity_sync_level == "batch"
        self._need_unique_relations = self.relation_sync_level == "batch"
        if (
            self._overlap_negative_sampling
            and self._force_overlap_pool
            and self.config.get("negative_sampling.sampling_type") != "pooled"
        ):
            self.config.set(
                "negative_sampling.sampling_type", "pooled", log=True
            )
            self._sampler = KgeSampler.create(
                self.config, "negative_sampling", self.dataset
            )
        self.entity_pre_pull = self.config.get("job.distributed.entity_pre_pull")
        self.relation_pre_pull = self.config.get("job.distributed.relation_pre_pull")
        self.pre_localize_batch = int(
            self.config.get("job.distributed.pre_localize_batch")
        )
        self.entity_mapper_tensors = deque()
        self.relation_mapper_tensors = deque()
        mapper_device = (
            self._map_ids_device
            if self._map_ids_device.type == "cuda"
            else None
        )
        for i in range(self._effective_num_workers + 1):
            self.entity_mapper_tensors.append(
                torch.full(
                    (self.dataset.num_entities(),),
                    -1,
                    dtype=torch.long,
                    device=mapper_device,
                )
            )
            self.relation_mapper_tensors.append(
                torch.full(
                    (self.dataset.num_relations(),),
                    -1,
                    dtype=torch.long,
                    device=mapper_device,
                )
            )

        # also defines the local entities
        self._initialize_parameter_server(init_for_load_only=init_for_load_only)

        self.hot_entity_tensor = self._load_hot_entity_cache()
        if self.hot_entity_tensor is not None:
            self._localize_hot_entities()
            try:
                self.model.get_s_embedder().enable_hot_cache(self.hot_entity_tensor)
                self.config.log(
                    f"Hot entity cache enabled for {len(self.hot_entity_tensor)} entities."
                )
            except Exception as exc:
                self.config.log(f"Failed to enable hot entity cache: {exc}")

        def stop_and_wait(job):
            job.parameter_client.stop()
            job.parameter_client.barrier()
        self.early_stop_hooks.append(stop_and_wait)

        def check_stopped(job):
            print("checking for", job.parameter_client.rank)
            job.parameter_client.barrier()
            return job.parameter_client.is_stopped()
        self.early_stop_conditions.append(check_stopped)
        self.work_pre_localized = False
        if self.config.get("job.distributed.pre_localize_partition"):
            self.pre_localized_entities = None
            self.pre_localized_relations = None
            self.pre_batch_hooks.append(self._pre_localize_work)

        if self.__class__ == TrainingJobNegativeSamplingDistributed:
            for f in Job.job_created_hooks:
                f(self)

    def _initialize_parameter_server(self, init_for_load_only=False):
        # initialize the parameter server
        #  each worker takes as many entities as it can fit, inits and pushes
        #  init work is distributed by the work scheduler
        if not init_for_load_only and not self.config.get(
            "lookup_embedder.pretrain.model_filename"
        ):
            # only the first worker initializes the relations
            if self.parameter_client.rank == self.min_rank:
                self.model.get_p_embedder().push_all()
            entity_embedding_layer_size = self.model.get_s_embedder()._embeddings.weight.data.shape[0]
            self.local_entities = self.work_scheduler_client.get_local_entities()
            self.parameter_client.localize(self.local_entities, asynchronous=True)
            init_work_packages = self.local_entities.split(entity_embedding_layer_size)
            for init_work_package in init_work_packages:
                self.model.get_s_embedder().initialize(
                    self.model.get_s_embedder()._embeddings.weight.data
                )
                self.model.get_s_embedder()._normalize_embeddings()
                self._push_init_to_parameter_server(init_work_package)
        self.parameter_client.barrier()

    def _load_hot_entity_cache(self):
        if not self.config.get("job.distributed.hot_entity_cache.enable"):
            return None
        cache_file = self.config.get("job.distributed.hot_entity_cache.file")
        dataset_folder = Path(self.dataset.folder)
        if cache_file:
            cache_path = Path(cache_file)
            if not cache_path.is_absolute():
                cache_path = dataset_folder / cache_path
        else:
            cache_path = dataset_folder / "analysis_hot_entities.npy"
        if not cache_path.is_file():
            self.config.log(
                f"No hot-entity cache found at {cache_path}. Skipping hot-entity localization."
            )
            return None
        try:
            hot_entities = np.load(cache_path)
        except Exception as e:
            self.config.log(f"Failed to load hot entities from {cache_path}: {e}")
            return None
        max_entities = self.config.get("job.distributed.hot_entity_cache.max_entities")
        if max_entities is not None and max_entities > 0:
            hot_entities = hot_entities[:max_entities]
        if hot_entities.size == 0:
            self.config.log(
                f"Hot-entity cache at {cache_path} is empty. Skipping localization."
            )
            return None
        tensor = torch.from_numpy(hot_entities.astype(np.int64)).long()
        self.config.log(
            f"Loaded {len(tensor)} hot entities from {cache_path} for caching."
        )
        return tensor

    def _localize_hot_entities(self):
        if self.hot_entity_tensor is None:
            return
        try:
            self.model.get_s_embedder().localize(
                self.hot_entity_tensor, make_unique=True
            )
            self.config.log(
                f"Requested localization of {len(self.hot_entity_tensor)} hot entities."
            )
        except Exception as e:
            self.config.log(f"Failed to localize hot entities: {e}")

    def _push_init_to_parameter_server(self, entity_ids: torch.Tensor):
        push_tensor = torch.cat(
            (
                self.model.get_s_embedder()
                ._embeddings.weight.data[: len(entity_ids)]
                .cpu(),
                self.model.get_s_embedder().optimizer_values[: len(entity_ids)].cpu(),
            ),
            dim=1,
        )
        # Use set to avoid accumulating into non-zero parameter server memory.
        self.parameter_client.set(
            entity_ids + self.model.get_s_embedder().lapse_offset, push_tensor.cpu(),
        )

    @staticmethod
    def _pre_localize_work(job, batch_index):
        if batch_index % 100 != 0:
            return
        if not job.work_pre_localized:
            work, entities, relations, partition_id, partition_version, wait = job.work_scheduler_client.get_pre_localize_work()
            if wait:
                return
            if entities is not None:
                entities_ps_offset = job.model.get_s_embedder().lapse_offset
                job.pre_localized_entities = entities + entities_ps_offset
                job.parameter_client.localize(
                    job.pre_localized_entities, asynchronous=True
                )
            if relations is not None:
                relations_ps_offset = job.model.get_p_embedder().lapse_offset
                job.pre_localized_relations = relations + relations_ps_offset
                job.parameter_client.localize(
                    job.pre_localized_relations, asynchronous=True
                )
            job.work_pre_localized = True

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()

        self.num_examples = self.dataset.split(self.train_split).size(0)
        shuffle_partitions = self.config.get(
            "job.distributed.shuffle_partition_samples"
        )
        materialize_partitions = self.config.get(
            "job.distributed.materialize_partition_batches"
        )
        self._collate_fn = self._get_collate_fun()
        self.dataloader_dataset = BatchDataset(
            self.dataset.split(self.train_split),
            batch_size=self.batch_size,
            shuffle=shuffle_partitions,
            materialize=materialize_partitions,
            materialize_device=self.device if isinstance(self.device, str) else None,
        )
        self._materialize_partitions = materialize_partitions
        self._materialization_notice_logged = False
        self._stage_local_ids = bool(
            self.config.get("job.distributed.stage_local_ids")
        )
        if self._stage_local_ids and not materialize_partitions:
            self.config.log(
                "Disabled GPU staging of local IDs because materialized partitions are disabled."
            )
            self._stage_local_ids = False
        if self._stage_local_ids and (
            self.entity_sync_level != "partition"
            or self.relation_sync_level != "partition"
        ):
            self.config.log(
                "Disabled GPU staging of local IDs because sync level is not partition."
            )
            self._stage_local_ids = False
        if materialize_partitions:
            self.config.log(
                "Materialized partition batches enabled; staging triples once per partition chunk."
            )
        # initializing dataloader as soon as we got the triples from work scheduler
        self.loader = None
        if self._sampler.uses_pool():
            if self.local_entities is None:
                self.local_entities = self.work_scheduler_client.get_local_entities()
                self.parameter_client.localize(self.local_entities)
            self._sampler.set_pool(self.local_entities, S)
            self._sampler.set_pool(self.local_entities, O)
    def _get_collate_fun(self):
        # create the collate function
        def collate(batch):
            """For a batch of size n, returns a tuple of:

            - triples (tensor of shape [n,3], ),
            - negative_samples (list of tensors of shape [n,num_samples]; 3 elements
              in order S,P,O)
            """

            if batch[0] is None:
                # this can happen due to keeping the dataloader alive
                return None
            triple_ids = batch[0][0]
            epoch = batch[0][1]
            local_partition_id = batch[0][2]
            triples = self.dataloader_dataset.fetch_triples(triple_ids)
            gpu_triples = self.dataloader_dataset.fetch_triples_device(triple_ids)
            triples = self._pin_triples_if_needed(triples)
            sampler_triples = triples
            if (
                gpu_triples is not None
                and self._sampler.supports_device_sampling(gpu_triples)
            ):
                sampler_triples = gpu_triples
                if not self._gpu_sampling_logged:
                    self.config.log(
                        "GPU sampling enabled for negative sampling batches."
                    )
                    self._gpu_sampling_logged = True

            negative_samples = list()
            for slot in [S, P, O]:
                sample = self._sampler.sample(sampler_triples, slot)
                if (
                    self._non_blocking_transfer
                    and getattr(self, "_effective_num_workers", 0) == 0
                ):
                    sample = sample.pin_memory()
                negative_samples.append(sample)
            unique_time = -time.time()
            unique_entities = None
            unique_relations = None
            entity_embedder = self.model.get_s_embedder()
            relation_embedder = self.model.get_p_embedder()
            if self._need_unique_entities:
                use_gpu_unique = (
                    self._unique_on_gpu
                    and sampler_triples.device.type == "cuda"
                )
                # Unique-id construction stays on CPU even if samplers produced
                # device-resident buffers when configured.
                pos_source = sampler_triples if use_gpu_unique else triples
                pos_entities = pos_source[:, [S, O]].view(-1)
                if not use_gpu_unique and pos_entities.device.type != "cpu":
                    pos_entities = pos_entities.cpu()
                neg_s = negative_samples[S].unique_samples(remove_dropped=False)
                if use_gpu_unique:
                    if neg_s.device != sampler_triples.device:
                        neg_s = neg_s.to(
                            sampler_triples.device,
                            non_blocking=self._non_blocking_transfer,
                        )
                elif neg_s.device.type != "cpu":
                    neg_s = neg_s.cpu()
                neg_o = negative_samples[O].unique_samples(remove_dropped=False)
                if use_gpu_unique:
                    if neg_o.device != sampler_triples.device:
                        neg_o = neg_o.to(
                            sampler_triples.device,
                            non_blocking=self._non_blocking_transfer,
                        )
                elif neg_o.device.type != "cpu":
                    neg_o = neg_o.cpu()
                unique_entities = torch.unique(
                    torch.cat(
                        (
                            pos_entities,
                            neg_s,
                            neg_o,
                        )
                    ),
                    sorted=False,
                )
                if hasattr(entity_embedder, "apply_locality_order"):
                    unique_entities = entity_embedder.apply_locality_order(
                        unique_entities
                    )
            if self._need_unique_relations:
                use_gpu_unique = (
                    self._unique_on_gpu
                    and sampler_triples.device.type == "cuda"
                )
                pos_source = sampler_triples if use_gpu_unique else triples
                pos_relations = pos_source[:, [P]].view(-1)
                if not use_gpu_unique and pos_relations.device.type != "cpu":
                    pos_relations = pos_relations.cpu()
                neg_p = negative_samples[P].unique_samples(remove_dropped=False)
                if use_gpu_unique:
                    if neg_p.device != sampler_triples.device:
                        neg_p = neg_p.to(
                            sampler_triples.device,
                            non_blocking=self._non_blocking_transfer,
                        )
                elif neg_p.device.type != "cpu":
                    neg_p = neg_p.cpu()
                unique_relations = torch.unique(
                    torch.cat(
                        (
                            pos_relations,
                            neg_p,
                        )
                    ),
                    sorted=False,
                )
                if hasattr(relation_embedder, "apply_locality_order"):
                    unique_relations = relation_embedder.apply_locality_order(
                        unique_relations
                    )
            unique_time += time.time()

            return {
                "triples": triples,
                "negative_samples": negative_samples,
                "unique_entities": unique_entities,
                "unique_relations": unique_relations,
                "unique_time": unique_time,
                "epoch": epoch,
                "local_partition_id": local_partition_id,
                "triple_ids": triple_ids,
                "_gpu_triples": gpu_triples,
            }

        return collate

    def _pin_triples_if_needed(self, triples: torch.Tensor) -> torch.Tensor:
        if not self._non_blocking_transfer:
            return triples
        if getattr(self, "_effective_num_workers", 0) > 0:
            return triples
        if not isinstance(self.device, str) or not self.device.startswith("cuda"):
            return triples
        if not torch.cuda.is_available():
            return triples
        if triples.device.type != "cpu":
            return triples
        if triples.is_pinned():
            return triples
        return triples.pin_memory()

    def _resolve_map_ids_device(self) -> torch.device:
        if not self._map_ids_on_gpu:
            return torch.device("cpu")
        device = self.device
        if isinstance(device, torch.device):
            return device if device.type == "cuda" else torch.device("cpu")
        if isinstance(device, str) and device.startswith("cuda") and torch.cuda.is_available():
            return torch.device(device)
        return torch.device("cpu")

    def _ensure_partition_mapper_device(self, kind: str) -> Optional[torch.Tensor]:
        if self._map_ids_device.type != "cuda":
            return None
        if kind == "entity":
            size = self._entity_vocab_size
            attr = "_entity_partition_mapper_device"
        else:
            size = self._relation_vocab_size
            attr = "_relation_partition_mapper_device"
        mapper = getattr(self, attr)
        if (
            mapper is None
            or mapper.device != self._map_ids_device
            or mapper.numel() != size
        ):
            mapper = torch.full(
                (size,), -1, dtype=torch.long, device=self._map_ids_device
            )
            setattr(self, attr, mapper)
        return mapper

    def _map_ids_to_local(self, batch):
        map_device = self._map_ids_device
        if map_device.type == "cuda":
            if batch["triples"].device != map_device:
                batch["triples"] = batch["triples"].to(
                    map_device, non_blocking=self._non_blocking_transfer
                )
            for idx, ns in enumerate(batch["negative_samples"]):
                if ns.positive_triples.device != map_device:
                    batch["negative_samples"][idx] = ns.to(
                        map_device, non_blocking=self._non_blocking_transfer
                    )
        else:
            if batch["triples"].device.type != "cpu":
                batch["triples"] = batch["triples"].cpu()
            for idx, ns in enumerate(batch["negative_samples"]):
                if ns.positive_triples.device.type != "cpu":
                    batch["negative_samples"][idx] = ns.to("cpu")
        if batch["triples"].dtype != torch.long:
            batch["triples"] = batch["triples"].to(dtype=torch.long)
        if self._debug_id_bounds:
            self._debug_validate_batch_ids(batch)
        # map ids to local ids
        if self.entity_sync_level == "partition":
            if map_device.type == "cuda":
                entity_mapper = self._ensure_partition_mapper_device("entity")
                if entity_mapper is None:
                    entity_mapper = self.model.get_s_embedder().global_to_local_mapper
            else:
                entity_mapper = self.model.get_s_embedder().global_to_local_mapper
        else:
            entity_mapper = self.entity_mapper_tensors.popleft()
            unique_entities = batch.get("unique_entities")
            if unique_entities is None:
                raise RuntimeError("Missing unique_entities for batch-level sync.")
            if map_device.type == "cuda":
                unique_entities_device = unique_entities.to(
                    map_device, non_blocking=self._non_blocking_transfer
                )
                entity_mapper[unique_entities_device] = torch.arange(
                    len(unique_entities), dtype=torch.long, device=map_device
                )
            else:
                entity_mapper[unique_entities] = torch.arange(
                    len(unique_entities), dtype=torch.long
                )
        if self.relation_sync_level == "partition":
            if map_device.type == "cuda":
                relation_mapper = self._ensure_partition_mapper_device("relation")
                if relation_mapper is None:
                    relation_mapper = self.model.get_p_embedder().global_to_local_mapper
            else:
                relation_mapper = self.model.get_p_embedder().global_to_local_mapper
        else:
            relation_mapper = self.relation_mapper_tensors.popleft()
            unique_relations = batch.get("unique_relations")
            if unique_relations is None:
                raise RuntimeError("Missing unique_relations for batch-level sync.")
            if map_device.type == "cuda":
                unique_relations_device = unique_relations.to(
                    map_device, non_blocking=self._non_blocking_transfer
                )
                relation_mapper[unique_relations_device] = torch.arange(
                    len(unique_relations), dtype=torch.long, device=map_device
                )
            else:
                relation_mapper[unique_relations] = torch.arange(
                    len(unique_relations), dtype=torch.long
                )
        batch["triples"][:, S] = entity_mapper[batch["triples"][:, S]]
        batch["triples"][:, P] = relation_mapper[batch["triples"][:, P]]
        batch["triples"][:, O] = entity_mapper[batch["triples"][:, O]]
        batch["negative_samples"][S].map_samples(entity_mapper)
        batch["negative_samples"][P].map_samples(relation_mapper)
        batch["negative_samples"][O].map_samples(entity_mapper)

        # for debugging reset the entity mapper to -1
        # entity_mapper[:] = -1
        if self.entity_sync_level != "partition":
            if map_device.type == "cuda":
                entity_mapper[unique_entities_device] = -1
            else:
                entity_mapper[unique_entities] = -1
            self.entity_mapper_tensors.append(entity_mapper)
        if self.relation_sync_level != "partition":
            if map_device.type == "cuda":
                relation_mapper[unique_relations_device] = -1
            else:
                relation_mapper[unique_relations] = -1
            self.relation_mapper_tensors.append(relation_mapper)
        if self._debug_id_bounds:
            self._debug_validate_local_ids(batch)
        return batch

    def _debug_validate_batch_ids(self, batch):
        """Ensure triples/negatives stay within global vocab bounds."""
        triples = batch.get("triples")
        if triples is None or triples.numel() == 0:
            return

        def _check_range(values, limit, name):
            if values is None or values.numel() == 0:
                return
            max_id = int(values.max().item())
            min_id = int(values.min().item())
            if min_id < 0 or max_id >= limit:
                part = batch.get("local_partition_id")
                epoch = batch.get("epoch")
                raise RuntimeError(
                    f"[ID bounds] {name} out of range ({min_id},{max_id}) "
                    f"with limit {limit} (partition={part}, epoch={epoch})."
                )

        subjects = triples[:, S].view(-1)
        objects = triples[:, O].view(-1)
        relations = triples[:, P].view(-1)
        _check_range(subjects, self._entity_vocab_size, "triples.subject")
        _check_range(objects, self._entity_vocab_size, "triples.object")
        _check_range(relations, self._relation_vocab_size, "triples.relation")

        for slot, limit, desc in (
            (S, self._entity_vocab_size, "neg.subject"),
            (O, self._entity_vocab_size, "neg.object"),
            (P, self._relation_vocab_size, "neg.relation"),
        ):
            neg_sample = batch["negative_samples"][slot]
            samples = None
            try:
                samples = neg_sample.unique_samples(remove_dropped=False)
            except Exception:
                try:
                    samples = neg_sample.samples()
                except Exception:
                    samples = None
            if samples is None or samples.numel() == 0:
                continue
            _check_range(samples.view(-1), limit, desc)

    def _debug_validate_local_ids(self, batch):
        """Ensure localized ids fit inside the local embedder size."""
        entity_vocab = self.model.get_s_embedder().vocab_size
        relation_vocab = self.model.get_p_embedder().vocab_size

        def _check_local(values, limit, name):
            if values is None or values.numel() == 0:
                return
            max_id = int(values.max().item())
            min_id = int(values.min().item())
            if min_id < 0 or max_id >= limit:
                part = batch.get("local_partition_id")
                epoch = batch.get("epoch")
                raise RuntimeError(
                    f"[Local ID bounds] {name} out of range ({min_id},{max_id}) "
                    f"with local limit {limit} (partition={part}, epoch={epoch})."
                )

        triples = batch.get("triples")
        if triples is not None and triples.numel() > 0:
            _check_local(triples[:, S].view(-1), entity_vocab, "local subject")
            _check_local(triples[:, O].view(-1), entity_vocab, "local object")
            _check_local(triples[:, P].view(-1), relation_vocab, "local relation")

        for slot, limit, desc in (
            (S, entity_vocab, "local neg subject"),
            (O, entity_vocab, "local neg object"),
            (P, relation_vocab, "local neg relation"),
        ):
            neg_sample = batch["negative_samples"][slot]
            samples = None
            try:
                samples = neg_sample.unique_samples(remove_dropped=False)
            except Exception:
                try:
                    samples = neg_sample.samples()
                except Exception:
                    samples = None
            if samples is None or samples.numel() == 0:
                continue
            _check_local(samples.view(-1), limit, desc)

    def _local_ids_out_of_range(self, batch) -> bool:
        """Return True if any ids exceed local embedder bounds."""
        entity_vocab = self.model.get_s_embedder().vocab_size
        relation_vocab = self.model.get_p_embedder().vocab_size

        def _out_of_range(values, limit):
            if values is None or values.numel() == 0:
                return False
            max_id = int(values.max().item())
            min_id = int(values.min().item())
            return min_id < 0 or max_id >= limit

        triples = batch.get("triples")
        if triples is not None and triples.numel() > 0:
            if (
                _out_of_range(triples[:, S].view(-1), entity_vocab)
                or _out_of_range(triples[:, O].view(-1), entity_vocab)
                or _out_of_range(triples[:, P].view(-1), relation_vocab)
            ):
                return True
        for slot, limit in ((S, entity_vocab), (O, entity_vocab), (P, relation_vocab)):
            neg_sample = batch["negative_samples"][slot]
            samples = None
            try:
                samples = neg_sample.unique_samples(remove_dropped=False)
            except Exception:
                try:
                    samples = neg_sample.samples()
                except Exception:
                    samples = None
            if samples is None or samples.numel() == 0:
                continue
            if _out_of_range(samples.view(-1), limit):
                return True
        return False

    def _ensure_partition_relations(self, work, work_relations):
        if work is None or work.numel() == 0:
            return work_relations
        triples = getattr(self.dataloader_dataset, "triples", None)
        if triples is None:
            return work_relations
        work_cpu = work if work.device.type == "cpu" else work.cpu()
        if work_cpu.dtype != torch.long:
            work_cpu = work_cpu.to(dtype=torch.long)
        triples_cpu = triples if triples.device.type == "cpu" else triples.cpu()
        rel_ids = triples_cpu.index_select(0, work_cpu)[:, P]
        unique_rel = torch.unique(rel_ids)
        if unique_rel.numel() == 0:
            return work_relations
        if work_relations is None or work_relations.numel() == 0:
            return unique_rel.to(dtype=torch.long)
        work_rel_cpu = (
            work_relations
            if work_relations.device.type == "cpu"
            else work_relations.cpu()
        ).to(dtype=torch.long)
        work_rel_sorted = torch.unique(work_rel_cpu)
        pos = torch.searchsorted(work_rel_sorted, unique_rel)
        in_bounds = pos < work_rel_sorted.numel()
        if in_bounds.any():
            matches = work_rel_sorted[pos[in_bounds]] == unique_rel[in_bounds]
            in_bounds = in_bounds.clone()
            in_bounds[in_bounds] = matches
        if in_bounds.all():
            return work_relations
        missing = unique_rel[~in_bounds]
        if self._debug_id_bounds:
            self.config.log(
                "Extending relation partition set with "
                f"{missing.numel()} missing ids from work."
            )
        merged = torch.unique(torch.cat((work_rel_sorted, missing)))
        target_device = work_relations.device
        return merged.to(device=target_device, dtype=torch.long)

    def _maybe_remap_staged_batch(self, batch):
        """Remap staged local ids if they appear to be global."""
        if not self._stage_local_ids:
            return batch
        if not self._debug_id_bounds:
            return batch
        if self._local_ids_out_of_range(batch):
            self.config.log(
                "Detected out-of-range local ids in staged batch; "
                "remapping with global_to_local mapper."
            )
            batch = self._map_ids_to_local(batch)
        self._debug_validate_local_ids(batch)
        return batch

    def _prepare_batch_ahead(self, batches: deque, new_batch=None):
        if not batches:
            return
        head = batches[0]
        should_prefetch = self.entity_pre_pull > 0 or self.relation_pre_pull > 0
        if should_prefetch and not head.get("_device_ready"):
            gpu_triples = head.get("_gpu_triples")
            if gpu_triples is not None:
                head["triples"] = gpu_triples
            elif head["triples"].device.type != "cuda":
                head["triples"] = head["triples"].to(
                    self.device, non_blocking=self._non_blocking_transfer
                )
        lookahead_entries = head.get("_lookahead_negative_samples")
        for idx, ns in enumerate(head["negative_samples"]):
            ns.positive_triples = head["triples"]
            if lookahead_entries and len(lookahead_entries) > idx:
                payload = lookahead_entries[idx]
                if self._validate_lookahead_payload(payload):
                    ns.attach_lookahead(payload)
                    self._loader_debug_stats["lookahead_hits"] += 1
                else:
                    self._loader_debug_stats["lookahead_rejected"] += 1
                lookahead_entries[idx] = None
        if lookahead_entries and all(entry is None for entry in lookahead_entries):
            head.pop("_lookahead_negative_samples", None)
        if should_prefetch and not head.get("_device_ready"):
            head["negative_samples"] = [
                ns.to(self.device, non_blocking=self._non_blocking_transfer)
                for ns in head["negative_samples"]
            ]
            head["_device_ready"] = True
        # prepare look-ahead negatives for the next batch
        target = new_batch if new_batch is not None else batches[-1]
        self._prepare_negative_lookahead(target)
        if (
            self.entity_sync_level == "batch"
            and self.entity_pre_pull > 0
            and target["unique_entities"] is not None
            and not target.get("_entity_prefetched")
        ):
            self.model.get_s_embedder().pre_pull(target["unique_entities"])
            self.model.get_s_embedder().pre_pulled_to_device()
            target["_entity_prefetched"] = True
        if (
            self.relation_sync_level == "batch"
            and self.relation_pre_pull > 0
            and target["unique_relations"] is not None
            and not target.get("_relation_prefetched")
        ):
            self.model.get_p_embedder().pre_pull(target["unique_relations"])
            self.model.get_p_embedder().pre_pulled_to_device()
            target["_relation_prefetched"] = True
        if (
            self.pre_localize_batch > 0
            and target["unique_entities"] is not None
        ):
            self.model.get_s_embedder().localize(
                target["unique_entities"],
                asynchronous=True
            )

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        # move triples and negatives to GPU. With some implementaiton effort, this may
        # be avoided.
        result.prepare_time -= time.time()
        # result.cpu_gpu_time -= time.time()
        device_ready = batch.get("_device_ready", False)
        gpu_triples = batch.get("_gpu_triples")
        if gpu_triples is not None:
            batch["triples"] = gpu_triples
        elif not device_ready and batch["triples"].device.type != "cuda":
            batch["triples"] = batch["triples"].to(
                self.device, non_blocking=self._non_blocking_transfer
            )
        lookahead_entries = batch.get("_lookahead_negative_samples")
        for idx, ns in enumerate(batch["negative_samples"]):
            ns.positive_triples = batch["triples"]
            if lookahead_entries and len(lookahead_entries) > idx:
                payload = lookahead_entries[idx]
                if self._validate_lookahead_payload(payload):
                    ns.attach_lookahead(payload)
                    self._loader_debug_stats["lookahead_hits"] += 1
                else:
                    self._loader_debug_stats["lookahead_rejected"] += 1
                lookahead_entries[idx] = None
        if lookahead_entries and all(entry is None for entry in lookahead_entries):
            batch.pop("_lookahead_negative_samples", None)
        if not device_ready:
            batch["negative_samples"] = [
                ns.to(self.device, non_blocking=self._non_blocking_transfer)
                for ns in batch["negative_samples"]
            ]
            batch["_device_ready"] = True
        # result.cpu_gpu_time += time.time()
        result.unique_time += batch["unique_time"]
        if self.entity_sync_level == "batch":
            unique_entities = batch["unique_entities"]

            result.ps_wait_time -= time.time()
            if not self.entity_async_write_back:
                self.optimizer.wait_for_pending("entity")
            result.ps_wait_time += time.time()
            if self.entity_localize and not self.entity_partition_localized:
                self.model.get_s_embedder().localize(
                    unique_entities, asynchronous=True
                )
            result.pull_and_map_time -= time.time()
            (
                entity_pull_time,
                cpu_gpu_time,
            ) = self.model.get_s_embedder()._pull_embeddings(unique_entities)
            result.pull_and_map_time += time.time()
            result.entity_pull_time += entity_pull_time
            result.cpu_gpu_time += cpu_gpu_time
        if self.relation_sync_level == "batch":
            unique_relations = batch["unique_relations"]
            result.ps_wait_time -= time.time()
            if not self.relation_async_write_back:
                self.optimizer.wait_for_pending("relation")
            result.ps_wait_time += time.time()
            if self.relation_localize and not self.relation_partition_localized:
                self.model.get_p_embedder().localize(
                    unique_relations, asynchronous=True
                )
            result.pull_and_map_time -= time.time()
            (
                relation_pull_time,
                cpu_gpu_time,
            ) = self.model.get_p_embedder()._pull_embeddings(unique_relations)
            result.pull_and_map_time += time.time()
            result.relation_pull_time += relation_pull_time
            result.cpu_gpu_time += cpu_gpu_time

        batch["labels"] = [None] * 3  # reuse label tensors b/w subbatches
        result.size = len(batch["triples"])
        result.prepare_time += time.time()



    def _init_dataloader(self):
        mp_context = (
            torch.multiprocessing.get_context("fork")
            if self._effective_num_workers > 0
            else None
        )
        pin_memory = self.config.get("train.pin_memory")
        if (
            not pin_memory
            and isinstance(self.device, str)
            and self.device.startswith("cuda")
        ):
            pin_memory = True
        self._non_blocking_transfer = bool(
            isinstance(self.device, str)
            and self.device.startswith("cuda")
            and pin_memory
        )
        self.loader = torch.utils.data.DataLoader(
            self.dataloader_dataset,
            sampler=InfiniteSequentialSampler(self.dataloader_dataset),
            collate_fn=self._collate_fn,
            # shuffle needs to be False since it is handled in the dataset object
            shuffle=False,
            # batch size needs to be 1 since it is handled in the dataset object
            # batch_size=self.batch_size,
            num_workers=self._effective_num_workers,
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=pin_memory,
            multiprocessing_context=mp_context,
        )

    def _use_materialized_iterator(self):
        return self._materialize_partitions and self._effective_num_workers == 0

    def _reset_partition_gradient_trace(self):
        self._partition_grad_sum = 0.0
        self._partition_grad_sum_t = (
            torch.zeros((), device=self.device)
            if self._track_partition_gradients
            else None
        )
        self._partition_grad_samples = 0
        self._reset_partition_relation_gradient_trace()

    def _reset_partition_relation_gradient_trace(self):
        if not self._relation_gradient_trace or self._relation_grad_sum is None:
            return
        self._relation_grad_sum.zero_()
        self._relation_grad_count.zero_()

    def _accumulate_partition_gradient(self):
        if not self._track_partition_gradients:
            return
        if self.is_forward_only or self._current_partition_id is None:
            return
        grad_norm = self._compute_batch_gradient_norm()
        if grad_norm is None:
            return
        if self._partition_grad_sum_t is None:
            self._partition_grad_sum_t = grad_norm
        else:
            if grad_norm.device != self._partition_grad_sum_t.device:
                grad_norm = grad_norm.to(self._partition_grad_sum_t.device)
            self._partition_grad_sum_t = self._partition_grad_sum_t + grad_norm
        self._partition_grad_samples += 1
        self._accumulate_partition_relation_gradients()
        if (
            self._gradient_log_interval > 0
            and self._partition_grad_samples % self._gradient_log_interval == 0
        ):
            if self._partition_grad_sum_t is None:
                avg_grad = self._partition_grad_sum / max(
                    1, self._partition_grad_samples
                )
            else:
                avg_grad = float(
                    (self._partition_grad_sum_t / max(1, self._partition_grad_samples))
                    .detach()
                    .cpu()
                )
            message = (
                f"Partition {self._current_partition_id} gradient norm avg="
                f"{avg_grad:.6f} (samples={self._partition_grad_samples})."
            )
            if (
                self._gradient_log_top_relations > 0
                and self._relation_gradient_trace
                and self._relation_grad_sum is not None
            ):
                mask = self._relation_grad_count > 0
                if mask.any():
                    rel_ids = torch.nonzero(mask, as_tuple=False).view(-1)
                    rel_avg = (
                        self._relation_grad_sum[mask]
                        / self._relation_grad_count[mask].to(self._relation_grad_sum.dtype)
                    )
                    top_k = min(self._gradient_log_top_relations, rel_avg.numel())
                    vals, idx = torch.topk(rel_avg, k=top_k, largest=True)
                    top_rel_ids = rel_ids[idx].tolist()
                    top_rel_vals = vals.tolist()
                    pairs = ", ".join(
                        f"{rid}:{val:.4f}"
                        for rid, val in zip(top_rel_ids, top_rel_vals)
                    )
                    message += f" Top relations: {pairs}."
            self.config.log(message)

    def _compute_batch_gradient_norm(self):
        total = None
        has_grad = False
        for param in self.model.parameters():
            if param.grad is None:
                continue
            has_grad = True
            grad = param.grad.detach()
            if grad.is_sparse:
                grad = grad.coalesce()
                values = grad.values()
                part = values.pow(2).sum()
            elif grad.layout != torch.strided:
                values = grad.values() if hasattr(grad, "values") else grad._values()
                part = values.pow(2).sum()
            else:
                part = grad.pow(2).sum()
            if total is None:
                total = part
            else:
                if part.device != total.device:
                    part = part.to(total.device)
                total = total + part
        if not has_grad:
            return torch.zeros((), device=self.device)
        if total is None:
            return torch.zeros((), device=self.device)
        return torch.sqrt(total)

    def _accumulate_partition_relation_gradients(self):
        if (
            not self._relation_gradient_trace
            or self._current_partition_id is None
            or self.is_forward_only
        ):
            return
        embedder = self.model.get_p_embedder()
        pulled_ids = getattr(embedder, "pulled_ids", None)
        grad = getattr(embedder._embeddings.weight, "grad", None)
        if grad is None:
            return
        grad = grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce()
            row_ids = grad.indices()[0]
            row_norms = torch.zeros(
                grad.size(0), device=grad.device, dtype=grad.dtype
            )
            row_norms.index_add_(0, row_ids, grad.values().pow(2).sum(dim=1))
            grad_norms = row_norms.sqrt()
        else:
            grad_norms = grad.pow(2).sum(dim=1).sqrt()
        grad_norms_cpu = grad_norms.to(
            device="cpu", dtype=self._relation_grad_sum.dtype
        )
        pulled_ids_cpu = None
        if pulled_ids is not None:
            pulled_ids_cpu = pulled_ids.detach().to(device="cpu", dtype=torch.long)

        # Prefer mapping gradient rows back to global relation ids via pulled_ids.
        # In distributed embedders, grad_norms may be the full local table, while pulled_ids
        # contains only the active pulled rows. In that case, slice grad_norms to the pulled region
        # instead of falling back to local row indices.
        if pulled_ids_cpu is None:
            if not hasattr(self, "_relation_grad_fallback_logged"):
                self._relation_grad_fallback_logged = False
            if not self._relation_grad_fallback_logged:
                self.config.log(
                    "Relation gradient trace: pulled_ids missing; "
                    "falling back to embedding row indices (may be incorrect in distributed mode)."
                )
                self._relation_grad_fallback_logged = True
            rel_ids = torch.arange(
                grad_norms_cpu.numel(), dtype=torch.long, device="cpu"
            )
            rel_vals = grad_norms_cpu
        else:
            n_rows = int(min(grad_norms_cpu.numel(), pulled_ids_cpu.numel()))
            if grad_norms_cpu.numel() != pulled_ids_cpu.numel():
                if not hasattr(self, "_relation_grad_mismatch_logged"):
                    self._relation_grad_mismatch_logged = False
                if not self._relation_grad_mismatch_logged:
                    self.config.log(
                        f"Relation gradient trace: size mismatch grad_rows={grad_norms_cpu.numel()} "
                        f"pulled_ids={pulled_ids_cpu.numel()}; slicing to n={n_rows}."
                    )
                    self._relation_grad_mismatch_logged = True
            rel_ids = pulled_ids_cpu[:n_rows]
            rel_vals = grad_norms_cpu[:n_rows]
        mask = rel_vals != 0
        if not mask.any():
            return
        rel_ids = rel_ids[mask]
        rel_vals = rel_vals[mask]
        self._relation_grad_sum.index_add_(0, rel_ids, rel_vals)
        ones = torch.ones_like(rel_ids, dtype=self._relation_grad_count.dtype)
        self._relation_grad_count.index_add_(0, rel_ids, ones)

    def _flush_partition_gradient_trace(self):
        if not self._track_partition_gradients:
            return None
        if (
            self._current_partition_id is None
            or self._partition_grad_samples <= 0
        ):
            return None
        count = self._partition_grad_samples
        if self._partition_grad_sum_t is not None:
            grad_sum = float(self._partition_grad_sum_t.detach().cpu())
        else:
            grad_sum = float(self._partition_grad_sum)
        summary = {
            "grad_sum": grad_sum,
            "grad_count": count,
            "grad_avg": grad_sum / max(1, count),
        }
        partition_id = self._current_partition_id
        rel_ids = None
        rel_sums = None
        rel_counts = None
        rel_partitions = None
        if self._relation_gradient_trace and self._relation_grad_sum is not None:
            mask = self._relation_grad_count > 0
            if mask.any():
                rel_ids = torch.nonzero(mask, as_tuple=False).view(-1)
                rel_sums = self._relation_grad_sum[mask]
                rel_counts = self._relation_grad_count[mask]
                summary["relation_grad_count"] = int(rel_ids.numel())
                if self._relation_partition_map is not None:
                    rel_partitions = self._relation_partition_map.to(
                        device=rel_ids.device
                    )[rel_ids]
        if isinstance(partition_id, (list, tuple)):
            # Distribute the window's accumulated gradient across its member partitions.
            # IMPORTANT: a stratification stratum id like (i, j) is a single atomic work unit,
            # not a "window" of two partitions. Only split when _current_window_members is set.
            if self._current_window_members is not None:
                partition_ids = [int(x) for x in self._current_window_members.tolist()]
            else:
                partition_ids = [partition_id]
            per_sum = grad_sum / max(1, len(partition_ids))
            per_count = max(1, int(round(count / max(1, len(partition_ids)))))
            for pid in partition_ids:
                self.work_scheduler_client.register_partition_gradient(
                    pid, per_sum, per_count
                )
            if rel_ids is not None and rel_partitions is not None:
                for pid in torch.unique(rel_partitions).tolist():
                    pid_mask = rel_partitions == pid
                    if not torch.any(pid_mask):
                        continue
                    self.work_scheduler_client.register_partition_relation_gradient(
                        int(pid),
                        rel_ids[pid_mask],
                        rel_sums[pid_mask],
                        rel_counts[pid_mask],
                    )
        else:
            self.work_scheduler_client.register_partition_gradient(
                partition_id, grad_sum, count
            )
            if rel_ids is not None:
                self.work_scheduler_client.register_partition_relation_gradient(
                    partition_id, rel_ids, rel_sums, rel_counts
                )
        self._reset_partition_gradient_trace()
        self._last_partition_grad_summary = summary
        return summary

    def _prefetch_glow_window_entities(self, work_entities: torch.Tensor):
        start_time = time.time()
        stats = self._glow_trace_stats if self._glow_debug_trace else None
        if stats is not None:
            stats["prefetch_invocations"] += 1
        if (
            not self._prefetch_window_entities
            or not self.entity_localize
            or not self._window_work
            or self._current_window_entities is None
            or self._current_window_entities.numel() == 0
        ):
            if stats is not None:
                stats["prefetch_skipped_disabled"] += 1
            self._window_prefetch_key = None
            return time.time() - start_time
        window_key = None
        prefetch_key = None
        if (
            self._current_window_members is not None
            and self._current_window_members.numel() > 0
        ):
            window_key = tuple(
                int(x) for x in self._current_window_members.cpu().tolist()
            )
            prefetch_key = (self._current_partition_id, window_key)
            if prefetch_key == self._window_prefetch_key:
                if stats is not None:
                    stats["prefetch_skipped_repeat"] += 1
                return time.time() - start_time
        extras = self._current_window_entities
        if extras.device.type != "cpu":
            extras = extras.cpu()
        if work_entities is not None and work_entities.numel() > 0:
            work_cpu = work_entities
            if work_cpu.device.type != "cpu":
                work_cpu = work_cpu.cpu()
            try:
                mask = ~torch.isin(extras, work_cpu)
                extras = extras[mask]
            except Exception:
                self._window_prefetch_key = prefetch_key
                return time.time() - start_time
        # Skip prefetch when overlap is too low.
        try:
            union_count = int(self._current_window_entities.numel())
        except Exception:
            union_count = 0
        extras_count = int(extras.numel())
        overlap_ratio = (
            1.0 - (extras_count / max(1, union_count)) if union_count > 0 else 0.0
        )
        if stats is not None:
            stats["prefetch_overlap_samples"] += 1
            stats["prefetch_overlap_sum"] += overlap_ratio
            if stats["prefetch_overlap_min"] is None:
                stats["prefetch_overlap_min"] = overlap_ratio
                stats["prefetch_overlap_max"] = overlap_ratio
            else:
                stats["prefetch_overlap_min"] = min(
                    stats["prefetch_overlap_min"], overlap_ratio
                )
                stats["prefetch_overlap_max"] = max(
                    stats["prefetch_overlap_max"], overlap_ratio
                )
        min_overlap = float(
            self.config.get("job.distributed.glow.prefetch_min_overlap_ratio", 0.0)
        )
        if min_overlap > 0.0 and overlap_ratio < min_overlap:
            if stats is not None:
                stats["prefetch_skipped_overlap"] += 1
            self._window_prefetch_key = prefetch_key
            return time.time() - start_time
        # Throttle prefetch to every N windows.
        every_n = int(
            self.config.get("job.distributed.glow.prefetch_every_n_windows", 1)
        )
        self._glow_prefetch_window_count = getattr(
            self, "_glow_prefetch_window_count", 0
        ) + 1
        if every_n > 1 and (self._glow_prefetch_window_count % every_n) != 0:
            if stats is not None:
                stats["prefetch_skipped_throttle"] += 1
            self._window_prefetch_key = prefetch_key
            return time.time() - start_time
        max_prefetch = int(
            self.config.get(
                "job.distributed.glow.max_window_prefetch", 200000
            )
        )
        raw_extras = int(extras.numel())
        if raw_extras > max_prefetch:
            if stats is not None:
                stats["prefetch_capped_total"] += raw_extras - max_prefetch
            extras = extras[:max_prefetch]
        if extras.numel() == 0:
            if stats is not None:
                stats["prefetch_skipped_empty"] += 1
            self._window_prefetch_key = prefetch_key
            return time.time() - start_time
        if stats is not None:
            stats["prefetch_calls"] += 1
            stats["prefetch_extras_raw_total"] += raw_extras
            stats["prefetch_extras_total"] += int(extras.numel())
            stats["prefetch_extras_max"] = max(
                stats["prefetch_extras_max"], int(extras.numel())
            )
        self._dbg_glow_prefetch_calls = getattr(self, "_dbg_glow_prefetch_calls", 0) + 1
        if self._dbg_glow_prefetch_calls % 50 == 0:
            self.config.log(f"[DBG] glow_prefetch_calls={self._dbg_glow_prefetch_calls} extras_device={extras.device} extras_numel={extras.numel()}")

        try:
            # Ensure CPU ids to avoid GPU->CPU sync inside embedder.prefetch_window_pinned()
            extras_ids = extras.detach()
            if extras_ids.device.type != "cpu":
                extras_ids = extras_ids.cpu()
            extras_ids = extras_ids.to(dtype=torch.long)

            embedders = []
            try:
                embedders.append(self.model.get_s_embedder())
            except Exception:
                pass
            try:
                oemb = self.model.get_o_embedder()
                if len(embedders) == 0 or oemb is not embedders[0]:
                    embedders.append(oemb)
            except Exception:
                pass

            for embedder in embedders:
                if hasattr(embedder, "prefetch_window_pinned"):
                    embedder.prefetch_window_pinned(extras_ids, make_unique=False)
                else:
                    embedder.localize(extras_ids, asynchronous=True, make_unique=False)

            if window_key is not None:
                self.config.log(
                    f"Glow prelocalized {extras.numel()} overlapping entities for window {window_key}."
                )
        except Exception as exc:
            self.config.log(f"Glow window prefetch failed: {exc}")
        self._window_prefetch_key = prefetch_key
        return time.time() - start_time

    def _current_window_key(self):
        if (
            self._current_window_members is None
            or self._current_window_members.numel() == 0
        ):
            return None
        return tuple(int(x) for x in self._current_window_members.cpu().tolist())

    def _init_glow_trace_stats(self):
        self._glow_trace_stats = {
            "windows_total": 0,
            "windows_multi": 0,
            "window_members_total": 0,
            "window_entities_total": 0,
            "window_entities_max": 0,
            "prefetch_invocations": 0,
            "prefetch_calls": 0,
            "prefetch_skipped_disabled": 0,
            "prefetch_skipped_repeat": 0,
            "prefetch_skipped_overlap": 0,
            "prefetch_skipped_throttle": 0,
            "prefetch_skipped_empty": 0,
            "prefetch_extras_raw_total": 0,
            "prefetch_extras_total": 0,
            "prefetch_extras_max": 0,
            "prefetch_capped_total": 0,
            "prefetch_overlap_samples": 0,
            "prefetch_overlap_sum": 0.0,
            "prefetch_overlap_min": None,
            "prefetch_overlap_max": None,
        }

    def _validate_lookahead_payload(self, payload):
        if not payload:
            return False
        pid = payload.get("partition_id")
        if (
            pid is not None
            and self._current_partition_id is not None
            and pid != self._current_partition_id
        ):
            self._loader_debug_stats["lookahead_partition_miss"] += 1
            return False
        version = payload.get("partition_version")
        if (
            version is not None
            and self._current_partition_version is not None
            and version != self._current_partition_version
        ):
            self._loader_debug_stats["lookahead_version_miss"] += 1
            return False
        # window membership changes frequently due to Glow overlap scheduling; allow reuse
        # within the same partition/version even if the window shifted.
        return True

    def _prepare_negative_lookahead(self, batch):
        if (
            not self._lookahead_negatives
            or batch is None
            or "_lookahead_negative_samples" in batch
        ):
            return
        if not isinstance(self.device, str) or not self.device.startswith("cuda"):
            return
        lookahead_entries = []
        success = False
        context = {
            "partition_id": self._current_partition_id,
            "partition_version": self._current_partition_version,
        }
        window_key = self._current_window_key()
        if window_key is not None:
            context["window_key"] = window_key
        for ns in batch["negative_samples"]:
            payload = ns.build_lookahead_payload(
                self.device, non_blocking=self._non_blocking_transfer
            )
            if payload is None:
                lookahead_entries.append(None)
                continue
            payload.update(context)
            lookahead_entries.append(payload)
            success = True
        if success:
            self._loader_debug_stats["lookahead_prepared"] += 1
            payload_count = sum(1 for entry in lookahead_entries if entry is not None)
            self._loader_debug_stats["lookahead_payloads"] += payload_count
            batch["_lookahead_negative_samples"] = lookahead_entries
        elif self._lookahead_negatives:
            self._loader_debug_stats["lookahead_skipped"] += 1

    def _init_partition_maps(self):
        if self._partition_maps_ready:
            return
        num_partitions = int(self.config.get("job.distributed.num_partitions"))
        try:
            entity_map = self.dataset.load_entities_to_partitions(num_partitions)
            entity_map = np.asarray(entity_map).reshape(-1).astype(np.int64)
            if entity_map.size == 0:
                raise ValueError("empty entity partition map")
            self._entity_partition_map = torch.from_numpy(entity_map)
        except Exception as exc:
            self.config.log(
                f"Glow window_work: falling back to modulo entity map ({exc})."
            )
            self._entity_partition_map = (
                torch.arange(self.dataset.num_entities(), dtype=torch.long)
                % num_partitions
            )
        try:
            relation_map = self.dataset.load_relations_to_partitions(num_partitions)
            relation_map = np.asarray(relation_map).reshape(-1).astype(np.int64)
            if relation_map.size == 0:
                raise ValueError("empty relation partition map")
            self._relation_partition_map = torch.from_numpy(relation_map)
        except Exception as exc:
            self.config.log(
                f"Glow window_work: falling back to modulo relation map ({exc})."
            )
            self._relation_partition_map = (
                torch.arange(self.dataset.num_relations(), dtype=torch.long)
                % num_partitions
            )
        self._partition_maps_ready = True

    def _notify_partition_start(self):
        # Glow window work can use tuple partition ids (window keys). Only attach a
        # partition context when the id is a single integer partition id.
        if (
            isinstance(self._current_partition_id, (int, np.integer))
            and self._current_partition_version is not None
            and hasattr(self.optimizer, "set_partition_context")
        ):
            self.optimizer.set_partition_context(
                int(self._current_partition_id), int(self._current_partition_version)
            )
    
        # Versioned per-partition pushes are only needed when the optimizer actually performs
        # conflict-aware / causal merges. If those are disabled, routing every push by pid adds
        # a lot of overhead (extra masking + extra PS calls) and hurts epoch time.
        need_versioned_pushes = bool(
            getattr(self.optimizer, "conflict_free_merge", False)
            or getattr(self.optimizer, "causal_merge", False)
            or getattr(self.optimizer, "row_causal_merge", False)
        )
    
        if (
            need_versioned_pushes
            and self._current_window_members is not None
            and self._current_window_versions is not None
            and hasattr(self.optimizer, "set_partition_context_map")
        ):
            self._init_partition_maps()
            window_ids = [int(x) for x in self._current_window_members.tolist()]
            window_versions = [int(x) for x in self._current_window_versions.tolist()]
            version_map = {pid: ver for pid, ver in zip(window_ids, window_versions) if pid >= 0}
            self.optimizer.set_partition_context_map(
                version_map,
                entity_partition_map=self._entity_partition_map,
                relation_partition_map=self._relation_partition_map,
            )
    
        for embedder in (
            self.model.get_s_embedder(),
            self.model.get_p_embedder(),
        ):
            if (
                isinstance(self._current_partition_id, (int, np.integer))
                and self._current_partition_version is not None
                and hasattr(embedder, "set_partition_context")
            ):
                embedder.set_partition_context(
                    int(self._current_partition_id), int(self._current_partition_version)
                )


    def _notify_partition_end(self):
        finished = (self._current_partition_id, self._current_partition_version)
        if hasattr(self.optimizer, "finalize_partition_context"):
            self.optimizer.finalize_partition_context()
        if hasattr(self.optimizer, "clear_partition_context_map"):
            self.optimizer.clear_partition_context_map()
        for embedder in (
            self.model.get_s_embedder(),
            self.model.get_p_embedder(),
        ):
            if hasattr(embedder, "clear_partition_context"):
                embedder.clear_partition_context()
        self._last_finished_partition = finished
        self._current_window_members = None
        self._current_window_entities = None
        self._current_window_versions = None
        return finished

    def run_epoch(self) -> Dict[str, Any]:
        """ Runs an epoch and returns its trace entry. """

        # create initial trace entry
        self.current_trace["epoch"] = dict(
            type=self.type_str,
            scope="epoch",
            epoch=self.epoch,
            split=self.train_split,
            size=self.num_examples,
        )
        if not self.is_forward_only:
            self.current_trace["epoch"].update(
                lr=[group["lr"] for group in self.optimizer.param_groups],
            )

        # run pre-epoch hooks (may modify trace)
        for f in self.pre_epoch_hooks:
            f(self)

        trace_entry = None
        local_partition_counter = -1
        epoch_start_time = time.time()
        chunk_count = 0
        total_sum_loss = 0.0
        total_sum_penalty = 0.0
        total_sum_penalties = defaultdict(lambda: 0.0)
        total_examples = 0
        total_batches = 0
        total_expected_batches = 0
        total_prepare_time = 0.0
        total_forward_time = 0.0
        total_backward_time = 0.0
        total_optimizer_time = 0.0
        total_unique_time = 0.0
        total_pull_and_map_time = 0.0
        total_entity_pull_time = 0.0
        total_relation_pull_time = 0.0
        total_pre_pull_time = 0.0
        total_cpu_gpu_time = 0.0
        total_ps_wait_time = 0.0
        total_ps_set_time = 0.0
        total_dataloader_time = 0.0
        total_scheduler_time = 0.0
        total_chunk_time = 0.0
        total_embedding_mapping_time = 0.0
        total_grad_sum = 0.0
        total_grad_samples = 0
        total_relation_grad_samples = 0
        total_gpu_cache_s_hits = 0
        total_gpu_cache_s_misses = 0
        total_gpu_cache_o_hits = 0
        total_gpu_cache_o_misses = 0
        total_glow_window_union_time = 0.0
        total_glow_prefetch_time = 0.0
        total_glow_gradient_trace_time = 0.0
        total_glow_register_result_time = 0.0
        total_glow_conflict_time = 0.0
        profile_interval_batches = self.config.get("train.profile_interval_batches")
        profile_stats = defaultdict(float)
        profile_batch_counter = 0
        profile_total_batches = 0
        if self._glow_debug_trace:
            self._init_glow_trace_stats()

        def maybe_log_profile(force=False):
            nonlocal profile_stats, profile_batch_counter, profile_total_batches
            if profile_interval_batches <= 0:
                return
            if not force and profile_batch_counter < profile_interval_batches:
                return
            if profile_batch_counter == 0:
                return
            start = profile_total_batches - profile_batch_counter + 1
            end = profile_total_batches
            compute_time = (
                profile_stats["prepare"]
                + profile_stats["forward"]
                + profile_stats["backward"]
                + profile_stats["optimizer"]
            )
            total_time = compute_time + profile_stats["dataloader"]
            throughput = (
                profile_stats["examples"] / total_time if total_time > 0 else float("inf")
            )
            self.config.log(
                (
                    "[profile] batches {start}-{end}: {throughput:.1f} triples/s; "
                    "times(s) -> dataloader={data:.3f}, prepare={prep:.3f}, "
                    "forward={fwd:.3f}, backward={bwd:.3f}, optimizer={opt:.3f}, "
                    "entity_pull={epull:.3f}, relation_pull={rpull:.3f}, "
                    "pull_map={pull_map:.3f}"
                ).format(
                    start=start,
                    end=end,
                    throughput=throughput,
                    data=profile_stats["dataloader"],
                    prep=profile_stats["prepare"],
                    fwd=profile_stats["forward"],
                    bwd=profile_stats["backward"],
                    opt=profile_stats["optimizer"],
                    epull=profile_stats["entity_pull"],
                    rpull=profile_stats["relation_pull"],
                    pull_map=profile_stats["pull_and_map"],
                )
            )
            dataset_stats = {}
            if getattr(self, "dataloader_dataset", None) is not None:
                dataset_stats = self.dataloader_dataset.collect_debug_stats()
            loader_stats = dict(self._loader_debug_stats)
            self._loader_debug_stats.clear()
            debug_messages = []
            if dataset_stats:
                debug_messages.append(
                    "dataset=" + ", ".join(
                        f"{k}:{v}" for k, v in sorted(dataset_stats.items())
                    )
                )
            if loader_stats:
                debug_messages.append(
                    "lookahead=" + ", ".join(
                        f"{k}:{v}" for k, v in sorted(loader_stats.items())
                    )
                )
            if debug_messages:
                self.config.log("[profile-debug] " + " | ".join(debug_messages))
            profile_stats = defaultdict(float)
            profile_batch_counter = 0
        while True:
            # variables that record various statitics
            sum_loss = 0.0
            sum_penalty = 0.0
            sum_penalties = defaultdict(lambda: 0.0)
            chunk_time = -time.time()
            prepare_time = 0.0
            forward_time = 0.0
            backward_time = 0.0
            optimizer_time = 0.0
            unique_time = 0.0
            pull_and_map_time = 0.0
            entity_pull_time = 0.0
            relation_pull_time = 0.0
            pre_pull_time = 0.0
            cpu_gpu_time = 0.0
            ps_wait_time = 0.0
            ps_set_time = 0.0
            dataloader_time = 0.0
            scheduler_time = -time.time()
            glow_window_union_time = 0.0
            glow_prefetch_time = 0.0
            glow_gradient_trace_time = 0.0
            glow_register_result_time = 0.0
            glow_conflict_time = 0.0

            # load new work package
            (
                work,
                work_entities,
                work_relations,
                current_partition_id,
                current_partition_version,
                window_members,
                window_entities,
                window_versions,
            ) = self.work_scheduler_client.get_work()
            if work_entities is not None:
                work_entities = work_entities.long()
            if work_relations is not None:
                work_relations = work_relations.long()
            if self.relation_sync_level == "partition":
                work_relations = self._ensure_partition_relations(work, work_relations)
            self.entity_partition_localized = False
            self.relation_partition_localized = False
            if work is None:
                self.config.log(
                    "No work received from scheduler; ending epoch early."
                )
                break
            if work.numel() == 0:
                window_members_list = None
                if window_members is not None and len(window_members) > 0:
                    try:
                        window_members_list = [
                            int(x) for x in window_members.tolist()
                        ]
                    except Exception:
                        window_members_list = str(window_members)
                ent_count = int(work_entities.numel()) if work_entities is not None else 0
                rel_count = int(work_relations.numel()) if work_relations is not None else 0
                window_ent_count = (
                    int(window_entities.numel()) if window_entities is not None else 0
                )
                self.config.log(
                    "Received empty work package "
                    f"(partition_id={current_partition_id}, "
                    f"window_members={window_members_list}, "
                    f"version={current_partition_version}, "
                    f"entities={ent_count}, relations={rel_count}, "
                    f"window_entities={window_ent_count})."
                )
            self._current_partition_size = int(work.numel())
            if window_members is not None and len(window_members) > 0:
                self._current_window_members = window_members.long()
            else:
                self._current_window_members = None
            if window_entities is not None and len(window_entities) > 0:
                # Scheduler already deduplicates window entities.
                self._current_window_entities = window_entities.long()
            else:
                self._current_window_entities = None
            if window_versions is not None and len(window_versions) > 0:
                self._current_window_versions = window_versions.long()
            else:
                self._current_window_versions = None
            if current_partition_id is not None:
                if isinstance(current_partition_id, (int, np.integer)):
                    if current_partition_id >= 0:
                        self._current_partition_id = int(current_partition_id)
                    else:
                        if window_members is not None and len(window_members) > 0:
                            self._current_partition_id = tuple(
                                int(x) for x in window_members.tolist()
                            )
                        else:
                            self._current_partition_id = None
                else:
                    if isinstance(current_partition_id, torch.Tensor):
                        current_partition_id = current_partition_id.tolist()
                    self._current_partition_id = tuple(
                        int(x) for x in current_partition_id
                    )
            elif window_members is not None and len(window_members) > 0:
                self._current_partition_id = tuple(
                    int(x) for x in window_members.tolist()
                )
            else:
                self._current_partition_id = None
            if current_partition_version is not None and current_partition_version >= 0:
                self._current_partition_version = int(current_partition_version)
            else:
                self._current_partition_version = None
            self._notify_partition_start()
            self._reset_partition_gradient_trace()
            local_partition_counter += 1
            chunk_index = local_partition_counter
            self.work_pre_localized = False
            partition_start_time = time.time()
            entity_pull_ids = work_entities
            pool_entities = work_entities
            if (
                work_entities is not None
                and self.config.get("negative_sampling.sampling_type") == "pooled"
                and self._current_window_entities is not None
                and self._current_window_entities.numel() > 0
                and (
                    self._overlap_negative_sampling
                    or self.entity_sync_level != "partition"
                )
            ):
                max_pool_entities = int(
                    self.config.get(
                        "job.distributed.glow.max_pool_entities", 300000
                    )
                )
                if (
                    work_entities.numel()
                    + self._current_window_entities.numel()
                    <= max_pool_entities
                ):
                    union_start = time.time()
                    pool_entities = torch.unique(
                        torch.cat(
                            (work_entities, self._current_window_entities)
                        )
                    )
                    glow_window_union_time += time.time() - union_start
                else:
                    pool_entities = work_entities
                if self.entity_sync_level == "partition":
                    max_vocab = self.model.get_s_embedder().vocab_size
                    if pool_entities.numel() <= max_vocab:
                        entity_pull_ids = pool_entities
                    else:
                        if not self._overlap_sampling_logged:
                            self.config.log(
                                "Glow overlap negative sampling disabled for "
                                "partition-level sync because pool size "
                                f"{pool_entities.numel()} exceeds embedder "
                                f"vocab size {max_vocab}."
                            )
                            self._overlap_sampling_logged = True
                        pool_entities = work_entities
            if work_entities is not None and self.entity_localize:
                self.model.get_s_embedder().localize(entity_pull_ids)
                self.entity_partition_localized = True
            if work_relations is not None and self.relation_localize:
                self.model.get_p_embedder().localize(work_relations)
                self.relation_partition_localized = True
            if self.entity_sync_level == "partition":
                if work_entities is not None:
                    entity_pull_time -= time.time()
                    actual_entity_pull_time, entity_cpu_gpu_time = self.model.get_s_embedder()._pull_embeddings(
                        entity_pull_ids
                    )
                    self.model.get_s_embedder().global_to_local_mapper[
                        entity_pull_ids
                    ] = torch.arange(
                        len(entity_pull_ids), dtype=torch.long, device="cpu"
                    )
                    if self._map_ids_device.type == "cuda" and not self._stage_local_ids:
                        entity_mapper_device = self._ensure_partition_mapper_device(
                            "entity"
                        )
                        if entity_mapper_device is not None:
                            work_entities_device = entity_pull_ids.to(
                                self._map_ids_device, non_blocking=True
                            )
                            entity_mapper_device[work_entities_device] = torch.arange(
                                len(entity_pull_ids),
                                dtype=torch.long,
                                device=self._map_ids_device,
                            )
                    entity_pull_time += time.time()
                    cpu_gpu_time += entity_cpu_gpu_time
                else:
                    raise ValueError(
                        "the used work-scheduler seems not to support "
                        "syncing entities on a partition level"
                    )
            if self.relation_sync_level == "partition":
                if work_relations is not None:
                    relation_pull_time -= time.time()
                    actual_relation_pull_time, relation_cpu_gpu_time = self.model.get_p_embedder()._pull_embeddings(work_relations)
                    self.model.get_p_embedder().global_to_local_mapper[
                        work_relations
                    ] = torch.arange(
                        len(work_relations), dtype=torch.long, device="cpu"
                    )
                    if self._map_ids_device.type == "cuda" and not self._stage_local_ids:
                        relation_mapper_device = self._ensure_partition_mapper_device(
                            "relation"
                        )
                        if relation_mapper_device is not None:
                            work_relations_device = work_relations.to(
                                self._map_ids_device, non_blocking=True
                            )
                            relation_mapper_device[work_relations_device] = torch.arange(
                                len(work_relations),
                                dtype=torch.long,
                                device=self._map_ids_device,
                            )
                    relation_pull_time += time.time()
                    cpu_gpu_time += relation_cpu_gpu_time
                else:
                    raise ValueError(
                        "the used work-scheduler seems not to support "
                        "syncing relations on a partition level"
                    )

            glow_prefetch_time += (
                self._prefetch_glow_window_entities(work_entities) or 0.0
            )

            if work_entities is not None and self._sampler.uses_pool():
                pool_entities = (
                    pool_entities if pool_entities is not None else work_entities
                )
                if self._stage_local_ids:
                    pool_entities = torch.arange(
                        len(entity_pull_ids), dtype=torch.long
                    )
                self.local_entities = pool_entities
                pool_device = None
                if self._materialize_partitions and self.device.startswith("cuda"):
                    try:
                        pool_device = pool_entities.to(self.device, non_blocking=True)
                    except RuntimeError:
                        pool_device = None
                self._sampler.set_pool(pool_entities, S)
                self._sampler.set_pool(pool_entities, O)
                if pool_device is not None:
                    self._sampler.set_pool(pool_device, S)
                    self._sampler.set_pool(pool_device, O)
            if self._stage_local_ids and work_entities is not None:
                local_entities_count = len(entity_pull_ids)
                self._sampler.vocabulary_size[S] = local_entities_count
                self._sampler.vocabulary_size[O] = local_entities_count
                if work_relations is not None:
                    self._sampler.vocabulary_size[P] = len(work_relations)
            self.dataloader_dataset.set_samples(
                work,
                self.epoch,
                local_partition_counter,
                partition_version=self._current_partition_version,
                entity_mapper=(
                    self.model.get_s_embedder().global_to_local_mapper
                    if self._stage_local_ids
                    else None
                ),
                relation_mapper=(
                    self.model.get_p_embedder().global_to_local_mapper
                    if self._stage_local_ids
                    else None
                ),
                stage_local_ids=self._stage_local_ids,
            )
            if (
                self._materialize_partitions
                and not self._materialization_notice_logged
                and self.dataloader_dataset.materialization_count() > 0
            ):
                self.config.log(
                    f"Materialized {self.dataloader_dataset.materialized_size} triples "
                    f"for partition {local_partition_counter}; reusing staged buffer for batches."
                )
                self._materialization_notice_logged = True
            if self._use_materialized_iterator():
                self._materialized_iterator = MaterializedBatchIterator(
                    self.dataloader_dataset, self._collate_fn
                )
            if self._use_materialized_iterator():
                self.loader = None
                self.iter_dataloader = None
            else:
                if self.loader is None:
                    self._init_dataloader()
                    self.iter_dataloader = iter(self.loader)
                object.__setattr__(
                    self.loader,
                    "sampler",
                    InfiniteSequentialSampler(self.dataloader_dataset),
                )
                object.__setattr__(
                    self.iter_dataloader,
                    "_sampler_iter",
                    iter(self.iter_dataloader._index_sampler),
                )
            scheduler_time += time.time()

            # process each batch
            pre_load_batches = deque()
            batch_index = 0
            prefetch_enabled = bool(
                self._lookahead_negatives
                or self.entity_pre_pull > 0
                or self.relation_pre_pull > 0
                or self.pre_localize_batch > 0
            )
            num_prepulls = (
                max(self.entity_pre_pull, self.relation_pre_pull, self.pre_localize_batch, 1)
                if prefetch_enabled
                else 0
            )
            use_materialized_iter = self._use_materialized_iterator() and self._materialized_iterator is not None
            materialized_iter = iter(self._materialized_iterator) if use_materialized_iter else None
            chunk_examples = 0
            chunk_batches = 0
            while batch_index < len(self.dataloader_dataset):
                exhausted = False
                while len(pre_load_batches) < num_prepulls + 1:
                    prepare_start = time.time()
                    load_start = time.time()
                    dataloader_time -= load_start
                    if use_materialized_iter:
                        try:
                            next_batch = next(materialized_iter)
                        except StopIteration:
                            next_batch = None
                            exhausted = True
                        load_end = time.time()
                        dataloader_time += load_end
                        if profile_interval_batches > 0 and next_batch is not None:
                            profile_stats["dataloader"] += load_end - load_start
                        if next_batch is None:
                            prepare_time += time.time() - prepare_start
                            break
                        if not self._stage_local_ids:
                            next_batch = self._map_ids_to_local(next_batch)
                        else:
                            next_batch = self._maybe_remap_staged_batch(next_batch)
                    else:
                        next_batch = next(self.iter_dataloader)
                        while next_batch["epoch"] != self.epoch or next_batch[
                            "local_partition_id"] != local_partition_counter:
                            next_batch = next(self.iter_dataloader)
                        if not self._stage_local_ids:
                            next_batch = self._map_ids_to_local(next_batch)
                        else:
                            next_batch = self._maybe_remap_staged_batch(next_batch)
                        load_end = time.time()
                        dataloader_time += load_end
                        if profile_interval_batches > 0:
                            profile_stats["dataloader"] += load_end - load_start
                    if exhausted:
                        prepare_time += time.time() - prepare_start
                        break
                    if next_batch is not None and "triple_ids" in next_batch:
                        if next_batch.get("_gpu_triples") is not None:
                            self._loader_debug_stats["gpu_triple_batches"] += 1
                        else:
                            device_view = self.dataloader_dataset.write_triples_to_device(
                                next_batch["triple_ids"], next_batch["triples"], self.device
                            )
                            if device_view is not None:
                                next_batch["_gpu_triples"] = device_view
                                self._loader_debug_stats["gpu_triple_batches"] += 1
                            else:
                                self._loader_debug_stats["cpu_triple_batches"] += 1
                    pre_load_batches.append(next_batch)
                    pre_pull_time -= time.time()
                    if prefetch_enabled and next_batch is not None:
                        self._prepare_batch_ahead(pre_load_batches, new_batch=next_batch)
                    pre_pull_time += time.time()
                    prepare_time += time.time() - prepare_start
                if exhausted and not pre_load_batches:
                    break
                if prefetch_enabled:
                    self._prepare_batch_ahead(pre_load_batches)
                batch = pre_load_batches.popleft()

                # create initial batch trace (yet incomplete)
                total_batches = (
                    len(self.loader)
                    if self.loader is not None
                    else self.dataloader_dataset.get_real_len()
                )
                self.current_trace["batch"] = {
                    "type": self.type_str,
                    "scope": "batch",
                    "epoch": self.epoch,
                    "split": self.train_split,
                    "batch": batch_index,
                    "batches": total_batches,
                }
                if not self.is_forward_only:
                    self.current_trace["batch"].update(
                        lr=[group["lr"] for group in self.optimizer.param_groups],
                    )

                # run the pre-batch hooks (may update the trace)
                for f in self.pre_batch_hooks:
                    f(self, batch_index)

                # process batch (preprocessing + forward pass + backward pass on loss)
                batch_result: TrainingJob._ProcessBatchResult = self._auto_subbatched_process_batch(
                    batch_index, batch
                )
                sum_loss += batch_result.avg_loss * batch_result.size
                chunk_examples += batch_result.size

                # determine penalty terms (forward pass)
                batch_forward_time = batch_result.forward_time - time.time()
                penalties_torch = self.model.penalty(
                    epoch=self.epoch,
                    batch_index=batch_index,
                    num_batches=self.dataloader_dataset.get_real_len(),
                    batch=batch,
                )
                batch_forward_time += time.time()

                # backward pass on penalties
                batch_backward_time = batch_result.backward_time - time.time()
                penalty = 0.0
                for index, (penalty_key, penalty_value_torch) in enumerate(
                    penalties_torch
                ):
                    if not self.is_forward_only:
                        penalty_value_torch.backward()
                    penalty += penalty_value_torch.item()
                    sum_penalties[penalty_key] += penalty_value_torch.item()
                sum_penalty += penalty
                batch_backward_time += time.time()
                self._accumulate_partition_gradient()

                # determine full cost
                cost_value = batch_result.avg_loss + penalty

                # abort on nan
                if self.abort_on_nan and math.isnan(cost_value):
                    raise FloatingPointError("Cost became nan, aborting training job")

                # print memory stats
                if self.epoch == 1 and batch_index == 0:
                    if self.device.startswith("cuda"):
                        with torch.cuda.device(self.device):
                            self.config.log(
                                "CUDA memory after first batch: allocated={:14,} "
                                "reserved={:14,} max_allocated={:14,}".format(
                                    torch.cuda.memory_allocated(self.device),
                                    torch.cuda.memory_reserved(self.device),
                                    torch.cuda.max_memory_allocated(self.device),
                                )
                            )

                # update parameters
                batch_optimizer_time = -time.time()
                if not self.is_forward_only:
                    self.optimizer.step()
                batch_optimizer_time += time.time()

                if self.entity_sync_level == "batch":
                    self.model.get_s_embedder().push_back()
                if self.relation_sync_level == "batch":
                    self.model.get_p_embedder().push_back()

                # update batch trace with the results
                self.current_trace["batch"].update(
                    {
                        "size": batch_result.size,
                        "avg_loss": batch_result.avg_loss,
                        # "penalties": [p.item() for k, p in penalties_torch],
                        "penalty": penalty,
                        "cost": cost_value,
                        "prepare_time": batch_result.prepare_time,
                        "forward_time": batch_forward_time,
                        "backward_time": batch_backward_time,
                        "optimizer_time": batch_optimizer_time,
                        "event": "batch_completed",
                    }
                )

                # run the post-batch hooks (may modify the trace)
                for f in self.post_batch_hooks:
                    f(self)

                # output, then clear trace
                if self.trace_batch:
                    self.trace(**self.current_trace["batch"])
                self.current_trace["batch"] = None

                # print console feedback
                self.config.print(
                    (
                        "\r"  # go back
                        + "{}  batch{: "
                        + str(1 + int(math.ceil(math.log10(self.dataloader_dataset.get_real_len()))))
                        + "d}/{}"
                        + ", avg_loss {:.4E}, penalty {:.4E}, cost {:.4E}, time {:6.2f}s"
                        + "\033[K"  # clear to right
                    ).format(
                        self.config.log_prefix,
                        batch_index,
                        self.dataloader_dataset.get_real_len() - 1,
                        batch_result.avg_loss,
                        penalty,
                        cost_value,
                        batch_result.prepare_time
                        + batch_forward_time
                        + batch_backward_time
                        + batch_optimizer_time,
                    ),
                    end="",
                    flush=True,
                )

                # update epoch times
                prepare_time += batch_result.prepare_time
                forward_time += batch_forward_time
                backward_time += batch_backward_time
                optimizer_time += batch_optimizer_time
                pull_and_map_time += batch_result.pull_and_map_time
                entity_pull_time += batch_result.entity_pull_time
                relation_pull_time += batch_result.relation_pull_time
                unique_time += batch_result.unique_time
                cpu_gpu_time += batch_result.cpu_gpu_time
                ps_wait_time += batch_result.ps_wait_time

                batch_index += 1
                chunk_batches += 1

                if profile_interval_batches > 0:
                    profile_batch_counter += 1
                    profile_total_batches += 1
                    profile_stats["examples"] += batch_result.size
                    profile_stats["prepare"] += batch_result.prepare_time
                    profile_stats["forward"] += batch_forward_time
                    profile_stats["backward"] += batch_backward_time
                    profile_stats["optimizer"] += batch_optimizer_time
                    profile_stats["pull_and_map"] += batch_result.pull_and_map_time
                    profile_stats["entity_pull"] += batch_result.entity_pull_time
                    profile_stats["relation_pull"] += batch_result.relation_pull_time
                    profile_stats["unique"] += batch_result.unique_time
                    profile_stats["cpu_gpu"] += batch_result.cpu_gpu_time
                    profile_stats["ps_wait"] += batch_result.ps_wait_time
                    maybe_log_profile()

            # all done; now trace and log
            chunk_time += time.time()
            self.config.print("\033[2K\r", end="", flush=True)  # clear line and go back

            other_time = (
                chunk_time
                - prepare_time
                - forward_time
                - backward_time
                - optimizer_time
                - scheduler_time
            )

            if self.entity_sync_level == "partition":
                ps_set_time -= time.time()
                if self._skip_partition_set:
                    if hasattr(self.optimizer, "flush_pending_pushes"):
                        self.optimizer.flush_pending_pushes()
                else:
                    self.model.get_s_embedder().set_embeddings()
                ps_set_time += time.time()
                # this is expensive and unnecessary
                # self.model.get_s_embedder().global_to_local_mapper[:] = -1
                self.model.get_s_embedder().push_back()
            if self.relation_sync_level == "partition":
                ps_set_time -= time.time()
                if self._skip_partition_set:
                    if hasattr(self.optimizer, "flush_pending_pushes"):
                        self.optimizer.flush_pending_pushes()
                else:
                    self.model.get_p_embedder().set_embeddings()
                ps_set_time += time.time()
                # self.model.get_p_embedder().global_to_local_mapper[:] = -1
                self.model.get_p_embedder().push_back()
            partition_duration = time.time() - partition_start_time
            chunk_partition_id = self._current_partition_id
            chunk_partition_version = self._current_partition_version
            chunk_partition_size = self._current_partition_size
            chunk_window_members_count = (
                len(self._current_window_members)
                if self._current_window_members is not None
                else 0
            )
            chunk_window_entities_count = (
                len(self._current_window_entities)
                if self._current_window_entities is not None
                else 0
            )
            if self._glow_debug_trace and self._glow_trace_stats is not None:
                stats = self._glow_trace_stats
                if chunk_window_members_count > 0:
                    stats["windows_total"] += 1
                    stats["window_members_total"] += int(chunk_window_members_count)
                    stats["window_entities_total"] += int(chunk_window_entities_count)
                    stats["window_entities_max"] = max(
                        stats["window_entities_max"], int(chunk_window_entities_count)
                    )
                    if chunk_window_members_count > 1:
                        stats["windows_multi"] += 1
            grad_trace_start = time.time()
            chunk_grad_summary = self._flush_partition_gradient_trace()
            glow_gradient_trace_time += time.time() - grad_trace_start
            window_members = self._current_window_members
            window_versions = self._current_window_versions
            finished_context = self._notify_partition_end()
            if (
                window_members is not None
                and window_versions is not None
                and hasattr(self.work_scheduler_client, "register_window_result")
            ):
                if self._debug_window_versions_remaining > 0:
                    try:
                        members = [
                            int(x) for x in window_members.tolist()
                        ]
                        versions = [
                            int(x) for x in window_versions.tolist()
                        ]
                        self.config.log(
                            f"Glow window versions {members} -> {versions}"
                        )
                    except Exception as exc:
                        self.config.log(
                            f"Glow window versions debug failed: {exc}"
                        )
                    self._debug_window_versions_remaining -= 1
                register_start = time.time()
                conflicts = self.work_scheduler_client.register_window_result(
                    partition_duration,
                    window_members,
                    window_versions,
                    chunk_partition_size,
                )
                glow_register_result_time += time.time() - register_start
                conflict_start = time.time()
                self._handle_window_conflicts(conflicts)
                glow_conflict_time += time.time() - conflict_start
            else:
                register_start = time.time()
                status, replay_version = self.work_scheduler_client.register_partition_result(
                    partition_duration,
                    self._current_partition_version,
                    self._current_partition_size,
                )
                glow_register_result_time += time.time() - register_start
                conflict_start = time.time()
                self._handle_partition_conflict(status, replay_version, finished_context)
                glow_conflict_time += time.time() - conflict_start
            self._last_finished_partition = None
            self._current_partition_id = None
            self._current_partition_version = None
            self._current_partition_size = 0
            self.work_scheduler_client.work_done()

            chunk_batches_total = (
                len(self.loader)
                if self.loader is not None
                else self.dataloader_dataset.get_real_len()
            )
            chunk_loss_divisor = max(1, chunk_examples)
            chunk_penalty_divisor = max(1, chunk_batches)
            chunk_trace = dict(self.current_trace["epoch"])
            chunk_embedding_mapping_time = (
                self.model.get_s_embedder().mapping_time
                + self.model.get_p_embedder().mapping_time
            )
            chunk_trace.update(
                dict(
                    scope="chunk",
                    event="chunk_completed",
                    chunk_index=chunk_index,
                    partition_id=chunk_partition_id,
                    partition_version=chunk_partition_version,
                    partition_size=chunk_partition_size,
                    window_members_count=chunk_window_members_count,
                    window_entities_count=chunk_window_entities_count,
                    processed_examples=chunk_examples,
                    processed_batches=chunk_batches,
                    avg_loss=sum_loss / chunk_loss_divisor,
                    avg_penalty=sum_penalty / chunk_penalty_divisor,
                    avg_penalties={
                        k: p / chunk_penalty_divisor for k, p in sum_penalties.items()
                    },
                    avg_cost=sum_loss / chunk_loss_divisor
                    + sum_penalty / chunk_penalty_divisor,
                    chunk_time=chunk_time,
                    prepare_time=prepare_time,
                    ps_wait_time=ps_wait_time,
                    unique_time=unique_time,
                    pull_and_map_time=pull_and_map_time,
                    pre_pull_time=pre_pull_time,
                    entity_pull_time=entity_pull_time,
                    relation_pull_time=relation_pull_time,
                    ps_set_time=ps_set_time,
                    cpu_gpu_time=cpu_gpu_time,
                    forward_time=forward_time,
                    backward_time=backward_time,
                    optimizer_time=optimizer_time,
                    scheduler_time=scheduler_time,
                    other_time=other_time,
                    glow_window_union_time=glow_window_union_time,
                    glow_prefetch_time=glow_prefetch_time,
                    glow_gradient_trace_time=glow_gradient_trace_time,
                    glow_register_result_time=glow_register_result_time,
                    glow_conflict_time=glow_conflict_time,
                    embedding_mapping_time=chunk_embedding_mapping_time,
                    batches=chunk_batches_total,
                    dataloader_time=dataloader_time,
                )
            )
            gpu_cache_stats = None
            embedder_s = self.model.get_s_embedder()
            if hasattr(embedder_s, "get_and_reset_gpu_cache_stats"):
                gpu_cache_stats = embedder_s.get_and_reset_gpu_cache_stats()
            if gpu_cache_stats:
                chunk_trace.update(
                    dict(
                        gpu_cache_hits=gpu_cache_stats.get("hits", 0),
                        gpu_cache_misses=gpu_cache_stats.get("misses", 0),
                    )
                )
                total_gpu_cache_s_hits += int(gpu_cache_stats.get("hits", 0))
                total_gpu_cache_s_misses += int(gpu_cache_stats.get("misses", 0))
            # Collect o-embedder stats if it is a distinct embedder.
            try:
                embedder_o = self.model.get_o_embedder()
            except Exception:
                embedder_o = None
            if (
                embedder_o is not None
                and embedder_o is not embedder_s
                and hasattr(embedder_o, "get_and_reset_gpu_cache_stats")
            ):
                o_stats = embedder_o.get_and_reset_gpu_cache_stats()
                if o_stats:
                    total_gpu_cache_o_hits += int(o_stats.get("hits", 0))
                    total_gpu_cache_o_misses += int(o_stats.get("misses", 0))
            if chunk_grad_summary:
                chunk_trace.update(
                    dict(
                        grad_sum=chunk_grad_summary["grad_sum"],
                        grad_samples=chunk_grad_summary["grad_count"],
                        grad_avg=chunk_grad_summary["grad_avg"],
                        relation_grad_samples=chunk_grad_summary.get(
                            "relation_grad_count", 0
                        ),
                    )
                )
            trace_entry = self.trace(**chunk_trace, echo=False, log=True)

            total_sum_loss += sum_loss
            total_sum_penalty += sum_penalty
            for k, p in sum_penalties.items():
                total_sum_penalties[k] += p
            total_examples += chunk_examples
            total_batches += chunk_batches
            total_expected_batches += chunk_batches_total
            total_prepare_time += prepare_time
            total_forward_time += forward_time
            total_backward_time += backward_time
            total_optimizer_time += optimizer_time
            total_unique_time += unique_time
            total_pull_and_map_time += pull_and_map_time
            total_entity_pull_time += entity_pull_time
            total_relation_pull_time += relation_pull_time
            total_pre_pull_time += pre_pull_time
            total_cpu_gpu_time += cpu_gpu_time
            total_ps_wait_time += ps_wait_time
            total_ps_set_time += ps_set_time
            total_dataloader_time += dataloader_time
            total_scheduler_time += scheduler_time
            total_chunk_time += chunk_time
            total_embedding_mapping_time += chunk_embedding_mapping_time
            if chunk_grad_summary:
                total_grad_sum += chunk_grad_summary["grad_sum"]
                total_grad_samples += chunk_grad_summary["grad_count"]
                total_relation_grad_samples += chunk_grad_summary.get(
                    "relation_grad_count", 0
                )
            total_glow_window_union_time += glow_window_union_time
            total_glow_prefetch_time += glow_prefetch_time
            total_glow_gradient_trace_time += glow_gradient_trace_time
            total_glow_register_result_time += glow_register_result_time
            total_glow_conflict_time += glow_conflict_time
            chunk_count += 1
            self.model.get_p_embedder().mapping_time = 0.0
            self.model.get_s_embedder().mapping_time = 0.0

            # run hooks (may modify trace)
            for f in self.post_epoch_hooks:
                f(self)

        epoch_time = time.time() - epoch_start_time
        epoch_loss_divisor = max(1, total_examples)
        epoch_penalty_divisor = max(1, total_batches)
        duplication_factor = (
            (total_examples / self.num_examples) if self.num_examples > 0 else 0.0
        )
        pull_stats = None
        push_stats = None
        if hasattr(self.parameter_client, "get_and_reset_pull_stats"):
            try:
                pull_stats = self.parameter_client.get_and_reset_pull_stats()
            except Exception:
                pull_stats = None
        if hasattr(self.parameter_client, "get_and_reset_push_stats"):
            try:
                push_stats = self.parameter_client.get_and_reset_push_stats()
            except Exception:
                push_stats = None
        # --- GPU cache stats (best-effort; never crash training) ---
        try:
            stats = {}
            if total_gpu_cache_s_hits or total_gpu_cache_s_misses:
                total = total_gpu_cache_s_hits + total_gpu_cache_s_misses
                hit_rate = (total_gpu_cache_s_hits / total) if total > 0 else 0.0
                stats["gpu_cache_s_hits"] = total_gpu_cache_s_hits
                stats["gpu_cache_s_misses"] = total_gpu_cache_s_misses
                stats["gpu_cache_s_hit_rate"] = float(hit_rate)
            if total_gpu_cache_o_hits or total_gpu_cache_o_misses:
                total = total_gpu_cache_o_hits + total_gpu_cache_o_misses
                hit_rate = (total_gpu_cache_o_hits / total) if total > 0 else 0.0
                stats["gpu_cache_o_hits"] = total_gpu_cache_o_hits
                stats["gpu_cache_o_misses"] = total_gpu_cache_o_misses
                stats["gpu_cache_o_hit_rate"] = float(hit_rate)

            if stats:
                # put into trace
                try:
                    self.current_trace["epoch"].update(stats)
                except Exception:
                    pass

                # log
                msg = " | ".join(
                    f"{k}={v:.4f}" if k.endswith("hit_rate") else f"{k}={v}"
                    for k, v in stats.items()
                )
                try:
                    self.config.log(f"GPU-cache stats: {msg}")
                except Exception:
                    print(f"GPU-cache stats: {msg}")
        except Exception as _exc:
            # never fail training due to stats
            pass

        if self._glow_debug_trace and self._glow_trace_stats is not None:
            stats = self._glow_trace_stats
            windows_total = stats["windows_total"]
            prefetch_calls = stats["prefetch_calls"]
            overlap_samples = stats["prefetch_overlap_samples"]
            avg_members = (
                stats["window_members_total"] / max(1, windows_total)
                if windows_total > 0
                else 0.0
            )
            avg_entities = (
                stats["window_entities_total"] / max(1, windows_total)
                if windows_total > 0
                else 0.0
            )
            avg_overlap = (
                stats["prefetch_overlap_sum"] / max(1, overlap_samples)
                if overlap_samples > 0
                else 0.0
            )
            avg_extras = (
                stats["prefetch_extras_total"] / max(1, prefetch_calls)
                if prefetch_calls > 0
                else 0.0
            )
            glow_trace = {
                "glow_window_count": windows_total,
                "glow_window_multi_count": stats["windows_multi"],
                "glow_window_avg_members": float(avg_members),
                "glow_window_avg_entities": float(avg_entities),
                "glow_window_max_entities": int(stats["window_entities_max"]),
                "glow_prefetch_invocations": stats["prefetch_invocations"],
                "glow_prefetch_calls": prefetch_calls,
                "glow_prefetch_skipped_disabled": stats["prefetch_skipped_disabled"],
                "glow_prefetch_skipped_repeat": stats["prefetch_skipped_repeat"],
                "glow_prefetch_skipped_overlap": stats["prefetch_skipped_overlap"],
                "glow_prefetch_skipped_throttle": stats["prefetch_skipped_throttle"],
                "glow_prefetch_skipped_empty": stats["prefetch_skipped_empty"],
                "glow_prefetch_extras_avg": float(avg_extras),
                "glow_prefetch_extras_max": int(stats["prefetch_extras_max"]),
                "glow_prefetch_extras_total": int(stats["prefetch_extras_total"]),
                "glow_prefetch_capped_total": int(stats["prefetch_capped_total"]),
                "glow_prefetch_overlap_avg": float(avg_overlap),
                "glow_prefetch_overlap_min": stats["prefetch_overlap_min"],
                "glow_prefetch_overlap_max": stats["prefetch_overlap_max"],
            }
            try:
                self.current_trace["epoch"].update(glow_trace)
            except Exception:
                pass
            self.config.log(
                "Glow trace: "
                f"windows={windows_total} "
                f"multi={stats['windows_multi']} "
                f"avg_members={avg_members:.2f} "
                f"avg_entities={avg_entities:.0f} "
                f"max_entities={stats['window_entities_max']} "
                f"prefetch_calls={prefetch_calls} "
                f"skipped(disabled={stats['prefetch_skipped_disabled']},"
                f"repeat={stats['prefetch_skipped_repeat']},"
                f"overlap={stats['prefetch_skipped_overlap']},"
                f"throttle={stats['prefetch_skipped_throttle']},"
                f"empty={stats['prefetch_skipped_empty']}) "
                f"extras_avg={avg_extras:.0f} "
                f"overlap_avg={avg_overlap:.4f}"
            )

        self.current_trace["epoch"].update(
            dict(
                scope="epoch",
                event="epoch_completed",
                processed_examples=total_examples,
                processed_batches=total_batches,
                expected_batches=total_expected_batches,
                chunk_count=chunk_count,
                duplication_factor=duplication_factor,
                avg_loss=total_sum_loss / epoch_loss_divisor,
                avg_penalty=total_sum_penalty / epoch_penalty_divisor,
                avg_penalties={
                    k: p / epoch_penalty_divisor for k, p in total_sum_penalties.items()
                },
                avg_cost=total_sum_loss / epoch_loss_divisor
                + total_sum_penalty / epoch_penalty_divisor,
                epoch_time=epoch_time,
                chunk_time=total_chunk_time,
                prepare_time=total_prepare_time,
                ps_wait_time=total_ps_wait_time,
                unique_time=total_unique_time,
                pull_and_map_time=total_pull_and_map_time,
                pre_pull_time=total_pre_pull_time,
                entity_pull_time=total_entity_pull_time,
                relation_pull_time=total_relation_pull_time,
                ps_set_time=total_ps_set_time,
                cpu_gpu_time=total_cpu_gpu_time,
                forward_time=total_forward_time,
                backward_time=total_backward_time,
                optimizer_time=total_optimizer_time,
                scheduler_time=total_scheduler_time,
                glow_window_union_time=total_glow_window_union_time,
                glow_prefetch_time=total_glow_prefetch_time,
                glow_gradient_trace_time=total_glow_gradient_trace_time,
                glow_register_result_time=total_glow_register_result_time,
                glow_conflict_time=total_glow_conflict_time,
                other_time=epoch_time
                - total_prepare_time
                - total_forward_time
                - total_backward_time
                - total_optimizer_time
                - total_scheduler_time,
                embedding_mapping_time=total_embedding_mapping_time,
                batches=total_expected_batches,
                dataloader_time=total_dataloader_time,
                grad_sum=total_grad_sum,
                grad_samples=total_grad_samples,
                grad_avg=(
                    total_grad_sum / max(1, total_grad_samples)
                    if total_grad_samples > 0
                    else 0.0
                ),
                relation_grad_samples=total_relation_grad_samples,
            )
        )
        if pull_stats:
            self.current_trace["epoch"].update(
                dict(
                    ps_pull_calls=pull_stats.get("calls", 0),
                    ps_pull_keys=pull_stats.get("keys", 0),
                    ps_pull_bytes=pull_stats.get("bytes", 0),
                )
            )
        if push_stats:
            self.current_trace["epoch"].update(
                dict(
                    ps_push_calls=push_stats.get("calls", 0),
                    ps_push_keys=push_stats.get("keys", 0),
                    ps_push_bytes=push_stats.get("bytes", 0),
                )
            )
        trace_entry = self.trace(**self.current_trace["epoch"], echo=False, log=True)
        self.config.log(
            format_trace_entry("train_epoch", trace_entry, self.config), prefix="  "
        )
        if hasattr(self, "optimizer") and hasattr(
            self.optimizer, "flush_pending_pushes"
        ):
            self.optimizer.flush_pending_pushes()
        self.current_trace["epoch"] = None
        maybe_log_profile(force=True)
        return trace_entry

    def handle_validation(self, metric_name):
        # move all models to cpu and store as tmp model
        tmp_model = self.model.cpu()
        #self.valid_job.model = tmp_model
        del self.model
        if hasattr(self.valid_job, "model"):
            del self.valid_job.model
        gc.collect()
        if "cuda" in self.device:
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        self.parameter_client.barrier()
        num_eval_workers = self.config.get("job.distributed.num_eval_workers")
        if self.parameter_client.rank in range(self.min_rank, self.min_rank + num_eval_workers):
            # create a model for validation with entity embedder size
            #  batch_size x 2 + eval.chunk_size
            self.config.set(self.config.get("model") + ".create_eval", True)

            tmp_pretrain_model_filename = self.config.get("lookup_embedder.pretrain.model_filename")
            self.config.set("lookup_embedder.pretrain.model_filename", "")
            self.model = KgeModel.create(
                self.config, self.dataset, parameter_client=self.parameter_client
            )
            self.model.get_s_embedder().to_device(move_optim_data=False)
            self.model.get_p_embedder().to_device(move_optim_data=False)
            self.config.set("lookup_embedder.pretrain.model_filename", tmp_pretrain_model_filename)
            self.config.set(self.config.get("model") + ".create_eval", False)

            self.valid_job.model = self.model
            # validate and update learning rate
            super(TrainingJobNegativeSamplingDistributed, self).handle_validation(
                metric_name
            )

            # clean up valid model
            del self.model
            del self.valid_job.model
            gc.collect()
            if "cuda" in self.device:
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
        else:
            self.kge_lr_scheduler.step()
        self.parameter_client.barrier()
        self.model = tmp_model.to(self.device)
        del tmp_model
        gc.collect()

    def handle_running_checkpoint(self, checkpoint_every, checkpoint_keep):
        # since it is rather expensive to handle checkpoints in every epoch we only
        # do it every time we are evaluating now
        valid_every = self.config.get("valid.every")
        self.parameter_client.barrier()
        if self.parameter_client.rank == self.min_rank:
            self.save(self.config.checkpoint_file(self.epoch))
            delete_checkpoint_epoch = 0
            if checkpoint_every == 0:
                # do not keep any old checkpoints
                delete_checkpoint_epoch = self.epoch - valid_every
                # checkpoint every does not help a lot if we only store on valid
            elif checkpoint_keep > 0:
                # keep a maximum number of checkpoint_keep checkpoints
                delete_checkpoint_epoch = (
                        self.epoch - valid_every - valid_every * checkpoint_keep
                )
            if delete_checkpoint_epoch > 0:
                self._delete_checkpoint(
                    delete_checkpoint_epoch
                )
        self.parameter_client.barrier()

    def _delete_checkpoint(self, checkpoint_id):
        filename = self.config.checkpoint_file(checkpoint_id)
        super(TrainingJobNegativeSamplingDistributed, self)._delete_checkpoint(
            checkpoint_id
        )
        file, file_ending = filename.rsplit(".", 1)
        if os.path.exists(f"{file}_entities"):
            shutil.rmtree(f"{file}_entities")
        if os.path.exists(f"{file}_relations.{file_ending}"):
            os.remove(f"{file}_relations.{file_ending}")

    def save(self, filename) -> None:
        if self.parameter_client.rank == get_min_rank(self.config):
            # todo: we do not need to store the weights of the emebdders and optim here
            super(TrainingJobNegativeSamplingDistributed, self).save(filename)
            local_model_size = self.model.get_s_embedder().vocab_size
            global_model_size = self.model.get_s_embedder().complete_vocab_size
            entity_dim = self.model.get_s_embedder().dim
            optimizer_dim = self.model.get_s_embedder().optimizer_dim
            chunk_size = min(max(1000000, local_model_size), global_model_size)
            empty_pull_tensor = torch.empty(
                [chunk_size, entity_dim + optimizer_dim], device="cpu"
            )
            num_entities = self.dataset.num_entities()
            file, file_ending = filename.rsplit(".", 1)
            entities_dir = f"{file}_entities"
            if not os.path.exists(entities_dir):
                os.mkdir(entities_dir)
            for chunk_number in range(math.ceil(num_entities / chunk_size)):
                chunk_start = chunk_size * chunk_number
                chunk_end = min(chunk_size * (chunk_number + 1), num_entities)
                entity_ids = torch.arange(chunk_start, chunk_end, dtype=torch.long)
                lapse_offset = self.model.get_s_embedder().lapse_offset
                pull_tensor = empty_pull_tensor[: len(entity_ids)]
                self.parameter_client.pull(entity_ids + lapse_offset, pull_tensor)
                torch.save(
                    pull_tensor,
                    os.path.join(
                        entities_dir, f"{chunk_start}-{chunk_end}.{file_ending}"
                    ),
                )
            lapse_offset = self.model.get_p_embedder().lapse_offset
            pull_tensor = self.model.get_p_embedder().pull_tensors[0][1]
            relation_ids = torch.arange(self.dataset.num_relations(), dtype=torch.long)
            self.parameter_client.pull(relation_ids + lapse_offset, pull_tensor)
            torch.save(pull_tensor, f"{file}_relations.{file_ending}")
    def _handle_partition_conflict(
        self, status: int, replay_version: int, finished_context
    ):
        if status != 1:
            return
        if (
            self.config.get("job.distributed.conflict_free_merge")
            and self.config.get("job.distributed.parameter_server") == "shared"
        ):
            self.config.log(
                "Skipping replay for conflicting partition updates "
                "because conflict_free_merge is enabled."
            )
            return
        partition_id = None
        partition_version = None
        if finished_context is not None:
            partition_id, partition_version = finished_context
        if partition_id is None or partition_version is None:
            return
        if not hasattr(self.optimizer, "replay_partition_updates"):
            return
        try:
            replayed = self.optimizer.replay_partition_updates(
                partition_id, partition_version, replay_version
            )
        except Exception as exc:
            self.config.log(
                f"Failed to replay updates for partition {partition_id} "
                f"(version {partition_version}): {exc}"
            )
            return
        if replayed:
            self.config.log(
                f"Replayed updates for partition {partition_id} "
                f"(original version {partition_version}, "
                f"replay version {replay_version})."
            )

    def _handle_window_conflicts(self, conflicts):
        if not conflicts:
            return
        if (
            self.config.get("job.distributed.conflict_free_merge")
            and self.config.get("job.distributed.parameter_server") == "shared"
        ):
            self.config.log(
                "Skipping replay for conflicting window updates "
                "because conflict_free_merge is enabled."
            )
            return
        if not hasattr(self.optimizer, "replay_partition_updates"):
            return
        for partition_id, partition_version, replay_version in conflicts:
            try:
                replayed = self.optimizer.replay_partition_updates(
                    partition_id, partition_version, replay_version
                )
            except Exception as exc:
                self.config.log(
                    f"Failed to replay updates for window partition {partition_id} "
                    f"(version {partition_version}): {exc}"
                )
                continue
            if replayed:
                self.config.log(
                    f"Replayed updates for window partition {partition_id} "
                    f"(original version {partition_version}, "
                    f"replay version {replay_version})."
                )
