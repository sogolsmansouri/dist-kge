import math
import time
import json
import itertools
import multiprocessing as mp
from enum import IntEnum
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import numba
import torch
import torch.distributed as dist
from collections import deque, defaultdict
from copy import deepcopy
from kge.distributed.misc import get_min_rank, set_master_environment, initialize_worker_groups
from kge.distributed.stratification_schedule_creator import StratificationScheduleCreator
from kge.misc import set_seeds


TORCH_TO_NP_DTYPE = {
    torch.long: np.int64,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int: np.int32,
}


class SCHEDULER_CMDS(IntEnum):
    GET_WORK = 0
    WORK_DONE = 1
    WORK = 2
    NO_WORK = 3
    WAIT = 4
    BARRIER = 5
    SHUTDOWN = 6
    INIT_INFO = 7
    GET_INIT_WORK = 8
    GET_LOCAL_ENT = 9
    PRE_LOCALIZE_WORK = 10
    REGISTER_EVAL_RESULT = 11
    GET_EVAL_RESULT = 12
    REGISTER_PARTITION_RESULT = 13
    REGISTER_PARTITION_GRADIENT = 14
    REGISTER_PARTITION_RELATION_GRADIENT = 15
    REGISTER_WINDOW_RESULT = 16

@dataclass
class WorkPackage:

    partition_id = None
    partition_version = None
    partition_data = None
    entities_in_partition = None
    relations_in_partition = None
    window_members = None
    window_entities = None
    window_versions = None
    wait = False
    reuse_partition_version = False


class WorkScheduler(mp.get_context("fork").Process):
    def __init__(
        self,
        config,
        dataset,
    ):
        self._config_check(config)
        super(WorkScheduler, self).__init__(daemon=False, name="work-scheduler")
        self.config = config
        self.dataset = dataset
        self.min_rank = get_min_rank(config)
        self.rank = self.min_rank - 1
        self.num_clients = config.get("job.distributed.num_workers")
        self.world_size = self.num_clients + self.min_rank
        self.num_partitions = config.get("job.distributed.num_partitions")
        self.done_workers = []
        self.asking_workers = []
        self.work_to_do = deque(list(range(self.num_partitions)))
        self.wait_time = 0.4
        self.repartition_epoch = config.get("job.distributed.repartition_epoch")
        self.init_up_to_entity = -1
        self.num_processed_partitions = 0
        self.eval_hists = []
        self.active_partition_per_worker = dict()
        self.active_partition_chunk_sizes = dict()
        self._active_partition_start_time = {}
        self._worker_last_seen = {}
        self._disabled_workers = set()
        self._stall_reported = set()
        self._worker_stall_timeout = float(
            config.get("job.distributed.worker_stall_timeout") or 0
        )
        self._worker_stall_action = str(
            config.get("job.distributed.worker_stall_action") or "fail"
        ).lower()
        self.partition_stats = dict()
        self.partition_gradient_stats = defaultdict(lambda: {"sum": 0.0, "count": 0})
        self.partition_relation_gradient_stats = defaultdict(
            lambda: defaultdict(lambda: {"sum": 0.0, "count": 0})
        )
        self.gradient_snapshot_interval = max(
            0, int(config.get("job.distributed.gradient_snapshot_interval") or 0)
        )
        if (
            self.gradient_snapshot_interval
            and self.gradient_snapshot_interval > self.num_partitions
        ):
            config.log(
                "Clamping gradient_snapshot_interval "
                f"{self.gradient_snapshot_interval} to num_partitions "
                f"{self.num_partitions} to ensure updates per epoch."
            )
            self.gradient_snapshot_interval = self.num_partitions
        graph_cfg = config.get("job.distributed.gradient_graph") or {}
        self._gradient_graph_enabled = bool(graph_cfg.get("enable", False))
        self._gradient_graph_top_relations = max(
            0, int(graph_cfg.get("top_relations", 0))
        )
        self._gradient_graph_export_interval = max(
            0, int(graph_cfg.get("export_interval", 0))
        )
        if (
            self._gradient_graph_enabled
            and self._gradient_graph_export_interval
            and self._gradient_graph_export_interval > self.num_partitions
        ):
            config.log(
                "Clamping gradient_graph.export_interval "
                f"{self._gradient_graph_export_interval} to num_partitions "
                f"{self.num_partitions} to ensure updates per epoch."
            )
            self._gradient_graph_export_interval = self.num_partitions
        self._gradient_graph_last_export = 0
        self._gradient_graph_partitions = (
            defaultdict(dict) if self._gradient_graph_enabled else None
        )
        self._gradient_graph_relations = (
            defaultdict(dict) if self._gradient_graph_enabled else None
        )
        cluster_cfg = graph_cfg.get("clustering") or {}
        self._gradient_graph_cluster_enabled = bool(cluster_cfg.get("enable", False))
        self._gradient_graph_cluster_update_interval = max(
            0, int(cluster_cfg.get("update_interval", 0))
        )
        if (
            self._gradient_graph_cluster_enabled
            and self._gradient_graph_cluster_update_interval
            and self._gradient_graph_cluster_update_interval > self.num_partitions
        ):
            config.log(
                "Clamping gradient_graph.clustering.update_interval "
                f"{self._gradient_graph_cluster_update_interval} to "
                f"num_partitions {self.num_partitions} to ensure updates per epoch."
            )
            self._gradient_graph_cluster_update_interval = self.num_partitions
        if (
            self._gradient_graph_cluster_enabled
            and self._gradient_graph_cluster_update_interval == 0
            and self._gradient_graph_export_interval == 0
            and self.gradient_snapshot_interval == 0
        ):
            config.log(
                "Gradient graph clustering enabled but no update interval is set; "
                "set gradient_graph.clustering.update_interval, "
                "gradient_graph.export_interval, or gradient_snapshot_interval."
            )
        self._gradient_graph_cluster_min_affinity = float(
            cluster_cfg.get("min_affinity", 0.1)
        )
        self._gradient_graph_cluster_min_affinity = min(
            1.0, max(0.0, self._gradient_graph_cluster_min_affinity)
        )
        self._gradient_graph_cluster_min_shared = max(
            1, int(cluster_cfg.get("min_shared_relations", 1))
        )
        self._gradient_graph_cluster_max_size = max(
            0, int(cluster_cfg.get("max_cluster_size", 0))
        )
        self._gradient_graph_cluster_top_relations = max(
            0, int(cluster_cfg.get("top_relations", 0))
        )
        self._gradient_graph_cluster_max_partitions_per_relation = max(
            0, int(cluster_cfg.get("max_partitions_per_relation", 0))
        )
        self._gradient_graph_cluster_last_update = 0
        self._gradient_graph_clusters = {}
        self._gradient_graph_cluster_members = {}
        self._gradient_updates = 0
        self._last_gradient_snapshot = 0
        self.partition_issue_versions = defaultdict(int)
        self.partition_committed_versions = defaultdict(lambda: -1)
        self.active_partition_versions = dict()
        if config.get("job.distributed.scheduler_data_type") not in ["int", "int32", "int64", "long"]:
            raise ValueError("Only long and int is supported as dtype for the scheduler communication")
        self.data_type = getattr(torch, config.get("job.distributed.scheduler_data_type"))
        if self.repartition_epoch:
            self.repartition_future = None
            self.repartition_worker_pool = None
        # map complex partition identifiers (e.g., tuples) to scalar aliases for IPC
        self._partition_alias_lookup = {}
        self._alias_to_partition = {}
        self._next_partition_alias = -2

    def _init_in_started_process(self):
        self.partitions = self._load_partitions(self.num_partitions)
        self._define_local_entities()

    def _normalize_partition_key(self, partition_id):
        if isinstance(partition_id, (list, tuple)):
            return tuple(int(x) for x in partition_id)
        return partition_id

    def _encode_partition_id(self, partition_id):
        if partition_id is None:
            return -1
        key = self._normalize_partition_key(partition_id)
        if isinstance(key, int):
            return key
        alias = self._partition_alias_lookup.get(key)
        if alias is None:
            alias = self._next_partition_alias
            self._next_partition_alias -= 1
            self._partition_alias_lookup[key] = alias
            self._alias_to_partition[alias] = key
        return alias

    def _decode_partition_id(self, partition_id):
        if partition_id is None:
            return None
        if partition_id >= 0 or partition_id == -1:
            return partition_id
        return self._alias_to_partition.get(partition_id, partition_id)

    def _serialize_partition_key(self, partition_id):
        if isinstance(partition_id, (list, tuple)):
            return [int(x) for x in partition_id]
        if isinstance(partition_id, np.integer):
            return int(partition_id)
        return partition_id

    def _define_local_entities(self):
        """
        Currently, this method is only used for the initialization of the parameter
        server and the local sampler.
        It assigns a random set of entities to each client.
        Note that partitioning types assigning entity sets for partitions define those
        entities together with the WorkPackage.
        """
        if self._set_local_entities_from_partitions():
            return
        entity_keys = torch.arange(self.dataset.num_entities(), dtype=self.data_type)
        local_entities = entity_keys[torch.randperm(len(entity_keys))].chunk(self.num_clients)
        self.local_entities = dict(zip(range(self.min_rank, self.min_rank + self.num_clients), local_entities))

    def _load_entity_partition_assignments(self):
        if self.num_partitions <= 0:
            return None
        partition_type = getattr(self.dataset, "_partition_type", None)
        if partition_type is None:
            return None
        try:
            assignments = self.dataset.load_entities_to_partitions(self.num_partitions)
        except (FileNotFoundError, OSError):
            return None
        except Exception as e:
            self.config.log(
                f"Failed to load entity partition assignments for local pools: {e}"
            )
            return None
        assignments = torch.as_tensor(assignments, dtype=torch.long).view(-1)
        if assignments.numel() == 0:
            return None
        return assignments

    def _set_local_entities_from_partitions(self):
        assignments = self._load_entity_partition_assignments()
        if assignments is None:
            return False
        partition_entities = dict()
        for partition_id in range(self.num_partitions):
            indexes = torch.nonzero(assignments == partition_id, as_tuple=False).view(-1)
            if indexes.numel() == 0:
                continue
            partition_entities[partition_id] = indexes.to(dtype=self.data_type)
        if not partition_entities:
            return False
        worker_assignments = {
            rank: []
            for rank in range(self.min_rank, self.min_rank + self.num_clients)
        }
        ordered_partitions = sorted(partition_entities.keys())
        for idx, partition_id in enumerate(ordered_partitions):
            rank = self.min_rank + (idx % self.num_clients)
            worker_assignments[rank].append(partition_entities[partition_id])
        if ordered_partitions:
            fill_index = 0
            for rank, segments in worker_assignments.items():
                if segments:
                    continue
                partition_id = ordered_partitions[fill_index % len(ordered_partitions)]
                fill_index += 1
                worker_assignments[rank].append(partition_entities[partition_id])
        self.local_entities = dict()
        for rank, entity_lists in worker_assignments.items():
            merged = torch.unique(torch.cat(entity_lists)).to(dtype=self.data_type)
            self.local_entities[rank] = merged
        return True

    def _config_check(self, config):
        if (
            config.get("job.distributed.entity_sync_level") == "partition"
            and not config.get("negative_sampling.sampling_type") == "pooled"
        ):
            raise ValueError(
                "entity sync level 'partition' only supported with 'pooled' sampling."
            )

    @staticmethod
    def create(
        config,
        dataset,
    ):
        if config.get("job.type") != "train":
            partition_type = "random"
        else:
            partition_type = config.get("job.distributed.partition_type")

        if partition_type == "random":
            return RandomWorkScheduler(config=config, dataset=dataset)
        elif partition_type == "relation":
            return RelationWorkScheduler(config=config, dataset=dataset)
        elif partition_type == "graph-cut":
            return GraphCutWorkScheduler(config=config, dataset=dataset)
        elif partition_type == "stratification":
            return StratificationWorkScheduler(config=config, dataset=dataset)
        elif partition_type == "glow":
            return GlowWorkScheduler(config=config, dataset=dataset)
        else:
            raise NotImplementedError()

    def run(self):
        set_seeds(config=self.config)
        self._init_in_started_process()
        set_master_environment(self.config)
        # we have to have a huge timeout here, since it is only called after a complete
        #  epoch on a partition
        print("start scheduler with rank", self.rank, "world_size", self.world_size)
        # we need to create the worker group here as well it need to be defined in
        #  all processes
        initialize_worker_groups(self.config, self.rank)
        barrier_count = 0
        shutdown_count = 0
        epoch_time = None
        if self.repartition_epoch:
            if self.repartition_worker_pool is None:
                self.repartition_worker_pool = mp.Pool(processes=1)
            self._repartition_in_background()

        while True:
            # cmd_buffer consists of cmd_number, key_len
            cmd_buffer = torch.full((2,), -1, dtype=self.data_type)

            # refill work and distribute to all asking workers
            if self._effective_done_workers() == self.num_clients:
                epoch_time += time.time()
                self.config.log(f"complete_epoch_time: {epoch_time}")
                epoch_time = None
                self.num_processed_partitions = 0
                self._refill_work()
                self._on_epoch_completed()
                for worker, machine_id in self.asking_workers:
                    work_package = self._next_work(worker, machine_id)
                    self._send_work(worker, cmd_buffer, work_package)
                self.done_workers = []
                self.asking_workers = []
                continue

            self._check_worker_stalls(time.time())
            rank = dist.recv(cmd_buffer)
            self._worker_last_seen[rank] = time.time()
            cmd = cmd_buffer[0].item()
            if cmd == SCHEDULER_CMDS.GET_WORK:
                if epoch_time is None:
                    epoch_time = -time.time()
                machine_id = cmd_buffer[1].item()
                if rank in self._disabled_workers:
                    work_package = WorkPackage()
                    self._send_work(rank, cmd_buffer, work_package)
                    continue
                if rank in self.done_workers:
                    self.asking_workers.append((rank, machine_id))
                    continue
                work_package = self._next_work(rank, machine_id)
                self._send_work(rank, cmd_buffer, work_package)
            elif cmd == SCHEDULER_CMDS.WORK_DONE:
                self._handle_work_done(rank)
            elif cmd == SCHEDULER_CMDS.BARRIER:
                barrier_count += 1
                if barrier_count == self.num_clients:
                    barrier_count = 0
                    dist.barrier()
            elif cmd == SCHEDULER_CMDS.SHUTDOWN:
                shutdown_count += 1
                if shutdown_count == self.num_clients:
                    print("shutting down work scheduler")
                    if self.repartition_epoch:
                        if self.repartition_worker_pool is not None:
                            self.repartition_worker_pool.close()
                            self.repartition_worker_pool.terminate()
                    self._on_scheduler_shutdown()
                    break
            elif cmd == SCHEDULER_CMDS.INIT_INFO:
                self._handle_init_info(rank)
            elif cmd == SCHEDULER_CMDS.GET_INIT_WORK:
                self._handle_get_init_work(
                    rank=rank, embedding_layer_size=cmd_buffer[1].item()
                )
            elif cmd == SCHEDULER_CMDS.GET_LOCAL_ENT:
                self._handle_get_local_entities(rank=rank)
            elif cmd == SCHEDULER_CMDS.PRE_LOCALIZE_WORK:
                machine_id = cmd_buffer[1].item()
                work_package = self._handle_pre_localize_work(
                    rank=rank, machine_id=machine_id
                )
                self._send_work(
                    rank, cmd_buffer, work_package, pre_localize=True
                )
            elif cmd == SCHEDULER_CMDS.REGISTER_EVAL_RESULT:
                self._handle_register_eval_result(rank, cmd_buffer)
            elif cmd == SCHEDULER_CMDS.GET_EVAL_RESULT:
                self._handle_get_eval_result(rank)
            elif cmd == SCHEDULER_CMDS.REGISTER_PARTITION_RESULT:
                self._handle_register_partition_result(rank)
            elif cmd == SCHEDULER_CMDS.REGISTER_WINDOW_RESULT:
                self._handle_register_window_result(rank)
            elif cmd == SCHEDULER_CMDS.REGISTER_PARTITION_GRADIENT:
                self._handle_register_partition_gradient(rank)
            elif cmd == SCHEDULER_CMDS.REGISTER_PARTITION_RELATION_GRADIENT:
                self._handle_register_partition_relation_gradient(rank)
            else:
                raise ValueError(
                    f"The work scheduler received an unknown command: {cmd}"
                )

    def _next_work(
        self, rank, machine_id
    ) -> WorkPackage:
        raise NotImplementedError()

    def _refill_work(self):
        self.work_to_do = deque(list(range(self.num_partitions)))

    def _repartition_in_background(self):
        pass

    def _effective_done_workers(self):
        if not self._disabled_workers:
            return len(self.done_workers)
        return len(set(self.done_workers) | self._disabled_workers)

    def _record_partition_start(self, rank, partition_id):
        self._active_partition_start_time[rank] = time.time()
        self._stall_reported.discard((rank, partition_id))

    def _clear_partition_state(self, rank):
        self._active_partition_start_time.pop(rank, None)
        self._stall_reported = {
            key for key in self._stall_reported if key[0] != rank
        }

    def _drop_stalled_worker(self, rank, partition_id, elapsed):
        self.config.log(
            "Scheduler watchdog dropping stalled worker "
            f"{rank} partition={partition_id} elapsed={elapsed:.1f}s."
        )
        self._disabled_workers.add(rank)
        self.active_partition_per_worker.pop(rank, None)
        self.active_partition_chunk_sizes.pop(rank, None)
        self.active_partition_versions.pop(rank, None)
        self._clear_partition_state(rank)

    def _check_worker_stalls(self, now):
        if self._worker_stall_timeout <= 0:
            return
        if not self.active_partition_per_worker:
            return
        for rank, start_time in list(self._active_partition_start_time.items()):
            if rank in self._disabled_workers:
                continue
            elapsed = now - start_time
            if elapsed < self._worker_stall_timeout:
                continue
            partition_id = self.active_partition_per_worker.get(rank)
            report_key = (rank, partition_id)
            if report_key in self._stall_reported:
                continue
            self._stall_reported.add(report_key)
            action = self._worker_stall_action
            if action == "drop":
                self._drop_stalled_worker(rank, partition_id, elapsed)
                continue
            if action != "fail":
                self.config.log(
                    "Scheduler watchdog invalid action "
                    f"'{action}', falling back to 'fail'."
                )
            raise RuntimeError(
                "Scheduler watchdog detected stalled worker "
                f"{rank} partition={partition_id} elapsed={elapsed:.1f}s"
            )

    def _send_work(
        self, rank, cmd_buffer, work_package, pre_localize=False
    ):
        if (
            work_package.partition_data is None
            and getattr(self, "glow_window_work", False)
            and work_package.window_members is not None
        ):
            window_members = [int(pid) for pid in work_package.window_members]
            partition_slices = []
            for pid in window_members:
                partition_tensor = None
                if 0 <= pid < len(self.partitions):
                    partition_tensor = self.partitions[pid]
                if partition_tensor is None or len(partition_tensor) == 0:
                    continue
                partition_slices.append(partition_tensor)
            if partition_slices:
                work_package.partition_data = (
                    torch.cat(partition_slices).contiguous()
                )
                if work_package.partition_id is None:
                    work_package.partition_id = tuple(window_members)
                if work_package.window_versions is None:
                    window_versions = []
                    for pid in window_members:
                        version = self.partition_issue_versions[pid]
                        self.partition_issue_versions[pid] = version + 1
                        window_versions.append(int(version))
                    work_package.window_versions = window_versions
                if work_package.entities_in_partition is None:
                    window_entities = None
                    if hasattr(self, "_get_window_entities"):
                        window_entities = self._get_window_entities(
                            tuple(window_members)
                        )
                    if window_entities is None:
                        window_entities = self.local_entities.get(rank)
                    work_package.entities_in_partition = window_entities
                if getattr(self, "_glow_debug", False):
                    self._glow_log(
                        "Rebuilt window work in _send_work for "
                        f"{tuple(window_members)} "
                        f"size={int(work_package.partition_data.numel())}."
                    )
            elif getattr(self, "_glow_debug", False):
                self._glow_log(
                    "Failed to rebuild window work in _send_work for "
                    f"{tuple(window_members)}; no partition data."
                )
        debug_glow = getattr(self, "_glow_debug", False)
        if debug_glow:
            local_rank = rank - self.min_rank
            rank_label = f"{rank}"
            if 0 <= local_rank < self.num_clients:
                rank_label = f"{rank} (local={local_rank})"
            context = "pre_localize" if pre_localize else "work"
            if work_package.wait:
                self._glow_log(
                    "Sending WAIT to rank "
                    f"{rank_label} ({context}) "
                    f"active={len(self.active_partition_per_worker)} "
                    f"disabled={len(self._disabled_workers)}."
                )
            elif work_package.partition_data is None:
                self._glow_log(
                    "Sending NO_WORK to rank "
                    f"{rank_label} ({context}) "
                    f"active={len(self.active_partition_per_worker)} "
                    f"disabled={len(self._disabled_workers)}."
                )
            else:
                window_members = work_package.window_members
                window_members = (
                    tuple(window_members) if window_members is not None else None
                )
                self._glow_log(
                    "Sending WORK to rank "
                    f"{rank_label} ({context}) "
                    f"partition_id={work_package.partition_id} "
                    f"size={len(work_package.partition_data)} "
                    f"window_members={window_members} "
                    f"active={len(self.active_partition_per_worker)} "
                    f"disabled={len(self._disabled_workers)}."
                )
        if (
            not pre_localize
            and work_package.partition_data is None
            and not work_package.wait
            and self.active_partition_per_worker
        ):
            work_package.wait = True
            if debug_glow:
                self._glow_log(
                    "Active partitions remain; converting NO_WORK to WAIT "
                    f"for rank {rank}."
                )
        partition_payload = work_package.partition_data
        if pre_localize and partition_payload is not None:
            partition_payload = torch.empty((0,), dtype=self.data_type)
            work_package.partition_version = None
        if partition_payload is not None:
            if not pre_localize and work_package.partition_id is not None:
                if work_package.reuse_partition_version:
                    if work_package.partition_version is None:
                        work_package.partition_version = self.partition_issue_versions[
                            work_package.partition_id
                        ]
                else:
                    current_version = self.partition_issue_versions[
                        work_package.partition_id
                    ]
                    work_package.partition_version = current_version
                    self.partition_issue_versions[work_package.partition_id] = (
                        current_version + 1
                    )
            cmd_buffer[0] = SCHEDULER_CMDS.WORK
            cmd_buffer[1] = len(partition_payload)
            dist.send(cmd_buffer, dst=rank)
            dist.send(partition_payload, dst=rank)
            partition_alias = self._encode_partition_id(work_package.partition_id)
            partition_info = torch.tensor(
                [
                    -1
                    if partition_alias is None
                    else int(partition_alias)
                ],
                dtype=self.data_type,
            )
            dist.send(partition_info, dst=rank)
            version_tensor = torch.tensor(
                [
                    -1
                    if work_package.partition_version is None
                    else int(work_package.partition_version)
                ],
                dtype=self.data_type,
            )
            dist.send(version_tensor, dst=rank)
            if not pre_localize:
                self.active_partition_per_worker[rank] = work_package.partition_id
                self._record_partition_start(rank, work_package.partition_id)
                if (
                    work_package.window_members is not None
                    and work_package.window_versions is not None
                ):
                    members = [
                        int(x) for x in work_package.window_members
                    ]
                    versions = [
                        int(x) for x in work_package.window_versions
                    ]
                    self.active_partition_versions[rank] = dict(
                        zip(members, versions)
                    )
                else:
                    self.active_partition_versions[rank] = (
                        work_package.partition_version
                    )
                self.active_partition_chunk_sizes[rank] = len(work_package.partition_data)
                if hasattr(self, "previous_partition_per_worker"):
                    self.previous_partition_per_worker[rank] = work_package.partition_id
            if work_package.entities_in_partition is None:
                cmd_buffer[1] = 0
                dist.send(cmd_buffer, dst=rank)
            else:
                cmd_buffer[1] = len(work_package.entities_in_partition)
                dist.send(cmd_buffer, dst=rank)
                dist.send(work_package.entities_in_partition, dst=rank)
            if work_package.relations_in_partition is None:
                cmd_buffer[1] = 0
                dist.send(cmd_buffer, dst=rank)
            else:
                cmd_buffer[1] = len(work_package.relations_in_partition)
                dist.send(cmd_buffer, dst=rank)
                dist.send(work_package.relations_in_partition, dst=rank)
            if not work_package.window_members:
                cmd_buffer[1] = 0
                dist.send(cmd_buffer, dst=rank)
            else:
                window_tensor = torch.as_tensor(
                    work_package.window_members, dtype=self.data_type
                )
                cmd_buffer[1] = len(window_tensor)
                dist.send(cmd_buffer, dst=rank)
                dist.send(window_tensor, dst=rank)
            window_entities_data = work_package.window_entities
            if (
                window_entities_data is None
                or not isinstance(window_entities_data, torch.Tensor)
                or window_entities_data.numel() == 0
            ):
                cmd_buffer[1] = 0
                dist.send(cmd_buffer, dst=rank)
            else:
                window_entities = torch.as_tensor(
                    window_entities_data, dtype=self.data_type
                )
                cmd_buffer[1] = len(window_entities)
                dist.send(cmd_buffer, dst=rank)
                dist.send(window_entities, dst=rank)
            window_versions_data = work_package.window_versions
            if (
                window_versions_data is None
                or (isinstance(window_versions_data, torch.Tensor)
                    and window_versions_data.numel() == 0)
                or (not isinstance(window_versions_data, torch.Tensor)
                    and len(window_versions_data) == 0)
            ):
                cmd_buffer[1] = 0
                dist.send(cmd_buffer, dst=rank)
            else:
                window_versions = torch.as_tensor(
                    window_versions_data, dtype=self.data_type
                )
                cmd_buffer[1] = len(window_versions)
                dist.send(cmd_buffer, dst=rank)
                dist.send(window_versions, dst=rank)
        elif work_package.wait:
            cmd_buffer[0] = SCHEDULER_CMDS.WAIT
            cmd_buffer[1] = self.wait_time
            dist.send(cmd_buffer, dst=rank)
        else:
            if not pre_localize:
                self.done_workers.append(rank)
            cmd_buffer[0] = SCHEDULER_CMDS.NO_WORK
            cmd_buffer[1] = 0
            dist.send(cmd_buffer, dst=rank)

    def _handle_work_done(self, rank):
        self.num_processed_partitions += 1
        print(f"trainer {rank} done with partition {self.num_processed_partitions}")
        self.active_partition_per_worker.pop(rank, None)
        self.active_partition_chunk_sizes.pop(rank, None)
        self.active_partition_versions.pop(rank, None)
        self._clear_partition_state(rank)

    def _handle_init_info(self, rank):
        max_entities = self._get_max_entities()
        max_relations = self._get_max_relations()
        init_data = torch.tensor([max_entities, max_relations], dtype=self.data_type)
        dist.send(init_data, dst=rank)

    def _handle_get_init_work(self, rank, embedding_layer_size):
        if self.init_up_to_entity == -1:
            print("initialize parameter server")
        self.init_up_to_entity += 1
        if self.init_up_to_entity >= self.dataset.num_entities():
            return_buffer = torch.tensor([-1, -1], dtype=self.data_type)
        else:
            entity_range_end = min(
                self.dataset.num_entities(),
                self.init_up_to_entity + embedding_layer_size,
            )
            if entity_range_end == self.dataset.num_entities():
                print("parameter server initialized")
            return_buffer = torch.tensor([self.init_up_to_entity, entity_range_end], dtype=self.data_type)
        self.init_up_to_entity += embedding_layer_size
        dist.send(return_buffer, dst=rank)

    def _handle_get_local_entities(self, rank):
        size_information = torch.tensor([len(self.local_entities[rank]), -1], dtype=self.data_type)
        dist.send(size_information, dst=rank)
        dist.send(self.local_entities[rank], dst=rank)

    def _handle_pre_localize_work(self, rank, machine_id):
        raise ValueError("The current partition scheme does not support pre-localizing")

    def _on_epoch_completed(self):
        pass

    def _on_scheduler_shutdown(self):
        self._export_gradient_statistics(final=True)

    def _handle_register_eval_result(self, rank, cmd_buffer):
        num_sub_hists = cmd_buffer[1]
        first_eval = False
        if len(self.eval_hists) == 0:
            first_eval = True
        for j in range(num_sub_hists):
            ranks = torch.empty(self.dataset.num_entities())
            dist.recv(ranks, src=rank)
            if first_eval:
                self.eval_hists.append(ranks)
            else:
                self.eval_hists[j] += ranks

    def _handle_get_eval_result(self, rank):
        for i, h in enumerate(self.eval_hists):
            dist.send(h, dst=rank)
        self.eval_hists = []

    def _handle_register_partition_result(self, rank):
        result_buffer = torch.zeros(1, dtype=torch.float32)
        dist.recv(result_buffer, src=rank)
        version_buffer = torch.zeros(1, dtype=self.data_type)
        dist.recv(version_buffer, src=rank)
        chunk_buffer = torch.zeros(1, dtype=self.data_type)
        dist.recv(chunk_buffer, src=rank)
        reported_version = int(version_buffer[0].item())
        reported_chunk = int(chunk_buffer[0].item())
        status, aux = self._register_partition_result(
            rank,
            float(result_buffer[0].item()),
            reported_version,
            reported_chunk,
        )
        ack = torch.tensor([status, aux], dtype=self.data_type)
        dist.send(ack, dst=rank)

    def _handle_register_window_result(self, rank):
        result_buffer = torch.zeros(1, dtype=torch.float32)
        dist.recv(result_buffer, src=rank)
        info_buffer = torch.zeros(1, dtype=self.data_type)
        dist.recv(info_buffer, src=rank)
        window_count = int(info_buffer[0].item())
        if window_count <= 0:
            ack = torch.tensor([0], dtype=self.data_type)
            dist.send(ack, dst=rank)
            return
        members = torch.empty(window_count, dtype=self.data_type)
        dist.recv(members, src=rank)
        versions = torch.empty(window_count, dtype=self.data_type)
        dist.recv(versions, src=rank)
        chunk_buffer = torch.zeros(1, dtype=self.data_type)
        dist.recv(chunk_buffer, src=rank)
        reported_chunk = int(chunk_buffer[0].item())
        conflicts = self._register_window_result(
            rank,
            float(result_buffer[0].item()),
            members,
            versions,
            reported_chunk,
        )
        conflict_count = len(conflicts)
        ack = torch.tensor([conflict_count], dtype=self.data_type)
        dist.send(ack, dst=rank)
        if conflict_count:
            ids = torch.tensor([c[0] for c in conflicts], dtype=self.data_type)
            orig = torch.tensor([c[1] for c in conflicts], dtype=self.data_type)
            replay = torch.tensor([c[2] for c in conflicts], dtype=self.data_type)
            dist.send(ids, dst=rank)
            dist.send(orig, dst=rank)
            dist.send(replay, dst=rank)

    def _handle_register_partition_gradient(self, rank):
        info_buffer = torch.zeros(2, dtype=self.data_type)
        dist.recv(info_buffer, src=rank)
        partition_id = int(info_buffer[0].item())
        partition_id = self._decode_partition_id(partition_id)
        sample_count = int(info_buffer[1].item())
        grad_buffer = torch.zeros(1, dtype=torch.float32)
        dist.recv(grad_buffer, src=rank)
        grad_sum = float(grad_buffer[0].item())
        self._register_partition_gradient(partition_id, grad_sum, sample_count)

    def _handle_register_partition_relation_gradient(self, rank):
        info_buffer = torch.zeros(2, dtype=self.data_type)
        dist.recv(info_buffer, src=rank)
        partition_id = int(info_buffer[0].item())
        partition_id = self._decode_partition_id(partition_id)
        num_relations = int(info_buffer[1].item())
        if num_relations <= 0:
            return
        rel_ids = torch.empty(num_relations, dtype=self.data_type)
        dist.recv(rel_ids, src=rank)
        rel_counts = torch.empty(num_relations, dtype=self.data_type)
        dist.recv(rel_counts, src=rank)
        rel_sums = torch.empty(num_relations, dtype=torch.float32)
        dist.recv(rel_sums, src=rank)
        self._register_partition_relation_gradient(
            partition_id, rel_ids, rel_sums, rel_counts
        )

    def _register_partition_gradient(self, partition_id, grad_sum, sample_count):
        if partition_id is None:
            return
        # Ignore sentinel NO_WORK (-1). Keep negative aliases (< -1) and complex ids (e.g., tuples).
        if isinstance(partition_id, (int, np.integer)):
            pid_int = int(partition_id)
            if pid_int == -1:
                return
            if pid_int < -1:
                partition_id = self._decode_partition_id(pid_int)
        stats = self.partition_gradient_stats[partition_id]
        stats["sum"] += grad_sum
        stats["count"] += max(0, sample_count)
        self._gradient_updates += 1
        if (
            self.gradient_snapshot_interval
            and self._gradient_updates
            >= self._last_gradient_snapshot + self.gradient_snapshot_interval
        ):
            self._export_gradient_statistics()
            self._last_gradient_snapshot = self._gradient_updates
        self._maybe_export_gradient_graph()

    def _register_partition_relation_gradient(
        self, partition_id, rel_ids, rel_sums, rel_counts
    ):
        if partition_id is None:
            return
        # Ignore sentinel NO_WORK (-1). Keep negative aliases (< -1) and complex ids (e.g., tuples).
        if isinstance(partition_id, (int, np.integer)):
            pid_int = int(partition_id)
            if pid_int == -1:
                return
            if pid_int < -1:
                partition_id = self._decode_partition_id(pid_int)
        if rel_ids is None:
            return
        rel_ids_list = rel_ids.tolist() if torch.is_tensor(rel_ids) else list(rel_ids)
        rel_sums_list = rel_sums.tolist() if torch.is_tensor(rel_sums) else list(rel_sums)
        rel_counts_list = rel_counts.tolist() if torch.is_tensor(rel_counts) else list(rel_counts)
        for rel_id, rel_sum, rel_count in zip(rel_ids_list, rel_sums_list, rel_counts_list):
            rel_id = int(rel_id)
            if rel_id < 0:
                continue
            stats = self.relation_gradient_stats[rel_id]
            stats["sum"] += float(rel_sum)
            stats["count"] += int(rel_count)
        self._update_gradient_graph(
            partition_id, rel_ids_list, rel_sums_list, rel_counts_list
        )

    def _export_gradient_statistics(self, final: bool = False):
        if not self.partition_gradient_stats:
            return
        output_folder = Path(self.config.folder) / "gradient_snapshots"
        output_folder.mkdir(parents=True, exist_ok=True)
        if final:
            snapshot_path = output_folder / "gradient_snapshot_final.json"
        else:
            snapshot_path = output_folder / f"gradient_snapshot_{self._gradient_updates}.json"
        summary = {}
        for pid, stats in self.partition_gradient_stats.items():
            count = max(1, stats["count"])
            summary[pid] = {
                "sum": stats["sum"],
                "count": stats["count"],
                "avg": stats["sum"] / count,
            }
        try:
            with open(snapshot_path, "w") as fp:
                json.dump(
                    {
                        "num_partitions": self.num_partitions,
                        "snapshot_index": self._gradient_updates,
                        "final": final,
                        "partitions": summary,
                    },
                    fp,
                    indent=2,
                )
            self.config.log(
                f"Exported gradient snapshot with {len(summary)} partitions to {snapshot_path}."
            )
        except Exception as exc:
            self.config.log(f"Failed to export gradient snapshot: {exc}")
        self._export_relation_gradient_statistics(output_folder, final=final)
        self._export_gradient_graph(final=final)

    def _maybe_export_gradient_graph(self):
        if not self._gradient_graph_enabled:
            return
        interval = self._gradient_graph_export_interval
        if interval <= 0:
            interval = self.gradient_snapshot_interval
        if interval <= 0:
            return
        if self._gradient_updates < self._gradient_graph_last_export + interval:
            return
        self._export_gradient_graph()
        self._gradient_graph_last_export = self._gradient_updates

    def _maybe_update_gradient_graph_clusters(self, force: bool = False):
        if (
            not self._gradient_graph_cluster_enabled
            or not self._gradient_graph_enabled
            or self._gradient_graph_relations is None
        ):
            return
        interval = self._gradient_graph_cluster_update_interval
        if interval <= 0:
            interval = self._gradient_graph_export_interval
        if interval <= 0:
            interval = self.gradient_snapshot_interval
        if interval <= 0 and not force:
            return
        if (
            not force
            and self._gradient_updates
            < self._gradient_graph_cluster_last_update + interval
        ):
            return
        self.config.log(
            "Glow gradient graph clustering update at snapshot "
            f"{self._gradient_updates} "
            f"(relations={len(self._gradient_graph_relations)}, "
            f"partitions={len(self._gradient_graph_partitions or {})}, "
            f"force={force})."
        )
        self._recompute_gradient_graph_clusters()
        self._gradient_graph_cluster_last_update = self._gradient_updates

    def _recompute_gradient_graph_clusters(self):
        if (
            not self._gradient_graph_cluster_enabled
            or self._gradient_graph_relations is None
        ):
            return
        if not self._gradient_graph_relations:
            self._gradient_graph_clusters = {}
            self._gradient_graph_cluster_members = {}
            self.config.log(
                "Glow gradient graph clustering skipped: no relation stats available."
            )
            return
        allowed_relations = None
        top_relations = self._gradient_graph_cluster_top_relations
        if (
            top_relations > 0
            and self._gradient_graph_partitions is not None
            and self._gradient_graph_partitions
        ):
            allowed_relations = {}
            for pid, rels in self._gradient_graph_partitions.items():
                if not rels:
                    continue
                ordered = sorted(
                    rels.items(),
                    key=lambda item: item[1].get("avg", 0.0),
                    reverse=True,
                )
                keep = {rel_id for rel_id, _ in ordered[:top_relations]}
                allowed_relations[pid] = keep
        pair_scores = defaultdict(float)
        pair_counts = defaultdict(int)
        max_parts = self._gradient_graph_cluster_max_partitions_per_relation
        for rel_id, part_map in self._gradient_graph_relations.items():
            if not part_map:
                continue
            items = []
            for pid, stats in part_map.items():
                if allowed_relations is not None:
                    rel_set = allowed_relations.get(pid)
                    if rel_set is None or rel_id not in rel_set:
                        continue
                avg = stats.get("avg")
                if avg is None:
                    count = max(1, stats.get("count", 0))
                    avg = stats.get("sum", 0.0) / count
                items.append((pid, float(avg)))
            if len(items) < 2:
                continue
            if max_parts > 0 and len(items) > max_parts:
                items = sorted(
                    items, key=lambda item: item[1], reverse=True
                )[:max_parts]
            for (pid_i, val_i), (pid_j, val_j) in itertools.combinations(
                items, 2
            ):
                denom = val_i + val_j
                if denom <= 0:
                    continue
                affinity = 1.0 - abs(val_i - val_j) / denom
                pair = tuple(sorted((int(pid_i), int(pid_j))))
                pair_scores[pair] += affinity
                pair_counts[pair] += 1
        all_partitions = list(range(self.num_partitions))
        parent = {pid: pid for pid in all_partitions}
        sizes = {pid: 1 for pid in all_partitions}

        def find(pid):
            root = pid
            while parent[root] != root:
                root = parent[root]
            while parent[pid] != pid:
                nxt = parent[pid]
                parent[pid] = root
                pid = nxt
            return root

        def union(pid_i, pid_j):
            root_i = find(pid_i)
            root_j = find(pid_j)
            if root_i == root_j:
                return False
            max_size = self._gradient_graph_cluster_max_size
            if max_size > 0 and sizes[root_i] + sizes[root_j] > max_size:
                return False
            if sizes[root_i] < sizes[root_j]:
                root_i, root_j = root_j, root_i
            parent[root_j] = root_i
            sizes[root_i] += sizes[root_j]
            return True

        min_affinity = self._gradient_graph_cluster_min_affinity
        min_shared = self._gradient_graph_cluster_min_shared
        pairs = []
        for pair, score in pair_scores.items():
            count = pair_counts.get(pair, 0)
            if count <= 0:
                continue
            avg = score / count
            pairs.append((avg, count, pair))
        pairs.sort(key=lambda item: item[0], reverse=True)
        for avg, count, (pid_i, pid_j) in pairs:
            if count < min_shared:
                continue
            if avg < min_affinity:
                continue
            if avg <= 0.0 and min_affinity <= 0.0:
                continue
            union(pid_i, pid_j)
        cluster_members = defaultdict(list)
        for pid in all_partitions:
            cluster_members[find(pid)].append(pid)
        clusters = {}
        for root, members in cluster_members.items():
            members_sorted = sorted(members)
            for pid in members_sorted:
                clusters[pid] = root
            cluster_members[root] = members_sorted
        self._gradient_graph_clusters = clusters
        self._gradient_graph_cluster_members = dict(cluster_members)
        max_size = max((len(m) for m in cluster_members.values()), default=0)
        min_size = min((len(m) for m in cluster_members.values()), default=0)
        self.config.log(
            "Glow gradient graph clustering produced "
            f"{len(cluster_members)} clusters "
            f"(min_size={min_size}, max_size={max_size})."
        )

    def _update_gradient_graph(
        self, partition_id, rel_ids_list, rel_sums_list, rel_counts_list
    ):
        if not self._gradient_graph_enabled or self._gradient_graph_partitions is None:
            return
        part_edges = self._gradient_graph_partitions[partition_id]
        rel_edges = self._gradient_graph_relations
        for rel_id, rel_sum, rel_count in zip(
            rel_ids_list, rel_sums_list, rel_counts_list
        ):
            if rel_count <= 0:
                continue
            rel_id = int(rel_id)
            edge = part_edges.setdefault(rel_id, {"sum": 0.0, "count": 0})
            edge["sum"] += float(rel_sum)
            edge["count"] += int(rel_count)
            edge["avg"] = edge["sum"] / max(1, edge["count"])
            if rel_edges is not None:
                rel_edge = rel_edges[rel_id].setdefault(
                    partition_id, {"sum": 0.0, "count": 0}
                )
                rel_edge["sum"] += float(rel_sum)
                rel_edge["count"] += int(rel_count)
                rel_edge["avg"] = rel_edge["sum"] / max(1, rel_edge["count"])
        if self._gradient_graph_top_relations > 0:
            self._prune_gradient_graph_partition(partition_id)
        self._maybe_update_gradient_graph_clusters()

    def _prune_gradient_graph_partition(self, partition_id):
        if (
            not self._gradient_graph_enabled
            or self._gradient_graph_partitions is None
            or self._gradient_graph_relations is None
        ):
            return
        part_edges = self._gradient_graph_partitions.get(partition_id)
        if not part_edges:
            return
        max_edges = self._gradient_graph_top_relations
        if max_edges <= 0 or len(part_edges) <= max_edges:
            return
        ordered = sorted(
            part_edges.items(),
            key=lambda item: item[1].get("avg", 0.0),
            reverse=True,
        )
        keep_ids = {rel_id for rel_id, _ in ordered[:max_edges]}
        for rel_id in list(part_edges.keys()):
            if rel_id in keep_ids:
                continue
            part_edges.pop(rel_id, None)
            rel_map = self._gradient_graph_relations.get(rel_id)
            if rel_map is None:
                continue
            rel_map.pop(partition_id, None)
            if not rel_map:
                self._gradient_graph_relations.pop(rel_id, None)

    def _export_gradient_graph(self, final: bool = False):
        if (
            not self._gradient_graph_enabled
            or self._gradient_graph_partitions is None
            or not self._gradient_graph_partitions
        ):
            return
        if self._gradient_graph_cluster_enabled:
            self._maybe_update_gradient_graph_clusters(force=final)
        output_folder = Path(self.config.folder) / "gradient_graph"
        output_folder.mkdir(parents=True, exist_ok=True)
        if final:
            graph_path = output_folder / "gradient_graph_final.json"
        else:
            graph_path = output_folder / f"gradient_graph_{self._gradient_updates}.json"
        edges = []
        for pid, rels in self._gradient_graph_partitions.items():
            for rel_id, stats in rels.items():
                count = max(1, int(stats.get("count", 0)))
                edges.append(
                    {
                        "partition": self._serialize_partition_key(pid),
                        "relation": int(rel_id),
                        "sum": float(stats.get("sum", 0.0)),
                        "count": int(stats.get("count", 0)),
                        "avg": float(stats.get("sum", 0.0)) / count,
                    }
                )
        clusters_payload = None
        if self._gradient_graph_clusters:
            cluster_members = defaultdict(list)
            for pid, cid in self._gradient_graph_clusters.items():
                cluster_members[cid].append(pid)
            clusters_payload = []
            for cid, members in sorted(
                cluster_members.items(), key=lambda item: len(item[1]), reverse=True
            ):
                clusters_payload.append(
                    {
                        "id": self._serialize_partition_key(cid),
                        "partitions": [
                            self._serialize_partition_key(pid)
                            for pid in sorted(members)
                        ],
                    }
                )
        try:
            with open(graph_path, "w") as fp:
                json.dump(
                    {
                        "num_partitions": int(self.num_partitions),
                        "num_relations": int(self.dataset.num_relations()),
                        "snapshot_index": int(self._gradient_updates),
                        "final": bool(final),
                        "top_relations": int(self._gradient_graph_top_relations),
                        "cluster_snapshot_index": int(
                            self._gradient_graph_cluster_last_update
                        ),
                        "cluster_config": {
                            "enabled": bool(self._gradient_graph_cluster_enabled),
                            "update_interval": int(
                                self._gradient_graph_cluster_update_interval
                            ),
                            "min_affinity": float(
                                self._gradient_graph_cluster_min_affinity
                            ),
                            "min_shared_relations": int(
                                self._gradient_graph_cluster_min_shared
                            ),
                            "max_cluster_size": int(
                                self._gradient_graph_cluster_max_size
                            ),
                            "top_relations": int(
                                self._gradient_graph_cluster_top_relations
                            ),
                            "max_partitions_per_relation": int(
                                self._gradient_graph_cluster_max_partitions_per_relation
                            ),
                        },
                        "clusters": clusters_payload,
                        "edges": edges,
                    },
                    fp,
                    indent=2,
                )
            self.config.log(
                f"Exported gradient bipartite graph with {len(edges)} edges to {graph_path}."
            )
        except Exception as exc:
            self.config.log(f"Failed to export gradient graph: {exc}")

    def _export_relation_gradient_statistics(self, output_folder, final: bool = False):
        if not self.partition_relation_gradient_stats:
            return
        if final:
            snapshot_path = output_folder / "relation_gradient_snapshot_final.json"
        else:
            snapshot_path = output_folder / f"relation_gradient_snapshot_{self._gradient_updates}.json"
        edges = []
        for pid, rel_stats in self.partition_relation_gradient_stats.items():
            partition_key = self._serialize_partition_key(pid)
            for rel_id, stats in rel_stats.items():
                count = max(0, stats["count"])
                if count <= 0:
                    continue
                edges.append(
                    {
                        "partition": partition_key,
                        "relation": int(rel_id),
                        "sum": stats["sum"],
                        "count": stats["count"],
                        "avg": stats["sum"] / max(1, stats["count"]),
                    }
                )
        if not edges:
            return
        try:
            with open(snapshot_path, "w") as fp:
                json.dump(
                    {
                        "num_partitions": self.num_partitions,
                        "num_relations": self.dataset.num_relations(),
                        "snapshot_index": self._gradient_updates,
                        "final": final,
                        "edges": edges,
                    },
                    fp,
                    indent=2,
                )
            self.config.log(
                f"Exported relation gradient snapshot with {len(edges)} edges to {snapshot_path}."
            )
        except Exception as exc:
            self.config.log(f"Failed to export relation gradient snapshot: {exc}")

    def _register_partition_result(
        self, rank, step_time, reported_version=None, reported_chunk_size=0
    ):
        partition_id = self.active_partition_per_worker.get(rank)
        if partition_id is None:
            return 0, -1
        chunk_size = self.active_partition_chunk_sizes.get(rank, 0)
        if reported_chunk_size and reported_chunk_size > 0:
            chunk_size = reported_chunk_size
        expected_version = self.active_partition_versions.get(rank)
        conflict = (
            reported_version is not None
            and expected_version is not None
            and reported_version != expected_version
        )
        if conflict:
            self.config.log(
                f"Partition version mismatch for worker {rank}: "
                f"expected {expected_version}, got {reported_version}."
            )
        history = self.partition_stats.setdefault(partition_id, deque(maxlen=8))
        history.append(step_time)
        avg_time = sum(history) / len(history)
        self._handle_partition_feedback(partition_id, avg_time)
        self._after_partition_result(
            partition_id, avg_time, conflict=conflict, chunk_size=chunk_size
        )
        status = 0
        aux_version = -1
        if reported_version is not None:
            committed = self.partition_committed_versions.get(partition_id, -1)
            if reported_version < committed:
                status = 1
                aux_version = self.partition_issue_versions[partition_id]
                self.partition_issue_versions[partition_id] = aux_version + 1
            else:
                self.partition_committed_versions[partition_id] = max(
                    committed, reported_version
                )
        return status, aux_version

    def _register_window_result(
        self,
        rank,
        step_time,
        window_members,
        window_versions,
        reported_chunk_size=0,
    ):
        if window_members is None or window_versions is None:
            return []
        members = [int(x) for x in window_members.tolist()]
        versions = [int(x) for x in window_versions.tolist()]
        expected_versions = self.active_partition_versions.get(rank)
        if not isinstance(expected_versions, dict):
            expected_versions = {}
        chunk_size = self.active_partition_chunk_sizes.get(rank, 0)
        if reported_chunk_size and reported_chunk_size > 0:
            chunk_size = reported_chunk_size
        per_chunk = 0
        if members:
            per_chunk = int(chunk_size / max(1, len(members)))
        conflicts = []
        for pid, reported_version in zip(members, versions):
            expected = expected_versions.get(pid)
            conflict = (
                reported_version is not None
                and expected is not None
                and reported_version != expected
            )
            history = self.partition_stats.setdefault(pid, deque(maxlen=8))
            history.append(step_time)
            avg_time = sum(history) / len(history)
            self._handle_partition_feedback(pid, avg_time)
            self._after_partition_result(
                pid, avg_time, conflict=conflict, chunk_size=per_chunk
            )
            if reported_version is None:
                continue
            committed = self.partition_committed_versions.get(pid, -1)
            if reported_version < committed:
                replay_version = self.partition_issue_versions[pid]
                self.partition_issue_versions[pid] = replay_version + 1
                conflicts.append((pid, reported_version, replay_version))
            else:
                self.partition_committed_versions[pid] = max(
                    committed, reported_version
                )
        return conflicts

    def _after_partition_result(self, partition_id, avg_time, **kwargs):
        pass

    def _handle_partition_feedback(self, partition_id, avg_time):
        if not self.work_to_do:
            return
        ordered = self._order_by_feedback(list(self.work_to_do))
        self.work_to_do = deque(ordered)

    def _order_by_feedback(self, partition_ids):
        if not self.partition_stats:
            return partition_ids
        scored = []
        for idx, pid in enumerate(partition_ids):
            history = self.partition_stats.get(pid)
            if not history:
                avg = 0.0
            else:
                avg = sum(history) / len(history)
            scored.append((-avg, idx))
        scored.sort()
        return [partition_ids[idx] for _, idx in scored]

    def _load_reordered_partitions(self, num_partitions):
        dataset_folder = Path(self.dataset.folder)
        path = (
            dataset_folder
            / "partitions"
            / self.dataset._partition_type
            / f"num_{num_partitions}"
            / "partition_triples.npz"
        )
        if not path.is_file():
            return None
        try:
            data = np.load(path, allow_pickle=False)
            partitions = []
            for part_id in range(num_partitions):
                key = f"part_{part_id}"
                if key not in data:
                    raise KeyError(f"{key} missing in {path}")
                arr = data[key].astype(TORCH_TO_NP_DTYPE[self.data_type])
                partitions.append(torch.from_numpy(arr).contiguous())
            self.config.log(f"Loaded locality-aware partition ordering from {path}.")
            return partitions
        except Exception as e:
            self.config.log(f"Failed to load locality-aware ordering from {path}: {e}")
            return None

    def _get_max_entities(self):
        return 0

    def _get_max_relations(self):
        return 0

    def _load_partitions(self, num_partitions):
        raise NotImplementedError()


class AdaptiveWorkScheduler(WorkScheduler):
    def __init__(self, config, dataset):
        super(AdaptiveWorkScheduler, self).__init__(config=config, dataset=dataset)
        cfg = config.get("job.distributed.scheduler_feedback") or {}
        self._adaptive_enabled = bool(cfg.get("enable", False))
        self._adaptive_min_history = max(1, int(cfg.get("min_history", 3)))
        self._adaptive_slow_factor = max(1.0, float(cfg.get("slow_partition_factor", 1.5)))
        self._adaptive_target_chunk = max(0, int(cfg.get("target_chunk_size", 0)))
        self._adaptive_min_chunk = max(1, int(cfg.get("min_chunk_size", 1)))
        self._adaptive_max_splits = max(1, int(cfg.get("max_splits_per_partition", 8)))
        decay = float(cfg.get("ema_decay", 0.5))
        decay = min(0.999, max(0.0, decay))
        self._adaptive_decay = decay
        self._adaptive_global_avg = None
        self._adaptive_offsets = defaultdict(int)
        self._adaptive_chunk_sizes = defaultdict(int)
        self._adaptive_split_counts = defaultdict(int)
        self._adaptive_total_lengths: Dict = {}
        if self._adaptive_enabled and config.get(
            "job.distributed.materialize_partition_batches"
        ):
            self._adaptive_enabled = False
            config.log(
                "Disabled scheduler feedback chunking because partition batches are materialized."
            )

    def _supports_adaptive_feedback(self):
        return False

    def _after_partition_result(self, partition_id, avg_time, **kwargs):
        super(AdaptiveWorkScheduler, self)._after_partition_result(
            partition_id, avg_time, **kwargs
        )
        if (
            not self._adaptive_enabled
            or not self._supports_adaptive_feedback()
            or partition_id is None
        ):
            return
        history = self.partition_stats.get(partition_id)
        if not history or len(history) < self._adaptive_min_history:
            return
        if self._adaptive_global_avg is None:
            self._adaptive_global_avg = avg_time
        else:
            self._adaptive_global_avg = (
                self._adaptive_decay * self._adaptive_global_avg
                + (1.0 - self._adaptive_decay) * avg_time
            )
        if self._adaptive_global_avg <= 0:
            return
        if avg_time < self._adaptive_global_avg * self._adaptive_slow_factor:
            return
        total_len = self._adaptive_total_lengths.get(partition_id)
        if total_len is None:
            total_len = self._adaptive_get_partition_length(partition_id)
            if total_len is not None:
                self._adaptive_total_lengths[partition_id] = total_len
        if total_len is None or total_len < self._adaptive_min_chunk * 2:
            return
        if self._adaptive_split_counts[partition_id] >= self._adaptive_max_splits:
            return
        chunk_size = self._adaptive_target_chunk
        if chunk_size <= 0 or chunk_size >= total_len:
            chunk_size = max(self._adaptive_min_chunk, total_len // 2)
        if chunk_size <= 0 or chunk_size >= total_len:
            return
        self._adaptive_chunk_sizes[partition_id] = chunk_size
        self._adaptive_split_counts[partition_id] += 1

    def _adaptive_get_partition_length(self, partition_id):
        return self._adaptive_total_lengths.get(partition_id)

    def _adaptive_requeue_partition(self, partition_id):
        if hasattr(self, "work_to_do") and isinstance(self.work_to_do, deque):
            self.work_to_do.appendleft(partition_id)

    def _adaptive_reset_state(self):
        self._adaptive_offsets.clear()
        self._adaptive_total_lengths.clear()

    def _adaptive_maybe_slice(self, partition_id, partition_tensor):
        if (
            not self._adaptive_enabled
            or not self._supports_adaptive_feedback()
            or partition_tensor is None
        ):
            return partition_tensor
        total = len(partition_tensor)
        if total == 0:
            return partition_tensor
        if partition_id not in self._adaptive_total_lengths:
            self._adaptive_total_lengths[partition_id] = total
        chunk_size = self._adaptive_chunk_sizes.get(partition_id, 0)
        if chunk_size <= 0 or chunk_size >= total:
            self._adaptive_offsets[partition_id] = 0
            return partition_tensor
        start = self._adaptive_offsets[partition_id]
        if start >= total:
            self._adaptive_chunk_sizes[partition_id] = 0
            self._adaptive_offsets[partition_id] = 0
            return partition_tensor
        take = min(chunk_size, total - start)
        end = start + take
        self._adaptive_offsets[partition_id] = end
        if end < total:
            self._adaptive_requeue_partition(partition_id)
        else:
            self._adaptive_chunk_sizes[partition_id] = 0
            self._adaptive_offsets[partition_id] = 0
        return partition_tensor.narrow(0, start, take).clone()

    def _refill_work(self):
        self._adaptive_reset_state()
        return super(AdaptiveWorkScheduler, self)._refill_work()


class RandomWorkScheduler(AdaptiveWorkScheduler):
    def __init__(
        self,
        config,
        dataset,
    ):
        dataset._partition_type = "random"
        super(RandomWorkScheduler, self).__init__(
            config=config,
            dataset=dataset,
        )

    def _next_work(
        self, rank, machine_id
    ) -> WorkPackage:
        """add work/partitions to the list of work to do"""
        try:
            work_package = WorkPackage()
            work_package.partition_id = self.work_to_do.pop()
            partition_tensor = self.partitions[work_package.partition_id]
            partition_slice = self._adaptive_maybe_slice(
                work_package.partition_id, partition_tensor
            )
            if partition_slice is None or len(partition_slice) == 0:
                return self._next_work(rank, machine_id)
            work_package.partition_data = partition_slice
            # those are not entities in the partition but "local" entities for the
            #  worker to allow local sampling
            work_package.entities_in_partition = self.local_entities[rank]
            return work_package
        except IndexError:
            return WorkPackage()

    def _load_partitions(self, num_partitions):
        num_triples = len(self.dataset.split("train"))
        # Use long indices for safe tensor indexing.
        permuted_triple_index = torch.randperm(num_triples)
        partitions = list(torch.chunk(permuted_triple_index, num_partitions))
        partitions = [p.clone() for p in partitions]
        return partitions

    def _refill_work(self):
        if self.repartition_epoch:
            self.partitions = self._load_partitions(self.num_partitions)
            self._define_local_entities()
        super(RandomWorkScheduler, self)._refill_work()
        reordered = self._order_by_feedback(list(self.work_to_do))
        self.work_to_do = deque(reordered)

    def _supports_adaptive_feedback(self):
        return True

    def _adaptive_get_partition_length(self, partition_id):
        if 0 <= partition_id < len(self.partitions):
            return len(self.partitions[partition_id])
        return None


class RelationWorkScheduler(AdaptiveWorkScheduler):
    def __init__(
        self,
        config,
        dataset,
    ):
        dataset._partition_type = "relation"
        super(RelationWorkScheduler, self).__init__(
            config=config,
            dataset=dataset,
        )

    def _init_in_started_process(self):
        super(RelationWorkScheduler, self)._init_in_started_process()
        self.relations_to_partition = self.dataset.load_relations_to_partitions(self.num_partitions)
        self.relations_to_partition = self._get_relations_in_partition()

    def _next_work(
        self, rank, machine_id
    ) -> WorkPackage:
        """add work/partitions to the list of work to do"""
        try:
            work_package = WorkPackage()
            work_package.partition_id = self.work_to_do.pop()
            partition_tensor = self.partitions[work_package.partition_id]
            partition_slice = self._adaptive_maybe_slice(
                work_package.partition_id, partition_tensor
            )
            if partition_slice is None or len(partition_slice) == 0:
                return self._next_work(rank, machine_id)
            work_package.partition_data = partition_slice
            work_package.relations_in_partition = self.relations_to_partition[work_package.partition_id]
            # those are not entities in the partition but "local" entities for the
            #  worker to allow local sampling
            work_package.entities_in_partition = self.local_entities[rank]
            return work_package
        except IndexError:
            return WorkPackage()

    def _load_partitions(self, num_partitions):
        np_type = TORCH_TO_NP_DTYPE[self.data_type]
        reordered = self._load_reordered_partitions(num_partitions)
        if reordered is not None:
            return reordered
        partition_assignment = self.dataset.load_train_partitions(num_partitions)
        # todo: let the partitions start at zero, then we do not need this unique
        partition_indexes = np.unique(partition_assignment)
        partitions = [
            torch.from_numpy(np.where(partition_assignment == i)[0].astype(np_type)).contiguous()
            for i in partition_indexes
        ]
        return partitions

    def _get_relations_in_partition(self):
        np_type = TORCH_TO_NP_DTYPE[self.data_type]
        relations_in_partition = dict()
        for partition in range(self.num_partitions):
            relations_in_partition[partition] = torch.from_numpy(
                np.where((self.relations_to_partition == partition),)[0].astype(np_type)
            ).contiguous()
        return relations_in_partition

    def _refill_work(self):
        if self.repartition_epoch:
            # self.partitions = self._load_partitions(self.num_partitions)
            self._define_local_entities()
        super(RelationWorkScheduler, self)._refill_work()
        reordered = self._order_by_feedback(list(self.work_to_do))
        self.work_to_do = deque(reordered)

    def _supports_adaptive_feedback(self):
        return True

    def _adaptive_get_partition_length(self, partition_id):
        if 0 <= partition_id < len(self.partitions):
            return len(self.partitions[partition_id])
        return None


class GraphCutWorkScheduler(WorkScheduler):
    def __init__(
        self,
        config,
        dataset,
    ):
        dataset._partition_type = "graph-cut"
        super(GraphCutWorkScheduler, self).__init__(
            config=config,
            dataset=dataset,
        )
        self.partition_costs = None
        self.max_chunk_size = max(
            0, int(config.get("job.distributed.graph_cut.max_chunk_size"))
        )
        self.min_chunk_size = max(
            0, int(config.get("job.distributed.graph_cut.min_chunk_size"))
        )
        self.dynamic_chunking = bool(
            config.get("job.distributed.graph_cut.dynamic_chunking")
        )
        self.target_chunk_time = float(
            config.get("job.distributed.graph_cut.target_chunk_time")
        )
        self.chunk_warmup_epochs = int(
            config.get("job.distributed.graph_cut.chunk_warmup_epochs")
        )
        self.export_feedback = bool(
            config.get("job.distributed.graph_cut.export_feedback")
        )
        self.completed_epochs = 0
        self._base_chunk_size = self.max_chunk_size
        if self.chunk_warmup_epochs > 0 and self.max_chunk_size > 0:
            warm_start = self.max_chunk_size // 4
            if self.min_chunk_size > 0:
                warm_start = max(self.min_chunk_size, warm_start)
            self._base_chunk_size = max(1, warm_start)
        self.partition_chunk_size = defaultdict(self._default_chunk_size)
        self.partition_next_offset = defaultdict(int)
        self.partition_done_offset = defaultdict(int)

    def _init_in_started_process(self):
        super(GraphCutWorkScheduler, self)._init_in_started_process()
        self.entities_to_partition = self.dataset.load_entities_to_partitions(self.num_partitions)
        self.entities_to_partition = self._get_entities_in_partition()
        self.previous_partition_per_worker = defaultdict(lambda: None)
        self.partition_costs = self._load_partition_costs(self.num_partitions)
        self._reset_partition_progress()

    def _config_check(self, config):
        super(GraphCutWorkScheduler, self)._config_check(config)
        if config.get("job.distributed.entity_sync_level") == "partition":
            raise ValueError(
                "Metis partitioning does not support entity sync level 'parititon'. "
                "Triples still have outside partition accesses."
            )

    def _load_partitions(self, num_partitions):
        np_type = TORCH_TO_NP_DTYPE[self.data_type]
        reordered = self._load_reordered_partitions(num_partitions)
        if reordered is not None:
            return reordered
        partition_assignment = self.dataset.load_train_partitions(num_partitions)
        # todo: let the partitions start at zero, then we do not need this unique
        partition_indexes = np.unique(partition_assignment)
        partitions = [
            torch.from_numpy(np.where(partition_assignment == i)[0].astype(np_type)).contiguous()
            for i in partition_indexes
        ]
        return partitions

    def _get_entities_in_partition(self):
        np_type = TORCH_TO_NP_DTYPE[self.data_type]
        entities_in_partition = dict()
        for partition in range(self.num_partitions):
            entities_in_partition[partition] = torch.from_numpy(
                np.where((self.entities_to_partition == partition),)[0].astype(np_type)
            ).contiguous()
        return entities_in_partition

    def _get_max_entities(self):
        return max([len(i) for i in self.entities_to_partition.values()])

    def _refill_work(self):
        self._reset_partition_progress()
        super(GraphCutWorkScheduler, self)._refill_work()
        if (
            self.partition_costs is not None
            and len(self.partition_costs) == self.num_partitions
        ):
            ordered = sorted(
                range(self.num_partitions),
                key=lambda idx: self.partition_costs[idx],
                reverse=True,
            )
            self.work_to_do = deque(ordered)

    def _load_partition_costs(self, num_partitions):
        if not self.config.get("job.distributed.graph_cut.cost_aware_schedule"):
            return None
        dataset_folder = Path(self.dataset.folder)
        cost_path = (
            dataset_folder
            / "partitions"
            / "graph-cut"
            / f"num_{num_partitions}"
            / "partition_costs.npy"
        )
        if not cost_path.is_file():
            self.config.log(
                f"graph-cut cost-aware scheduling enabled, but {cost_path} not found."
            )
            return None
        try:
            costs = np.load(cost_path)
        except Exception as e:
            self.config.log(f"Failed to load graph-cut partition costs: {e}")
            return None
        if len(costs) != num_partitions:
            self.config.log(
                f"Ignoring partition costs: expected {num_partitions} entries, got {len(costs)}."
            )
            return None
        self.config.log(
            f"Loaded graph-cut partition costs from {cost_path} for cost-aware scheduling."
        )
        return costs

    def _reset_partition_progress(self):
        self.partition_next_offset = defaultdict(int)
        self.partition_done_offset = defaultdict(int)
        if not hasattr(self, "partition_chunk_size"):
            self.partition_chunk_size = defaultdict(self._default_chunk_size)

    def _register_partition_result(
        self, rank, step_time, reported_version=None, reported_chunk_size=0
    ):
        partition_id = self.active_partition_per_worker.get(rank)
        if partition_id is None:
            return 0, -1
        if self.partition_costs is None or len(self.partition_costs) != self.num_partitions:
            self.partition_costs = np.zeros(self.num_partitions, dtype=np.float64)
        self.partition_costs[partition_id] = step_time
        remaining = sorted(
            list(self.work_to_do), key=lambda idx: self.partition_costs[idx], reverse=True
        )
        self.work_to_do = deque(remaining)
        chunk_size = self.active_partition_chunk_sizes.get(rank, 0)
        if reported_chunk_size and reported_chunk_size > 0:
            chunk_size = reported_chunk_size
        self._maybe_adjust_chunk_size(partition_id, chunk_size, step_time)
        return super(GraphCutWorkScheduler, self)._register_partition_result(
            rank,
            step_time,
            reported_version=reported_version,
            reported_chunk_size=reported_chunk_size,
        )

    def _next_work(
        self, rank, machine_id
    ) -> WorkPackage:
        try:
            work_package = WorkPackage()
            prev_work_id = self.previous_partition_per_worker[rank]
            if prev_work_id is not None and prev_work_id in self.work_to_do:
                work_package.partition_id = prev_work_id
                del self.work_to_do[self.work_to_do.index(prev_work_id)]
            else:
                work_package.partition_id = self.work_to_do.pop()
            partition_tensor = self.partitions[work_package.partition_id]
            partition_slice = self._slice_partition(
                work_package.partition_id, partition_tensor
            )
            if partition_slice is None or len(partition_slice) == 0:
                return self._next_work(rank, machine_id)
            work_package.partition_data = partition_slice
            work_package.entities_in_partition = self.entities_to_partition[work_package.partition_id]
            return work_package
        except IndexError:
            return WorkPackage()

    def _slice_partition(self, partition_id, partition_tensor):
        total = len(partition_tensor)
        chunk_cap = self._get_partition_chunk_size(partition_id, total)
        if chunk_cap <= 0 or chunk_cap >= total:
            self.partition_next_offset[partition_id] = total
            return partition_tensor
        start = self.partition_next_offset[partition_id]
        if start >= total:
            return None
        chunk_len = min(chunk_cap, total - start)
        end = start + chunk_len
        self.partition_next_offset[partition_id] = end
        if end < total:
            self.work_to_do.appendleft(partition_id)
        return partition_tensor.narrow(0, start, chunk_len).clone()

    def _handle_work_done(self, rank):
        if self.max_chunk_size <= 0:
            return super(GraphCutWorkScheduler, self)._handle_work_done(rank)
        partition_id = self.active_partition_per_worker.get(rank)
        chunk_size = self.active_partition_chunk_sizes.pop(rank, 0)
        if partition_id is None or chunk_size <= 0:
            return super(GraphCutWorkScheduler, self)._handle_work_done(rank)
        total = len(self.partitions[partition_id])
        self.partition_done_offset[partition_id] += chunk_size
        if self.partition_done_offset[partition_id] >= total:
            return super(GraphCutWorkScheduler, self)._handle_work_done(rank)
        # partition still has pending chunks; release worker but avoid double-counting
        self.active_partition_per_worker.pop(rank, None)
        print(
            f"trainer {rank} finished chunk "
            f"{self.partition_done_offset[partition_id]}/{total} of partition {partition_id}"
        )

    def _default_chunk_size(self):
        if self._base_chunk_size:
            return self._base_chunk_size
        return self.max_chunk_size

    def _get_partition_chunk_size(self, partition_id, total):
        size = self.partition_chunk_size.get(partition_id)
        if not size or size <= 0:
            if self.max_chunk_size > 0:
                size = self.max_chunk_size
            else:
                size = total
            self.partition_chunk_size[partition_id] = size
        return min(size, total)

    def _maybe_adjust_chunk_size(self, partition_id, chunk_size, step_time):
        if (
            not self.dynamic_chunking
            or self.max_chunk_size <= 0
            or step_time <= 0
            or chunk_size <= 0
        ):
            return
        if self.target_chunk_time <= 0:
            return
        current = self.partition_chunk_size.get(partition_id, chunk_size)
        min_size = self.min_chunk_size if self.min_chunk_size > 0 else max(1, chunk_size // 2)
        adjusted = False
        if step_time > self.target_chunk_time * 1.25 and current > min_size:
            new_size = max(min_size, int(current * (self.target_chunk_time / step_time)))
            current = max(1, new_size)
            adjusted = True
        elif step_time < self.target_chunk_time * 0.5 and current < self.max_chunk_size:
            new_size = min(self.max_chunk_size, int(current * (self.target_chunk_time / step_time)))
            current = max(1, new_size)
            adjusted = True
        if adjusted:
            self.partition_chunk_size[partition_id] = current

    def _on_epoch_completed(self):
        self.completed_epochs += 1
        if (
            self.chunk_warmup_epochs > 0
            and self.max_chunk_size > 0
            and self.completed_epochs <= self.chunk_warmup_epochs
        ):
            progress = self.completed_epochs / self.chunk_warmup_epochs
            target = max(
                self.min_chunk_size or 1,
                int(self.max_chunk_size * progress),
            )
            if target > self._base_chunk_size:
                self._base_chunk_size = target
                for pid in range(self.num_partitions):
                    current = self.partition_chunk_size.get(pid, target)
                    if current < target:
                        self.partition_chunk_size[pid] = target

    def _on_scheduler_shutdown(self):
        super(GraphCutWorkScheduler, self)._on_scheduler_shutdown()
        if not self.export_feedback or not self.partition_stats:
            return
        dataset_folder = Path(self.dataset.folder)
        output_folder = (
            dataset_folder
            / "partitions"
            / "graph-cut"
            / f"num_{self.num_partitions}"
        )
        output_folder.mkdir(parents=True, exist_ok=True)
        feedback = {}
        for pid, history in self.partition_stats.items():
            if history:
                feedback[pid] = sum(history) / len(history)
        path = output_folder / "partition_feedback.json"
        try:
            with open(path, "w") as fp:
                json.dump({"avg_step_time": feedback}, fp, indent=2)
            self.config.log(f"Exported graph-cut feedback to {path}.")
        except Exception as e:
            self.config.log(f"Failed to export graph-cut feedback: {e}")


class GlowWorkScheduler(AdaptiveWorkScheduler):
    def __init__(self, config, dataset):
        glow_cfg = config.get("job.distributed.glow") or {}
        self._glow_debug = bool(glow_cfg.get("debug_log", False))
        self._glow_debug_verbose = bool(
            glow_cfg.get("debug_log_verbose", False)
        )
        if self._glow_debug_verbose:
            self._glow_debug = True
        self._glow_debug_interval = max(
            1, int(glow_cfg.get("debug_log_interval", 50))
        )
        self._glow_debug_window_limit = max(
            0, int(glow_cfg.get("debug_log_window_limit", 0))
        )
        self._glow_debug_pick_count = 0
        self._glow_debug_result_count = 0
        base_partition_type = glow_cfg.get("base_partition_type") or "random"
        dataset._partition_type = base_partition_type
        self._glow_base_partition_type = base_partition_type
        self._glow_stratified_partitions = False
        super(GlowWorkScheduler, self).__init__(config=config, dataset=dataset)
        self.glow_window_size = max(1, int(glow_cfg.get("window_size", 2)))
        overlap = int(glow_cfg.get("window_overlap", 1))
        self.glow_window_overlap = max(0, min(self.glow_window_size - 1, overlap))
        self.glow_min_gradient = max(1, int(glow_cfg.get("min_gradient_count", 1)))
        self.glow_windows_enabled = bool(glow_cfg.get("enable_windows", True))
        self.glow_window_work = bool(glow_cfg.get("window_work", False))
        if self.glow_window_work and not self.glow_windows_enabled:
            self.glow_windows_enabled = True
            config.log("Enabled glow windows because window_work is active.")
        self.glow_concurrent_windows = bool(
            glow_cfg.get("concurrent_windows", False)
        )
        self.glow_overlap_sampling = bool(
            glow_cfg.get("overlap_negative_sampling", False)
        )
        ns_sampling_type = config.get("negative_sampling.sampling_type")
        entity_sync_level = config.get("job.distributed.entity_sync_level")
        self._send_window_entities = bool(
            glow_cfg.get("prefetch_window_entities", False)
            or self.glow_overlap_sampling
            or (ns_sampling_type == "pooled" and entity_sync_level != "partition")
        )
        self.glow_affinity_alpha = float(glow_cfg.get("affinity_alpha", 0.3))
        self.glow_affinity_alpha = min(0.999, max(0.0, self.glow_affinity_alpha))
        bandit_cfg = glow_cfg.get("bandit") or {}
        self.glow_bandit_enabled = bool(bandit_cfg.get("enable", False))
        self.glow_reward_alpha = float(bandit_cfg.get("reward_alpha", 0.3))
        self.glow_reward_alpha = min(0.999, max(0.0, self.glow_reward_alpha))
        self.glow_reward_scale = float(bandit_cfg.get("reward_scale", 1.0))
        self.glow_conflict_penalty = float(bandit_cfg.get("conflict_penalty", 0.0))
        self.glow_queue_penalty_scale = float(
            bandit_cfg.get("queue_penalty_scale", 0.0)
        )
        self.glow_overlap_penalty = float(
            bandit_cfg.get("overlap_penalty", 0.0)
        )
        reshape_cfg = glow_cfg.get("reshape") or {}
        self.reshape_enabled = bool(reshape_cfg.get("enable", False))
        self.reshape_interval = max(1, int(reshape_cfg.get("check_interval", 200)))
        self.reshape_hot_splits = max(0, int(reshape_cfg.get("hot_splits", 1)))
        self.reshape_cold_merges = max(0, int(reshape_cfg.get("cold_merges", 1)))
        self.reshape_min_size = max(1, int(reshape_cfg.get("min_partition_size", 1024)))
        gradient_cfg = glow_cfg.get("gradient_clustering") or {}
        self._gradient_cluster_enabled = bool(gradient_cfg.get("enable", False))
        self._gradient_cluster_top_relations = max(
            1, int(gradient_cfg.get("top_relations", 32))
        )
        self._gradient_cluster_max_partitions_per_relation = max(
            1, int(gradient_cfg.get("max_partitions_per_relation", 4))
        )
        self._gradient_cluster_affinity_alpha = float(
            gradient_cfg.get("affinity_alpha", self.glow_affinity_alpha)
        )
        self._gradient_cluster_affinity_alpha = min(
            0.999, max(0.0, self._gradient_cluster_affinity_alpha)
        )
        self._gradient_cluster_min_shared = max(
            1, int(gradient_cfg.get("min_shared_relations", 1))
        )
        overlap_cfg = glow_cfg.get("causal_overlap") or {}
        self._causal_overlap_enabled = bool(overlap_cfg.get("enable", False))
        self._causal_overlap_max_workers = max(
            1, int(overlap_cfg.get("max_workers_per_partition", 1))
        )
        self._causal_overlap_duplicate = bool(
            overlap_cfg.get("duplicate_partitions", False)
        )
        if self._causal_overlap_enabled:
            self._causal_overlap_max_workers = min(
                self._causal_overlap_max_workers, max(1, self.num_clients)
            )
            config.log(
                "Glow causal_overlap config: "
                f"max_workers={self._causal_overlap_max_workers} "
                f"duplicate_partitions={self._causal_overlap_duplicate} "
                f"causal_merge_row="
                f"{bool(config.get('job.distributed.causal_merge_row'))}."
            )
            if getattr(self, "_adaptive_enabled", False):
                self._adaptive_enabled = False
                config.log(
                    "Disabled scheduler feedback chunking because "
                    "causal_overlap requires stable partition versions."
                )
            if not self._causal_overlap_duplicate:
                config.log(
                    "Glow causal_overlap enabled without duplicate_partitions; "
                    "overlap will not re-issue partitions to multiple workers."
                )
            if not bool(config.get("job.distributed.causal_merge_row")):
                config.log(
                    "Glow causal_overlap enabled but causal_merge_row is disabled; "
                    "overlap will not be conflict-safe until causal_merge_row is set."
                )
        if self.glow_window_work and getattr(self, "_adaptive_enabled", False):
            self._adaptive_enabled = False
            config.log(
                "Disabled scheduler feedback chunking because "
                "window_work uses multi-partition windows."
            )
        if self.glow_window_work and self._causal_overlap_enabled:
            self._causal_overlap_enabled = False
            config.log(
                "Disabled causal_overlap because window_work already "
                "issues multi-partition windows."
            )
        if self._glow_debug:
            config.log(
                "Glow debug logging enabled "
                f"(verbose={self._glow_debug_verbose}, "
                f"interval={self._glow_debug_interval})."
            )
            if self._glow_debug_interval < 5:
                config.log(
                    "Glow debug_log_interval is very small; "
                    "clamping to 5 to reduce logging overhead."
                )
                self._glow_debug_interval = 5
        self._glow_windows: deque = deque()
        self._current_window_entry = None
        self._max_window_entities = 0
        self._served_partitions = set()
        self._latest_gradients = {}
        self._window_scores = defaultdict(float)
        self._window_counts = defaultdict(int)
        self._partition_to_window = {}
        self._window_work_key_map = {}
        self._partition_entities_map = None
        self._max_partition_entities = 0
        self._partition_relations_map = None
        self._all_relations = None
        self._reshape_counter = 0
        self._co_gradient_scores = defaultdict(float)
        self._co_gradient_counts = defaultdict(int)
        self._relation_to_partitions = defaultdict(dict)
        self._partition_relation_topk = {}
        self._causal_overlap_queue = deque()
        self._causal_overlap_versions = {}
        self._causal_overlap_active = defaultdict(int)
        graph_cfg = config.get("job.distributed.gradient_graph") or {}
        repartition_cfg = graph_cfg.get("repartition") or {}
        self._gradient_repartition_enabled = bool(repartition_cfg.get("enable", False))
        self._gradient_repartition_max_moves = max(
            0, int(repartition_cfg.get("max_moves_per_partition", 0))
        )
        self._gradient_repartition_min_size = max(
            1, int(repartition_cfg.get("min_partition_size", 2048))
        )
        self._gradient_repartition_max_size = max(
            0, int(repartition_cfg.get("max_partition_size", 0))
        )
        self._gradient_repartition_max_relations = max(
            0, int(repartition_cfg.get("max_relations", 0))
        )
        self._train_triples_cache = None

    def _glow_log(self, message: str):
        if self._glow_debug:
            self.config.log(f"Glow debug: {message}")

    def _glow_log_verbose(self, message: str):
        if self._glow_debug_verbose:
            self.config.log(f"Glow debug: {message}")

    def _init_in_started_process(self):
        super(GlowWorkScheduler, self)._init_in_started_process()
        self._glow_log(
            "Init settings "
            f"base_partition_type={self._glow_base_partition_type}, "
            f"num_partitions={self.num_partitions}, "
            f"num_workers={self.num_clients}, "
            f"window_size={self.glow_window_size}, "
            f"window_overlap={self.glow_window_overlap}, "
            f"concurrent_windows={self.glow_concurrent_windows}, "
            f"window_work={self.glow_window_work}, "
            f"overlap_sampling={self.glow_overlap_sampling}, "
            f"prefetch_window_entities={self._send_window_entities}, "
            f"lookahead_negatives="
            f"{self.config.get('job.distributed.glow.lookahead_negatives')}, "
            f"bandit={self.glow_bandit_enabled}, "
            f"reshape={self.reshape_enabled}, "
            f"gradient_clustering={self._gradient_cluster_enabled}, "
            f"causal_overlap={self._causal_overlap_enabled}."
        )
        if self.glow_windows_enabled and not self.glow_window_work:
            self.config.log(
                "Glow window_work is disabled; windows only affect "
                "ordering/prefetch, not dispatch. Enable window_work to "
                "use window scheduling."
            )
        effective = getattr(self, "_glow_effective_partitions", None)
        if effective is not None and effective != self.num_partitions:
            self.config.log(
                "Glow scheduler using stratification pair partitions: "
                f"{self.num_partitions} -> {effective}."
            )
            self.num_partitions = int(effective)
            self.work_to_do = deque(list(range(self.num_partitions)))
        self._load_existing_gradients()
        self._init_partition_entity_map()
        self._init_partition_relation_map()
        actual_partitions = len(self.partitions)
        if actual_partitions != self.num_partitions:
            self._glow_log(
                "Adjusting num_partitions from "
                f"{self.num_partitions} to {actual_partitions} "
                "to match loaded partitions."
            )
            self.num_partitions = actual_partitions
            self.work_to_do = deque(list(range(self.num_partitions)))
        self._current_window_entry = None
        self._rebuild_glow_windows(list(range(self.num_partitions)))

    def _load_partitions(self, num_partitions):
        base_type = getattr(self.dataset, "_partition_type", "random")
        self._glow_log(
            f"Loading partitions base_type={base_type}, num_partitions={num_partitions}."
        )
        if base_type == "random":
            num_triples = len(self.dataset.split("train"))
            # Use long indices for safe tensor indexing.
            permuted_triple_index = torch.randperm(num_triples)
            partitions = list(torch.chunk(permuted_triple_index, num_partitions))
            partitions = [p.clone() for p in partitions]
            return partitions
        reordered = self._load_reordered_partitions(num_partitions)
        if reordered is not None:
            return reordered
        partition_assignment = self.dataset.load_train_partitions(num_partitions)
        np_type = TORCH_TO_NP_DTYPE[self.data_type]
        self._glow_stratified_partitions = (
            isinstance(partition_assignment, np.ndarray)
            and partition_assignment.ndim == 2
            and partition_assignment.shape[1] == 2
        )
        if self._glow_stratified_partitions:
            use_pairs = bool(
                self.config.get(
                    "job.distributed.glow.stratification_pairs", True
                )
            )
            if use_pairs:
                pairs = np.unique(partition_assignment, axis=0)
                self._glow_log(
                    "Stratification pairs enabled; "
                    f"unique_pairs={len(pairs)}."
                )
                partitions = []
                for pair in pairs:
                    mask = (
                        (partition_assignment[:, 0] == pair[0])
                        & (partition_assignment[:, 1] == pair[1])
                    )
                    idx = np.where(mask)[0].astype(np_type)
                    if idx.size == 0:
                        continue
                    partitions.append(
                        torch.from_numpy(idx).contiguous()
                    )
                self._glow_effective_partitions = len(partitions)
                return partitions
            self._glow_log("Stratification pairs disabled.")
        partition_indexes = np.unique(partition_assignment)
        partitions = [
            torch.from_numpy(
                np.unique(np.where(partition_assignment == i)[0]).astype(np_type)
                if self._glow_stratified_partitions
                else np.where(partition_assignment == i)[0].astype(np_type)
            ).contiguous()
            for i in partition_indexes
        ]
        return partitions

    def _supports_adaptive_feedback(self):
        return True

    def _order_by_gradient(self, partitions):
        gradients = self.partition_gradient_stats or self._latest_gradients
        if not gradients:
            return partitions
        scored = []
        for pid in partitions:
            stats = gradients.get(pid)
            if not stats or stats["count"] < self.glow_min_gradient:
                avg = 0.0
            else:
                avg = stats["sum"] / max(1, stats["count"])
            scored.append((-avg, pid))
        scored.sort()
        return [pid for _, pid in scored]

    def _order_partitions_by_gradient_graph_clusters(self, partitions):
        clusters = getattr(self, "_gradient_graph_clusters", None)
        if not clusters:
            return partitions
        cluster_members = defaultdict(list)
        for pid in partitions:
            cluster_id = clusters.get(pid, pid)
            cluster_members[cluster_id].append(pid)
        gradients = self.partition_gradient_stats or self._latest_gradients
        cluster_scores = []
        for cluster_id, members in cluster_members.items():
            total = 0.0
            count = 0
            for pid in members:
                stats = gradients.get(pid)
                if not stats or stats["count"] < self.glow_min_gradient:
                    continue
                total += stats["sum"] / max(1, stats["count"])
                count += 1
            avg = total / max(1, count)
            cluster_scores.append((-avg, cluster_id))
        cluster_scores.sort()
        ordered = []
        seen = set()
        for _, cluster_id in cluster_scores:
            members = cluster_members[cluster_id]
            for pid in self._order_by_gradient(members):
                if pid not in seen:
                    ordered.append(pid)
                    seen.add(pid)
        for pid in partitions:
            if pid not in seen:
                ordered.append(pid)
                seen.add(pid)
        return ordered

    def _rebuild_glow_windows(self, ordered_partitions, preserve_served=False):
        self._glow_windows = deque()
        self._max_window_entities = 0
        self._window_work_key_map.clear()
        if preserve_served and self._served_partitions:
            ordered_partitions = [
                pid for pid in ordered_partitions
                if pid not in self._served_partitions
            ]
        else:
            self._served_partitions = set()
        if not ordered_partitions or not self.glow_windows_enabled:
            return
        ordered_partitions = self._order_partitions_by_gradient_graph_clusters(
            list(ordered_partitions)
        )
        if (
            self._glow_debug
            and getattr(self, "_gradient_graph_clusters", None)
        ):
            cluster_ids = {
                cid for cid in self._gradient_graph_clusters.values()
            }
            self._glow_log(
                "Gradient graph clustering used for window ordering "
                f"(clusters={len(cluster_ids)})."
            )
        if self._co_gradient_scores:
            if self._glow_debug:
                self._glow_log(
                    "Co-gradient affinity used for window ordering "
                    f"(pairs={len(self._co_gradient_scores)})."
                )
            ordered_partitions = self._cluster_partitions_by_affinity(
                list(ordered_partitions)
            )
        self._partition_to_window.clear()
        self._current_window_entry = None
        step = self.glow_window_size - self.glow_window_overlap
        if step <= 0:
            step = self.glow_window_size
        for start in range(0, len(ordered_partitions), step):
            window = ordered_partitions[start : start + self.glow_window_size]
            if window:
                window_entry = {
                    "key": tuple(window),
                    "partitions": deque(window),
                }
                self._glow_windows.append(window_entry)
            if (
                (self.glow_overlap_sampling or self.glow_window_work)
                and self._partition_entities_map is not None
            ):
                window_count = self._estimate_window_entity_count(window)
                if window_count > self._max_window_entities:
                    self._max_window_entities = window_count
        self._sort_glow_windows()
        self._served_partitions = set()
        if self._glow_debug:
            window_count = len(self._glow_windows)
            entity_counts = []
            overlap_ratios = []
            window_stats = []
            if self._partition_entities_map is not None:
                for entry in list(self._glow_windows):
                    window_key = entry.get("key")
                    unique_count, total_count, overlap_ratio = (
                        self._estimate_window_entity_overlap(window_key)
                    )
                    entity_counts.append(unique_count)
                    overlap_ratios.append(overlap_ratio)
                    window_stats.append(
                        (window_key, unique_count, total_count, overlap_ratio)
                    )
            msg = (
                "Rebuilt windows "
                f"count={window_count}, "
                f"size={self.glow_window_size}, "
                f"overlap={self.glow_window_overlap}, "
                f"step={step}, "
                f"max_window_entities={self._max_window_entities}."
            )
            if entity_counts:
                msg += (
                    " window_entities_avg="
                    f"{sum(entity_counts) / max(1, len(entity_counts)):.0f}, "
                    f"min={min(entity_counts)}, max={max(entity_counts)}."
                )
            if overlap_ratios:
                msg += (
                    " overlap_ratio_avg="
                    f"{sum(overlap_ratios) / max(1, len(overlap_ratios)):.4f}, "
                    f"min={min(overlap_ratios):.4f}, "
                    f"max={max(overlap_ratios):.4f}."
                )
            self._glow_log(msg)
            if self._glow_debug_verbose:
                limit = self._glow_debug_window_limit
                for idx, entry in enumerate(list(self._glow_windows)):
                    if limit and idx >= limit:
                        break
                    window_key = entry.get("key")
                    stats = None
                    for candidate in window_stats:
                        if candidate[0] == window_key:
                            stats = candidate
                            break
                    if stats is None:
                        unique_count, total_count, overlap_ratio = (
                            self._estimate_window_entity_overlap(window_key)
                        )
                    else:
                        _, unique_count, total_count, overlap_ratio = stats
                    self._glow_log_verbose(
                        f"Window[{idx}] key={window_key} "
                        f"entities_unique={unique_count}, "
                        f"entities_total={total_count}, "
                        f"overlap_ratio={overlap_ratio:.4f}."
                    )
        if (
            (self.glow_overlap_sampling or self.glow_window_work)
            and self._max_window_entities > self._max_partition_entities
        ):
            self._max_partition_entities = self._max_window_entities
            self.config.log(
                "Glow window sizing increased max partition entities to "
                f"{self._max_partition_entities}."
            )

    def _sort_glow_windows(self):
        if not self._glow_windows:
            return
        if not self.glow_bandit_enabled:
            return
        sorted_entries = sorted(
            list(self._glow_windows),
            key=lambda entry: self._window_scores.get(entry["key"], 0.0),
            reverse=True,
        )
        self._glow_windows = deque(sorted_entries)
        if self._glow_debug_verbose:
            limit = self._glow_debug_window_limit or 5
            summary = []
            for entry in list(self._glow_windows)[:limit]:
                key = entry.get("key")
                score = self._window_scores.get(key, 0.0)
                summary.append(f"{key}:{score:.4f}")
            if summary:
                self._glow_log_verbose(
                    "Sorted window scores (top): " + ", ".join(summary)
                )

    def _refill_work(self):
        super(GlowWorkScheduler, self)._refill_work()
        ordered = self._order_by_gradient(list(self.work_to_do))
        if self._glow_debug:
            gradients = (
                self.partition_gradient_stats
                if self.partition_gradient_stats
                else self._latest_gradients
            )
            if not gradients:
                self._glow_log(
                    "No gradient stats available; using original order."
                )
            else:
                source = (
                    "current"
                    if self.partition_gradient_stats
                    else "snapshot"
                )
                scored = []
                for pid in ordered:
                    stats = gradients.get(pid)
                    if not stats:
                        continue
                    count = stats.get("count", 0)
                    if count < self.glow_min_gradient:
                        continue
                    avg = stats.get("sum", 0.0) / max(1, count)
                    scored.append((avg, pid, count))
                scored.sort(reverse=True)
                top = ", ".join(
                    f"{pid}:{avg:.4f}({count})"
                    for avg, pid, count in scored[:5]
                )
                if not top:
                    top = "none"
                self._glow_log(
                    "Gradient ordering applied "
                    f"(source={source}, partitions_with_grad={len(scored)}, "
                    f"min_count={self.glow_min_gradient}, top=[{top}])."
                )
        self.work_to_do = deque(ordered)
        self._rebuild_glow_windows(ordered)
        self._reset_overlap_state()

    def _reset_overlap_state(self):
        self._causal_overlap_queue.clear()
        self._causal_overlap_versions.clear()
        self._causal_overlap_active.clear()

    def _register_window_result(
        self,
        rank,
        step_time,
        window_members,
        window_versions,
        reported_chunk_size=0,
    ):
        conflicts = super(GlowWorkScheduler, self)._register_window_result(
            rank,
            step_time,
            window_members,
            window_versions,
            reported_chunk_size=reported_chunk_size,
        )
        if not self.glow_window_work or window_members is None:
            return conflicts
        try:
            if isinstance(window_members, torch.Tensor):
                members = [int(x) for x in window_members.tolist()]
            else:
                members = [int(x) for x in window_members]
        except Exception:
            return conflicts
        if not members:
            return conflicts
        window_key = tuple(members)
        full_key = self._window_work_key_map.pop(window_key, window_key)
        conflict_flag = bool(conflicts)
        self._glow_log_verbose(
            f"Window result rank={rank}, window={full_key}, "
            f"step_time={step_time:.4f}, conflict={conflict_flag}."
        )
        self._after_partition_result(
            full_key,
            float(step_time),
            conflict=conflict_flag,
            chunk_size=reported_chunk_size or 0,
        )
        return conflicts

    def _cluster_partitions_by_affinity(self, ordered_partitions):
        if not self._co_gradient_scores or len(ordered_partitions) < 2:
            return ordered_partitions
        remaining = set(ordered_partitions)
        sequence = []
        min_shared = (
            self._gradient_cluster_min_shared
            if self._gradient_cluster_enabled
            else 0
        )
        while remaining:
            seed = None
            for pid in ordered_partitions:
                if pid in remaining:
                    seed = pid
                    break
            if seed is None:
                break
            window_group = [seed]
            remaining.remove(seed)
            while (
                len(window_group) < self.glow_window_size
                and remaining
            ):
                best = None
                best_score = float("-inf")
                for candidate in remaining:
                    total = 0.0
                    used = 0
                    for member in window_group:
                        key = tuple(sorted((candidate, member)))
                        if min_shared and self._co_gradient_counts.get(key, 0) < min_shared:
                            continue
                        total += self._co_gradient_scores.get(key, 0.0)
                        used += 1
                    avg_score = total / max(1, used)
                    if avg_score > best_score:
                        best_score = avg_score
                        best = candidate
                if best is None:
                    break
                window_group.append(best)
                remaining.remove(best)
            sequence.extend(window_group)
        if remaining:
            sequence.extend(pid for pid in ordered_partitions if pid in remaining)
        collected = []
        seen = set()
        for pid in sequence:
            if pid not in seen:
                collected.append(pid)
                seen.add(pid)
        for pid in ordered_partitions:
            if pid not in seen:
                collected.append(pid)
        return collected

    def _load_existing_gradients(self):
        snapshot_dir = Path(self.config.folder) / "gradient_snapshots"
        if not snapshot_dir.is_dir():
            return
        snapshots = sorted(snapshot_dir.glob("gradient_snapshot_*.json"))
        if not snapshots:
            final_path = snapshot_dir / "gradient_snapshot_final.json"
            if final_path.is_file():
                snapshots = [final_path]
        if not snapshots:
            return
        latest = snapshots[-1]
        try:
            with open(latest, "r") as fp:
                data = json.load(fp)
            partitions = data.get("partitions", {})
            for pid_str, stats in partitions.items():
                try:
                    pid = int(pid_str)
                except ValueError:
                    continue
                self._latest_gradients[pid] = {
                    "sum": stats.get("sum", 0.0),
                    "count": stats.get("count", 0),
                }
            self.config.log(f"Loaded gradient snapshot from {latest}.")
        except Exception as exc:
            self.config.log(f"Failed to load gradient snapshot {latest}: {exc}")

    def _init_partition_entity_map(self):
        # For stratification, the per-partition entity set is not derived directly
        # from entity_to_partitions; compute it from the actual triples.
        def compute_from_triples(reason):
            triples = self.dataset.split("train")
            mapping = {}
            max_entities = 0
            for partition_id, tensor_ids in enumerate(self.partitions):
                if tensor_ids is None or len(tensor_ids) == 0:
                    mapping[partition_id] = torch.empty((0,), dtype=self.data_type)
                    continue
                triples_subset = triples[tensor_ids.long()]
                entities = torch.unique(triples_subset[:, [0, 2]].reshape(-1))
                entities = entities.to(dtype=self.data_type).contiguous()
                mapping[partition_id] = entities
                max_entities = max(max_entities, int(entities.numel()))
            self._partition_entities_map = mapping
            self._max_partition_entities = max_entities
            self.config.log(
                f"Glow scheduler computed entity sets for {len(mapping)} partitions "
                f"from train triples ({reason})."
            )
            if self._glow_debug and mapping:
                counts = [int(v.numel()) for v in mapping.values()]
                self._glow_log(
                    "Partition entity counts "
                    f"avg={sum(counts) / max(1, len(counts)):.0f}, "
                    f"min={min(counts)}, max={max(counts)}."
                )

        if (
            self._glow_base_partition_type == "stratification"
            or self._glow_stratified_partitions
        ):
            compute_from_triples("stratification")
            return
        try:
            assignments = self.dataset.load_entities_to_partitions(self.num_partitions)
        except Exception as exc:
            self.config.log(
                f"Glow scheduler: could not load entity partition map ({exc}); "
                "computing from train triples."
            )
            compute_from_triples("fallback")
            return
        np_type = TORCH_TO_NP_DTYPE[self.data_type]
        mapping = {}
        for partition in range(self.num_partitions):
            indexes = np.where(assignments == partition)[0]
            if indexes.size == 0:
                mapping[partition] = torch.empty((0,), dtype=self.data_type)
            else:
                mapping[partition] = torch.from_numpy(
                    indexes.astype(np_type)
                ).contiguous()
        self._partition_entities_map = mapping
        self._max_partition_entities = max(
            (len(v) for v in mapping.values()), default=0
        )
        self.config.log(
            f"Glow scheduler loaded entity partition map for {len(mapping)} partitions."
        )
        if self._glow_debug and mapping:
            counts = [int(v.numel()) for v in mapping.values()]
            self._glow_log(
                "Partition entity counts "
                f"avg={sum(counts) / max(1, len(counts)):.0f}, "
                f"min={min(counts)}, max={max(counts)}."
            )

    def _init_partition_relation_map(self):
        if self.config.get("job.distributed.relation_sync_level") != "partition":
            self._partition_relations_map = None
            return
        try:
            assignments = self.dataset.load_relations_to_partitions(self.num_partitions)
        except Exception as exc:
            self.config.log(
                f"Glow scheduler: could not load relation partition map ({exc}); using all relations."
            )
            self._partition_relations_map = None
            return
        np_type = TORCH_TO_NP_DTYPE[self.data_type]
        mapping = {}
        for partition in range(self.num_partitions):
            indexes = np.where(assignments == partition)[0]
            if indexes.size == 0:
                mapping[partition] = torch.empty((0,), dtype=self.data_type)
            else:
                mapping[partition] = torch.from_numpy(
                    indexes.astype(np_type)
                ).contiguous()
        self._partition_relations_map = mapping
        self.config.log(
            f"Glow scheduler loaded relation partition map for {len(mapping)} partitions."
        )
        if self._glow_debug and mapping:
            counts = [int(v.numel()) for v in mapping.values()]
            self._glow_log(
                "Partition relation counts "
                f"avg={sum(counts) / max(1, len(counts)):.0f}, "
                f"min={min(counts)}, max={max(counts)}."
            )

    def _get_max_entities(self):
        if self._max_partition_entities > 0:
            return self._max_partition_entities
        if self._partition_entities_map:
            return max(len(v) for v in self._partition_entities_map.values())
        try:
            assignments = self.dataset.load_entities_to_partitions(self.num_partitions)
        except Exception:
            return 0
        counts = np.bincount(assignments.astype(np.int64), minlength=self.num_partitions)
        if counts.size == 0:
            return 0
        return int(counts.max())

    def _get_partition_relations(self, partition_id):
        if self._partition_relations_map is not None:
            return self._partition_relations_map.get(partition_id)
        if self._all_relations is None:
            self._all_relations = torch.arange(
                self.dataset.num_relations(), dtype=self.data_type
            )
        return self._all_relations

    def _pop_next_partition(self):
        if not self.glow_windows_enabled:
            # Stratification keeps work_to_do as a deque of partition ids.
            if isinstance(self.work_to_do, deque):
                try:
                    return self.work_to_do.pop()
                except IndexError:
                    return None
            # Fallback: dict/ordered mapping of partition ids -> data.
            if isinstance(self.work_to_do, dict):
                try:
                    return self.work_to_do.popitem()[0]
                except KeyError:
                    return None
            return None
        if not self._glow_windows:
            if self._glow_debug and self.work_to_do:
                self._glow_log(
                    "Windows empty but work_to_do still has "
                    f"{len(self.work_to_do)} entries."
                )
            return None
        if self.glow_concurrent_windows:
            while True:
                if self._current_window_entry is None:
                    if not self._glow_windows:
                        return None
                    self._current_window_entry = self._glow_windows.popleft()
                entry = self._current_window_entry
                partitions = entry.get("partitions")
                if partitions is None or len(partitions) == 0:
                    self._current_window_entry = None
                    continue
                pid = partitions.popleft()
                if pid in self._served_partitions:
                    continue
                self._served_partitions.add(pid)
                self._partition_to_window[pid] = entry["key"]
                if not partitions:
                    self._current_window_entry = None
                self._glow_debug_pick_count += 1
                if (
                    self._glow_debug
                    and self._glow_debug_pick_count % self._glow_debug_interval == 0
                ):
                    self._glow_log(
                        f"Selected partition {pid} from window {entry.get('key')} "
                        f"(served={len(self._served_partitions)}/{self.num_partitions})."
                    )
                return pid
        while True:
            if self._current_window_entry is None:
                if not self._glow_windows:
                    return None
                self._current_window_entry = self._glow_windows.popleft()
            entry = self._current_window_entry
            partitions = entry.get("partitions")
            if partitions is None or len(partitions) == 0:
                self._current_window_entry = None
                continue
            pid = partitions.popleft()
            if partitions:
                self._glow_windows.append(entry)
            self._current_window_entry = None
            if pid in self._served_partitions:
                continue
            self._served_partitions.add(pid)
            self._partition_to_window[pid] = entry["key"]
            self._glow_debug_pick_count += 1
            if (
                self._glow_debug
                and self._glow_debug_pick_count % self._glow_debug_interval == 0
            ):
                self._glow_log(
                    f"Selected partition {pid} from window {entry.get('key')} "
                    f"(served={len(self._served_partitions)}/{self.num_partitions})."
                )
            return pid

    def _pop_next_window(self):
        if not self.glow_windows_enabled or not self._glow_windows:
            return None
        while True:
            if not self._glow_windows:
                return None
            entry = self._glow_windows.popleft()
            window_key = entry.get("key")
            if not window_key:
                continue
            invalid = [
                pid
                for pid in window_key
                if not (0 <= int(pid) < len(self.partitions))
            ]
            if invalid and self._glow_debug:
                self._glow_log(
                    "Glow window contains invalid partition ids "
                    f"{invalid}; partitions_len={len(self.partitions)}."
                )
            remaining = [
                pid
                for pid in window_key
                if pid not in self._served_partitions
                and (0 <= int(pid) < len(self.partitions))
            ]
            if not remaining:
                continue
            for pid in remaining:
                self._served_partitions.add(pid)
            if self.glow_window_work:
                remaining_key = tuple(int(pid) for pid in remaining)
                full_key = tuple(int(pid) for pid in window_key)
                self._window_work_key_map[remaining_key] = full_key
            if self._glow_debug:
                self._glow_log(
                    f"Selected window {tuple(remaining)} "
                    f"from key={window_key}."
                )
            return tuple(remaining)

    def _after_partition_result(self, partition_id, avg_time, **kwargs):
        super(GlowWorkScheduler, self)._after_partition_result(
            partition_id, avg_time, **kwargs
        )
        if isinstance(partition_id, (list, tuple)):
            window_key = tuple(int(x) for x in partition_id)
        else:
            window_key = self._partition_to_window.pop(partition_id, None)
        self._update_window_affinity(window_key)
        if (
            not self.glow_bandit_enabled
            or partition_id is None
            or window_key is None
        ):
            return
        chunk_size = kwargs.get("chunk_size") or 0
        if chunk_size <= 0:
            chunk_size = self._adaptive_get_partition_length(partition_id) or 0
        reward = 0.0
        if avg_time and avg_time > 0:
            throughput = chunk_size / max(avg_time, 1e-6)
            reward = throughput * self.glow_reward_scale
        queue_ratio = 0.0
        if kwargs.get("conflict"):
            reward -= self.glow_conflict_penalty
        if self.glow_queue_penalty_scale > 0:
            pending = len(self.work_to_do) if hasattr(self, "work_to_do") else 0
            queue_ratio = pending / max(1, self.num_partitions)
            reward -= self.glow_queue_penalty_scale * queue_ratio
        if self.glow_overlap_penalty > 0 and self._causal_overlap_enabled:
            active = self._causal_overlap_active.get(partition_id, 0)
            if active > 1:
                reward -= self.glow_overlap_penalty * (active - 1)
        prev = self._window_scores.get(window_key, 0.0)
        alpha = self.glow_reward_alpha
        self._window_scores[window_key] = (1.0 - alpha) * prev + alpha * reward
        self._window_counts[window_key] = self._window_counts.get(window_key, 0) + 1
        count = self._window_counts[window_key]
        self._glow_debug_result_count += 1
        if (
            self._glow_debug
            and self._glow_debug_result_count % self._glow_debug_interval == 0
        ):
            self._glow_log(
                "Window result "
                f"window={window_key}, partition={partition_id}, "
                f"avg_time={avg_time:.4f}, chunk_size={chunk_size}, "
                f"reward={reward:.4f}, throughput="
                f"{chunk_size / max(avg_time, 1e-6):.2f}, "
                f"conflict={bool(kwargs.get('conflict'))}, "
                f"queue_ratio={queue_ratio:.3f}."
            )
        if count == 1 or count % 50 == 0:
            self.config.log(
                f"Glow bandit updated window {window_key} to score "
                f"{self._window_scores[window_key]:.4f} "
                f"(reward={reward:.4f}, count={count})."
            )
        self._sort_glow_windows()
        if self.reshape_enabled:
            self._reshape_counter += 1
            if self._reshape_counter >= self.reshape_interval:
                self._reshape_counter = 0
                self._maybe_reshape_partitions()

    def _update_window_affinity(self, window_key):
        if (
            not self.glow_windows_enabled
            or window_key is None
            or len(window_key) < 2
        ):
            return
        self._update_affinity_from_window(window_key)

    def ingest_external_gradient(self, pid_i, pid_j, sum_i, count_i, sum_j, count_j):
        gradients = {
            pid_i: {"sum": sum_i, "count": count_i},
            pid_j: {"sum": sum_j, "count": count_j},
        }
        self._update_affinity_from_window((pid_i, pid_j), custom_stats=gradients)

    def _update_affinity_from_window(self, window_key, custom_stats=None):
        gradients = self.partition_gradient_stats or self._latest_gradients
        if custom_stats:
            gradients = gradients.copy()
            gradients.update(custom_stats)
        values = []
        for pid in window_key:
            stats = gradients.get(pid)
            if not stats or stats["count"] <= 0:
                avg = 0.0
            else:
                avg = stats["sum"] / max(1, stats["count"])
            values.append((pid, abs(avg)))
        if not values:
            return
        alpha = self.glow_affinity_alpha
        for (pid_i, val_i), (pid_j, val_j) in itertools.combinations(values, 2):
            pair = tuple(sorted((pid_i, pid_j)))
            denom = val_i + val_j
            if denom <= 0:
                affinity = 0.0
            else:
                affinity = 1.0 - abs(val_i - val_j) / denom
            prev = self._co_gradient_scores.get(pair, 0.0)
            self._co_gradient_scores[pair] = (1.0 - alpha) * prev + alpha * affinity
            self._co_gradient_counts[pair] = self._co_gradient_counts.get(pair, 0) + 1

    def _register_partition_relation_gradient(
        self, partition_id, relation_ids, grad_sums, grad_counts
    ):
        super(GlowWorkScheduler, self)._register_partition_relation_gradient(
            partition_id, relation_ids, grad_sums, grad_counts
        )
        if not self._gradient_cluster_enabled or partition_id is None:
            return
        self._update_relation_affinity(
            int(partition_id), relation_ids, grad_sums, grad_counts
        )

    def _update_relation_affinity(
        self, partition_id: int, relation_ids, grad_sums, grad_counts
    ):
        if relation_ids is None:
            return
        rel_ids = relation_ids
        rel_sums = grad_sums
        rel_counts = grad_counts
        if not isinstance(rel_ids, torch.Tensor):
            rel_ids = torch.as_tensor(rel_ids, dtype=torch.long)
        if not isinstance(rel_sums, torch.Tensor):
            rel_sums = torch.as_tensor(rel_sums, dtype=torch.float32)
        if not isinstance(rel_counts, torch.Tensor):
            rel_counts = torch.as_tensor(rel_counts, dtype=torch.float32)
        rel_ids = rel_ids.to(device="cpu")
        rel_sums = rel_sums.to(device="cpu", dtype=torch.float32)
        rel_counts = rel_counts.to(device="cpu", dtype=torch.float32)
        if rel_ids.numel() == 0:
            return
        rel_counts = torch.clamp(rel_counts, min=1.0)
        rel_avg = rel_sums / rel_counts
        top_k = min(self._gradient_cluster_top_relations, rel_avg.numel())
        vals, idx = torch.topk(rel_avg, k=top_k, largest=True)
        top_rel_ids = rel_ids[idx].tolist()
        top_rel_vals = vals.tolist()
        previous = self._partition_relation_topk.get(partition_id)
        if previous:
            for rel in previous:
                rel_map = self._relation_to_partitions.get(rel)
                if rel_map is not None:
                    rel_map.pop(partition_id, None)
                    if not rel_map:
                        self._relation_to_partitions.pop(rel, None)
        self._partition_relation_topk[partition_id] = top_rel_ids
        alpha = self._gradient_cluster_affinity_alpha
        for rel_id, rel_val in zip(top_rel_ids, top_rel_vals):
            rel_map = self._relation_to_partitions.setdefault(rel_id, {})
            rel_map[partition_id] = rel_val
            if (
                len(rel_map) > self._gradient_cluster_max_partitions_per_relation
            ):
                drop = min(rel_map.items(), key=lambda item: item[1])
                rel_map.pop(drop[0], None)
            for other_pid, other_val in rel_map.items():
                if other_pid == partition_id:
                    continue
                denom = rel_val + other_val
                if denom <= 0:
                    affinity = 0.0
                else:
                    affinity = 1.0 - abs(rel_val - other_val) / denom
                pair = tuple(sorted((partition_id, other_pid)))
                prev = self._co_gradient_scores.get(pair, 0.0)
                self._co_gradient_scores[pair] = (
                    (1.0 - alpha) * prev + alpha * affinity
                )
                self._co_gradient_counts[pair] = (
                    self._co_gradient_counts.get(pair, 0) + 1
                )

    def _get_window_entities(self, window_key):
        if (
            window_key is None
            or not self.glow_windows_enabled
            or self._partition_entities_map is None
        ):
            return None
        tensors = []
        for pid in window_key:
            entries = self._partition_entities_map.get(pid)
            if entries is None or entries.numel() == 0:
                continue
            tensors.append(entries)
        if not tensors:
            return None
        if len(tensors) == 1:
            return tensors[0]
        result = torch.unique(torch.cat(tensors))
        if self._glow_debug_verbose:
            self._glow_log_verbose(
                f"Window entities key={tuple(window_key)} "
                f"count={int(result.numel())}."
            )
        return result

    def _get_window_relations(self, window_key):
        if (
            window_key is None
            or self.config.get("job.distributed.relation_sync_level")
            != "partition"
        ):
            return None
        if self._partition_relations_map is None:
            return self._get_partition_relations(window_key[0])
        tensors = []
        for pid in window_key:
            rels = self._partition_relations_map.get(pid)
            if rels is None or rels.numel() == 0:
                continue
            tensors.append(rels)
        if not tensors:
            return None
        if len(tensors) == 1:
            return tensors[0]
        return torch.unique(torch.cat(tensors))

    def _estimate_window_entity_count(self, window_key):
        if self._partition_entities_map is None or not window_key:
            return 0
        tensors = []
        for pid in window_key:
            entries = self._partition_entities_map.get(pid)
            if entries is None or entries.numel() == 0:
                continue
            tensors.append(entries)
        if not tensors:
            return 0
        if len(tensors) == 1:
            return int(tensors[0].numel())
        return int(torch.unique(torch.cat(tensors)).numel())

    def _estimate_window_entity_overlap(self, window_key):
        if self._partition_entities_map is None or not window_key:
            return 0, 0, 0.0
        tensors = []
        total_count = 0
        for pid in window_key:
            entries = self._partition_entities_map.get(pid)
            if entries is None or entries.numel() == 0:
                continue
            total_count += int(entries.numel())
            tensors.append(entries)
        if not tensors:
            return 0, 0, 0.0
        if len(tensors) == 1:
            unique_count = int(tensors[0].numel())
        else:
            unique_count = int(torch.unique(torch.cat(tensors)).numel())
        if total_count <= 0:
            return unique_count, total_count, 0.0
        overlap_ratio = max(0.0, 1.0 - unique_count / total_count)
        return unique_count, total_count, overlap_ratio

    def _maybe_reshape_partitions(self):
        if not self.reshape_enabled:
            return
        if self.reshape_hot_splits <= 0 or self.reshape_cold_merges <= 0:
            return
        served_count = len(self._served_partitions)
        mid_epoch = 0 < served_count < self.num_partitions
        epoch_complete = served_count >= self.num_partitions
        gradients = self.partition_gradient_stats or self._latest_gradients
        if not gradients:
            return
        stats = []
        for pid in range(len(self.partitions)):
            g = gradients.get(pid)
            if not g or g["count"] <= 0:
                avg = 0.0
            else:
                avg = g["sum"] / max(1, g["count"])
            stats.append((avg, pid))
        if len(stats) < 2:
            return
        stats.sort(reverse=True)
        hot_candidates = [
            pid
            for _, pid in stats
            if len(self.partitions[pid]) >= 2 * self.reshape_min_size
        ]
        cold_candidates = [
            pid
            for _, pid in sorted(stats, key=lambda item: item[0])
            if pid not in hot_candidates[: self.reshape_hot_splits]
        ]
        hot_ids = hot_candidates[: self.reshape_hot_splits]
        cold_ids = cold_candidates[: self.reshape_cold_merges]
        changes = []
        for hot_id, cold_id in zip(hot_ids, cold_ids):
            if hot_id == cold_id:
                continue
            moved = self._move_triples_from_hot_to_cold(hot_id, cold_id)
            if moved > 0:
                changes.append((hot_id, cold_id, moved))
        if changes:
            self.config.log(
                f"Glow reshape adjusted {len(changes)} partitions: "
                + ", ".join(
                    [f"{hot}->{cold} ({moved} triples)" for hot, cold, moved in changes]
                )
            )
            if epoch_complete:
                if self._glow_debug:
                    self._glow_log(
                        "Glow reshape changes will apply next epoch."
                    )
            else:
                self._rebuild_glow_windows(
                    list(range(self.num_partitions)),
                    preserve_served=mid_epoch,
                )
        moved = self._maybe_repartition_from_gradient_graph()
        if moved > 0:
            self.config.log(
                f"Glow gradient repartition moved {moved} triples across partitions."
            )
            if epoch_complete:
                if self._glow_debug:
                    self._glow_log(
                        "Glow gradient repartition will apply next epoch."
                    )
            else:
                self._rebuild_glow_windows(
                    list(range(self.num_partitions)),
                    preserve_served=mid_epoch,
                )

    def _move_triples_from_hot_to_cold(self, hot_id, cold_id):
        hot_tensor = self.partitions[hot_id]
        cold_tensor = self.partitions[cold_id]
        if hot_tensor is None or cold_tensor is None:
            return 0
        hot_len = len(hot_tensor)
        if hot_len < 2 * self.reshape_min_size:
            return 0
        split_point = max(self.reshape_min_size, hot_len // 2)
        if split_point >= hot_len:
            return 0
        keep = hot_tensor[:split_point].clone()
        moved = hot_tensor[split_point:].clone()
        if moved.numel() == 0:
            return 0
        self.partitions[hot_id] = keep.contiguous()
        self.partitions[cold_id] = torch.cat(
            [cold_tensor, moved], dim=0
        ).contiguous()
        self.partition_gradient_stats[hot_id] = {"sum": 0.0, "count": 0}
        self.partition_gradient_stats[cold_id] = {"sum": 0.0, "count": 0}
        if self._partition_entities_map is not None:
            self._recompute_partition_entities(hot_id)
            self._recompute_partition_entities(cold_id)
        return moved.numel()

    def _recompute_partition_entities(self, partition_id):
        if self._partition_entities_map is None:
            return
        triples = self.dataset.split("train")
        tensor_ids = self.partitions[partition_id]
        if tensor_ids is None or len(tensor_ids) == 0:
            self._partition_entities_map[partition_id] = torch.empty(
                (0,), dtype=self.data_type
            )
            return
        triples_subset = triples[tensor_ids.long()]
        entities = torch.unique(triples_subset[:, [0, 2]].reshape(-1))
        self._partition_entities_map[partition_id] = (
            entities.to(dtype=self.data_type).contiguous().cpu()
        )

    def _recompute_partition_relations(self, partition_id):
        if self._partition_relations_map is None:
            return
        triples = self._get_train_triples()
        tensor_ids = self.partitions[partition_id]
        if tensor_ids is None or len(tensor_ids) == 0:
            self._partition_relations_map[partition_id] = torch.empty(
                (0,), dtype=self.data_type
            )
            return
        triples_subset = triples[tensor_ids.long()]
        rels = torch.unique(triples_subset[:, 1].reshape(-1))
        self._partition_relations_map[partition_id] = (
            rels.to(dtype=self.data_type).contiguous().cpu()
        )

    def _get_train_triples(self):
        if self._train_triples_cache is None:
            self._train_triples_cache = self.dataset.split("train")
        return self._train_triples_cache

    def _maybe_repartition_from_gradient_graph(self):
        if (
            not self._gradient_repartition_enabled
            or self._gradient_repartition_max_moves <= 0
            or not self._gradient_graph_enabled
            or self._gradient_graph_relations is None
        ):
            return 0
        if not self._gradient_graph_relations:
            self.config.log(
                "Glow gradient repartition skipped: no relation stats available."
            )
            return 0
        num_relations = self.dataset.num_relations()
        if num_relations is None or num_relations <= 0:
            self.config.log(
                "Glow gradient repartition skipped: no relations in dataset."
            )
            return 0
        self.config.log(
            "Glow gradient repartition evaluating graph at snapshot "
            f"{self._gradient_updates} "
            f"(relations={len(self._gradient_graph_relations)}, "
            f"max_moves={self._gradient_repartition_max_moves}, "
            f"max_relations={self._gradient_repartition_max_relations})."
        )
        relation_owner = torch.full(
            (num_relations,), -1, dtype=torch.long
        )
        relation_score = torch.zeros((num_relations,), dtype=torch.float32)
        for rel_id, part_map in self._gradient_graph_relations.items():
            if not part_map:
                continue
            best_pid = None
            best_score = -1.0
            for pid, stats in part_map.items():
                score = stats.get("avg", 0.0)
                if score > best_score:
                    best_score = score
                    best_pid = pid
            if best_pid is None:
                continue
            rid = int(rel_id)
            if 0 <= rid < num_relations:
                relation_owner[rid] = int(best_pid)
                relation_score[rid] = float(best_score)
        if self._gradient_repartition_max_relations > 0:
            max_rel = min(self._gradient_repartition_max_relations, num_relations)
            if max_rel > 0:
                _, idx = torch.topk(
                    relation_score, k=max_rel, largest=True
                )
                mask = torch.zeros_like(relation_owner, dtype=torch.bool)
                mask[idx] = True
                relation_owner = torch.where(
                    mask, relation_owner, torch.full_like(relation_owner, -1)
                )
        triples = self._get_train_triples()
        moved_total = 0
        changed_partitions = set()
        max_size = self._gradient_repartition_max_size
        min_size = self._gradient_repartition_min_size
        for pid, part in enumerate(self.partitions):
            if part is None or len(part) == 0:
                continue
            if len(part) <= min_size:
                continue
            rels = triples[part.long(), 1]
            if not isinstance(rels, torch.Tensor):
                rels = torch.as_tensor(rels)
            if rels.device.type != "cpu":
                rels = rels.cpu()
            rels = rels.long()
            targets = relation_owner[rels]
            move_mask = (targets >= 0) & (targets != pid)
            if not move_mask.any():
                continue
            move_idx = move_mask.nonzero(as_tuple=False).view(-1)
            if move_idx.numel() > self._gradient_repartition_max_moves:
                move_idx = move_idx[: self._gradient_repartition_max_moves]
            if move_idx.numel() == 0:
                continue
            move_targets = targets[move_idx]
            keep_mask = torch.ones(len(part), dtype=torch.bool)
            keep_mask[move_idx] = False
            keep_part = part[keep_mask].contiguous()
            if len(keep_part) < min_size:
                continue
            moved_any = False
            for target_pid in torch.unique(move_targets).tolist():
                target_pid = int(target_pid)
                if target_pid < 0 or target_pid >= len(self.partitions):
                    continue
                target_part = self.partitions[target_pid]
                if target_part is None:
                    target_part = torch.empty((0,), dtype=part.dtype)
                target_mask = move_targets == target_pid
                move_local = move_idx[target_mask]
                if move_local.numel() == 0:
                    continue
                if max_size > 0 and len(target_part) + len(move_local) > max_size:
                    allowed = max_size - len(target_part)
                    if allowed <= 0:
                        continue
                    move_local = move_local[:allowed]
                if move_local.numel() == 0:
                    continue
                moved_triples = part[move_local].contiguous()
                self.partitions[target_pid] = torch.cat(
                    [target_part, moved_triples], dim=0
                ).contiguous()
                moved_total += moved_triples.numel()
                moved_any = True
                changed_partitions.add(target_pid)
            if moved_any:
                self.partitions[pid] = keep_part
                changed_partitions.add(pid)
        for pid in changed_partitions:
            self.partition_gradient_stats[pid] = {"sum": 0.0, "count": 0}
            if pid in self.partition_relation_gradient_stats:
                self.partition_relation_gradient_stats.pop(pid, None)
            if self._gradient_graph_partitions is not None:
                rels = self._gradient_graph_partitions.pop(pid, None)
                if rels and self._gradient_graph_relations is not None:
                    for rel_id in list(rels.keys()):
                        rel_map = self._gradient_graph_relations.get(rel_id)
                        if rel_map is None:
                            continue
                        rel_map.pop(pid, None)
                        if not rel_map:
                            self._gradient_graph_relations.pop(rel_id, None)
            if pid in self.partition_committed_versions:
                self.partition_committed_versions[pid] = -1
            self.partition_issue_versions[pid] += 1
            if self._partition_entities_map is not None:
                self._recompute_partition_entities(pid)
            if self._partition_relations_map is not None:
                self._recompute_partition_relations(pid)
        if moved_total > 0 and self._gradient_graph_cluster_enabled:
            self._maybe_update_gradient_graph_clusters(force=True)
        if moved_total == 0:
            self.config.log(
                "Glow gradient repartition found no moves to apply."
            )
        return moved_total

    def _next_work(self, rank, machine_id) -> WorkPackage:
        def build_work(partition_id, reuse_version=False, version=None):
            partition_tensor = self.partitions[partition_id]
            if partition_tensor is None or len(partition_tensor) == 0:
                return None
            partition_slice = self._adaptive_maybe_slice(
                partition_id, partition_tensor
            )
            if partition_slice is None or len(partition_slice) == 0:
                return None
            work_package = WorkPackage()
            work_package.partition_id = partition_id
            work_package.partition_data = partition_slice
            work_package.reuse_partition_version = reuse_version
            if reuse_version and version is not None:
                work_package.partition_version = version
            entities = None
            if self._partition_entities_map is not None:
                entities = self._partition_entities_map.get(partition_id)
            if entities is None:
                entities = self.local_entities.get(rank)
            work_package.entities_in_partition = entities
            if (
                self.config.get("job.distributed.relation_sync_level")
                == "partition"
            ):
                work_package.relations_in_partition = self._get_partition_relations(
                    partition_id
                )
            window_key = self._partition_to_window.get(partition_id)
            if window_key is not None:
                work_package.window_members = list(window_key)
                if self._send_window_entities:
                    window_entities = self._get_window_entities(window_key)
                    if window_entities is not None:
                        work_package.window_entities = window_entities
                        if self._glow_debug_verbose:
                            self._glow_log_verbose(
                                f"Attach window_entities size="
                                f"{int(window_entities.numel())} "
                                f"for window={window_key} "
                                f"partition={partition_id}."
                            )
            return work_package

        def build_window_work(window_members):
            window_members = [int(pid) for pid in window_members]
            partition_slices = []
            window_versions = []
            sizes = None
            if self._glow_debug_verbose:
                sizes = []
            for pid in window_members:
                if pid < 0 or pid >= len(self.partitions):
                    if self._glow_debug:
                        self._glow_log(
                            "Window work member pid "
                            f"{pid} out of range; "
                            f"partitions_len={len(self.partitions)}."
                        )
                    if sizes is not None:
                        sizes.append(0)
                    continue
                partition_tensor = self.partitions[pid]
                if sizes is not None:
                    sizes.append(
                        0 if partition_tensor is None else int(len(partition_tensor))
                    )
                if partition_tensor is None or len(partition_tensor) == 0:
                    continue
                partition_slices.append(partition_tensor)
                version = self.partition_issue_versions[pid]
                self.partition_issue_versions[pid] = version + 1
                window_versions.append(int(version))
            if sizes is not None:
                self._glow_log_verbose(
                    f"Window work build {tuple(window_members)} sizes={sizes}."
                )
            if not partition_slices:
                if self._glow_debug:
                    self._glow_log(
                        f"Window work build {tuple(window_members)} "
                        "produced empty partition_data."
                    )
                return None
            work_package = WorkPackage()
            work_package.partition_id = tuple(window_members)
            work_package.partition_data = torch.cat(partition_slices).contiguous()
            if self._glow_debug and work_package.partition_data.numel() == 0:
                self._glow_log(
                    f"Window work build {tuple(window_members)} "
                    "produced 0 triples after concat."
                )
            if self.config.get("job.distributed.shuffle_partition_samples"):
                # Use long indices for safe tensor indexing.
                perm = torch.randperm(
                    work_package.partition_data.numel(),
                    device=work_package.partition_data.device,
                )
                work_package.partition_data = work_package.partition_data[perm]
            work_package.window_members = list(window_members)
            work_package.window_versions = window_versions
            entities = None
            if self._partition_entities_map is not None:
                entities = self._get_window_entities(tuple(window_members))
            if entities is None:
                entities = self.local_entities.get(rank)
            work_package.entities_in_partition = entities
            if (
                self.config.get("job.distributed.relation_sync_level")
                == "partition"
            ):
                work_package.relations_in_partition = self._get_window_relations(
                    tuple(window_members)
                )
            if self._send_window_entities:
                window_entities = self._get_window_entities(tuple(window_members))
                if window_entities is not None:
                    work_package.window_entities = window_entities
                    if self._glow_debug_verbose:
                        self._glow_log_verbose(
                            f"Attach window_entities size="
                            f"{int(window_entities.numel())} "
                            f"for window={tuple(window_members)}."
                        )
            return work_package

        try:
            while True:
                if self.glow_window_work:
                    window_members = self._pop_next_window()
                    if window_members is None:
                        if self.active_partition_per_worker:
                            if self._glow_debug:
                                self._glow_log(
                                    "Window work queue empty; active partitions "
                                    f"remain, returning WAIT to rank {rank}."
                                )
                            work_package = WorkPackage()
                            work_package.wait = True
                            return work_package
                        if self._glow_debug:
                            self._glow_log(
                                "Window work queue empty; returning NO_WORK "
                                f"to rank {rank}."
                            )
                        return WorkPackage()
                    work_package = build_window_work(window_members)
                    if work_package is None:
                        if self._glow_debug:
                            self._glow_log(
                                f"Window work build {tuple(window_members)} "
                                "returned None."
                            )
                        continue
                    if self._glow_debug:
                        data = work_package.partition_data
                        if data is None:
                            data_info = "None"
                        else:
                            data_info = (
                                f"numel={int(data.numel())} "
                                f"shape={tuple(data.shape)}"
                            )
                        self._glow_log(
                            f"Built window work {tuple(window_members)} "
                            f"partition_data={data_info}."
                        )
                    if self._glow_debug:
                        size = (
                            int(work_package.partition_data.numel())
                            if work_package.partition_data is not None
                            else -1
                        )
                        self._glow_log(
                            f"Dispatching window work {tuple(window_members)} "
                            f"size={size} to rank {rank}."
                        )
                    return work_package
                if (
                    self._causal_overlap_enabled
                    and self._causal_overlap_duplicate
                    and self._causal_overlap_queue
                ):
                    partition_id = self._causal_overlap_queue.popleft()
                    version = self._causal_overlap_versions.get(partition_id)
                    if version is None:
                        continue
                    if (
                        self._causal_overlap_active.get(partition_id, 0)
                        >= self._causal_overlap_max_workers
                    ):
                        continue
                    work_package = build_work(
                        partition_id, reuse_version=True, version=version
                    )
                    if work_package is None:
                        continue
                    self._causal_overlap_active[partition_id] += 1
                    return work_package
                partition_id = self._pop_next_partition()
                if partition_id is None:
                    if (
                        self.glow_windows_enabled
                        and len(self._served_partitions) < self.num_partitions
                    ):
                        self._rebuild_glow_windows(
                            list(range(self.num_partitions)),
                            preserve_served=True,
                        )
                        partition_id = self._pop_next_partition()
                    if partition_id is None and self.active_partition_per_worker:
                        work_package = WorkPackage()
                        work_package.wait = True
                        return work_package
                    if partition_id is None:
                        return WorkPackage()
                if partition_id < 0 or partition_id >= len(self.partitions):
                    continue
                work_package = build_work(partition_id)
                if work_package is None:
                    continue
                if (
                    self._causal_overlap_enabled
                    and self._causal_overlap_max_workers > 1
                    and self._causal_overlap_duplicate
                    and partition_id not in self._causal_overlap_versions
                ):
                    self._causal_overlap_versions[partition_id] = (
                        self.partition_issue_versions[partition_id]
                    )
                    for _ in range(self._causal_overlap_max_workers - 1):
                        self._causal_overlap_queue.append(partition_id)
                if self._causal_overlap_enabled:
                    self._causal_overlap_active[partition_id] += 1
                return work_package
        except IndexError:
            if self._glow_debug:
                self._glow_log(
                    "IndexError while selecting work; returning NO_WORK."
                )
            return WorkPackage()

    def _handle_work_done(self, rank):
        if self._causal_overlap_enabled:
            partition_id = self.active_partition_per_worker.get(rank)
            if partition_id is not None:
                current = self._causal_overlap_active.get(partition_id, 0)
                if current <= 1:
                    self._causal_overlap_active.pop(partition_id, None)
                else:
                    self._causal_overlap_active[partition_id] = current - 1
        super(GlowWorkScheduler, self)._handle_work_done(rank)

class StratificationWorkScheduler(AdaptiveWorkScheduler):

    def __init__(
        self,
        config,
        dataset,
    ):
        dataset._partition_type = "stratification"
        self.combine_mirror_blocks = config.get("job.distributed.stratification.combine_mirror")
        cover_cfg = config.get("job.distributed.stratification.cover") or {}
        self._cover_enabled = bool(cover_cfg.get("enable", False))
        self._cover_q = max(2, int(cover_cfg.get("q", 4)))
        self._cover_log_groups = bool(cover_cfg.get("log_groups", False))
        super(StratificationWorkScheduler, self).__init__(
            config=config,
            dataset=dataset,
        )
        self.schedule_creator = None
        self.fixed_schedule = []
        self.current_iteration = set()
        self._cover_groups = []
        self._cover_group_index = 0
        self._cover_pending_states = deque()
        self._cover_active_states = {}
        if not self._cover_enabled:
            self.schedule_creator = StratificationScheduleCreator(
                num_partitions=self.num_partitions,
                num_workers=self.num_clients,
                randomize_iterations=True,
                combine_mirror_blocks=self.combine_mirror_blocks,
            )
            self.fixed_schedule = self.schedule_creator.create_schedule()
        self._pre_localized_strata: Dict[int, Tuple[int, int]] = {}
        # dictionary: key=worker_rank, value=block
        self.running_blocks: Dict[int, Tuple[int, int]] = {}
        self.active_only = self.config.get(
            "job.distributed.stratification.active_only"
        )
        self.num_max_entities = 0

    def _init_in_started_process(self):
        super(StratificationWorkScheduler, self)._init_in_started_process()
        # self.work_to_do = deepcopy(self.partitions)
        self._initialized_entity_blocks = set()
        entities_to_partition = self.dataset.load_entities_to_partitions(self.num_partitions)
        self._entities_in_strata = self._get_entities_in_strata(
            entities_to_partition,
            self.partitions,
            self.dataset.split("train").numpy(),
            self.active_only,
            self.combine_mirror_blocks,
            TORCH_TO_NP_DTYPE[self.data_type]
        )
        if self._cover_enabled:
            self._init_cover_schedule()
        else:
            # Always build the scheduled work queue for stratification runs
            # (even when a fixed schedule was precomputed in __init__).
            ordered = self._order_by_schedule(deepcopy(self.partitions))
            # Store only the partition ids in a deque; actual data stays in self.partitions.
            self.work_to_do = deque(ordered.keys())

    @staticmethod
    @numba.guvectorize(
        [(numba.int64[:], numba.int64, numba.int64, numba.int64[:])],
        "(n),(),()->(n)",
        nopython=True
    )
    def _get_partition(entity_ids, num_entities, num_partitions, res):
        """
        This method maps a (already mapped) entity id to it's entity_partition.
        NOTE: you cannot provide named parameters (kwargs) to this function
        Args:
            entity_ids: (mapped) entity ids np.array()
            num_entities: dataset.num_entities()
            num_partitions: int
            res: DON'T PROVIDE THIS. This is the resulting np.array of this vectorized
                function.

        Returns: np.array of entity ids mapped to partition

        """
        for i in range(len(entity_ids)):
            res[i] = math.floor(
                entity_ids[i] * 1.0 / num_entities * 1.0 * num_partitions
            )

    @staticmethod
    def _repartition(
        data,
        num_entities,
        num_partitions,
        active_only=True,
        combine_mirror_blocks=True,
        np_type=np.int64,
    ):
        """
        This needs to be a static method so that we can pickle and run in background
        Args:
            data: data to repartition (train-set)
            num_entities: dataset.num_entities()
            num_partitions: self.num_partitions

        Returns:
            partitions: dict of structure {(block_id 1, block_id 2): [triple ids]}
            entities_in_strata:
                dict of structure {(block_id 1, block_id 2): list of entity ids}
        """
        print("repartitioning data")
        start = -time.time()

        def random_map_entities():
            mapper = np.random.permutation(num_entities)
            mapped_data = deepcopy(data)  # drop reference to dataset
            mapped_data = mapped_data.numpy()
            mapped_data[:, 0] = mapper[mapped_data[:, 0]]
            mapped_data[:, 2] = mapper[mapped_data[:, 2]]
            return mapped_data, mapper

        mapped_data, mapped_entities = random_map_entities()
        print("repartition s")
        s_block = StratificationWorkScheduler._get_partition(
            mapped_data[:, 0], num_entities, num_partitions,
        )
        print("repartition o")
        o_block = StratificationWorkScheduler._get_partition(
            mapped_data[:, 2], num_entities, num_partitions,
        )
        print("map entity ids to partition")
        entity_to_partition = StratificationWorkScheduler._get_partition(
            mapped_entities, num_entities, num_partitions,
        )
        triple_partition_assignment = np.stack([s_block, o_block], axis=1)
        partitions = StratificationWorkScheduler._construct_partitions(
            triple_partition_assignment, num_partitions, np_type=np_type
        )
        entities_in_strata = StratificationWorkScheduler._get_entities_in_strata(
            entity_to_partition,
            partitions,
            data.numpy(),
            active_only,
            combine_mirror_blocks,
            np_type
        )
        print("repartitioning done")
        print("repartition_time", start + time.time())
        return partitions, entities_in_strata

    @staticmethod
    def _get_entities_in_strata(
        entities_to_partition,
        partitions,
        data,
        active_only,
        combine_mirror_blocks,
        np_type,
    ):
        entities_in_strata = dict()
        if active_only:
            for strata, strata_data in partitions.items():
                if combine_mirror_blocks:
                    if strata in entities_in_strata:
                        continue
                    if strata[0] == strata[1]:
                        if strata[0] % 2 == 0:
                            continue
                        mirror_strata = (strata[0] - 1, strata[1] - 1)
                    else:
                        mirror_strata = (strata[1], strata[0])
                    mirror_data = partitions[mirror_strata]
                    # for some reason, torch.cat hangs on some machines on larger
                    # datasets when run in background, use numpy instead
                    combined_strata_data = np.concatenate((strata_data, mirror_data))
                    unique_entities = torch.from_numpy(
                        np.unique(data[combined_strata_data][:, [0, 2]]).astype(np_type)
                    ).contiguous()
                    entities_in_strata[strata] = unique_entities
                    entities_in_strata[mirror_strata] = unique_entities
                else:
                    # np.unique is slightly faster than torch.unique
                    entities_in_strata[strata] = torch.from_numpy(
                        np.unique(data[strata_data][:, [0, 2]]).astype(np_type)
                    ).contiguous()
        else:
            for strata in partitions.keys():
                if strata in entities_in_strata:
                    continue
                mirror_strata = (strata[1], strata[0])
                if combine_mirror_blocks:
                    if strata[0] == strata[1]:
                        if strata[0] % 2 == 0:
                            continue
                        mirror_strata = (strata[0] - 1, strata[1] - 1)
                entities = torch.from_numpy(
                    np.where(
                        np.ma.mask_or(
                            (entities_to_partition == strata[0]),
                            (entities_to_partition == mirror_strata[0]),
                        )
                    )[0].astype(np_type)
                ).contiguous()
                entities_in_strata[strata] = entities
                entities_in_strata[mirror_strata] = entities
        return entities_in_strata

    def _get_max_entities(self):
        if self.num_max_entities > 0:
            # store the result so that we don't have to recompute for every trainer
            return self.num_max_entities
        if self.active_only:
            num_entities_in_strata = [len(i) for i in self._entities_in_strata.values()]
            len_std = np.std(num_entities_in_strata).item()
            if self.combine_mirror_blocks:
                max_num_entities, std_num_entities = self._get_mirrored_max_entities(
                    self.num_partitions,
                    list(self._entities_in_strata.values()),
                    return_std=True,
                )
                self.num_max_entities = max_num_entities + 2 * (round(std_num_entities))
            else:
                self.num_max_entities = max(num_entities_in_strata) + 5 * round(len_std)
        else:
            self.num_max_entities = max(
                [len(i) for i in self._entities_in_strata.values()]
            )
        return self.num_max_entities

    @staticmethod
    def _get_mirrored_max_entities(num_partitions, strata_entities, return_std=False):
        """
        Calculate how many entities occur at most if we combine mirrored blocks
        Combining blocks (0,1) and (1,0)
        For diagonals combine (0,0),(1,1), then (2,2),(3,3)...
        Count unique entities per combined block and return max
        Args:
            num_partitions: number of partitions
            strata_entities: list of unique entities occurring per strata
                assumes list is ordered

        Returns: max number of entities occurring in a combined mirror block

        """
        max_value = 0
        all_num_entities = []
        for i in range(num_partitions):
            for j in range(i, num_partitions):
                num_entities = 0
                # combine mirrored blocks
                if i % 2 == 0 and i == j:
                    # diagonal blocks: combine with following diagonal
                    concat_entities = np.concatenate(
                        (strata_entities[i], strata_entities[i + num_partitions])
                    )
                    num_entities = len(np.unique(concat_entities))
                elif i != j:
                    # combine (0,1) with (1,0) and so on
                    num_entities = len(
                        np.unique(
                            np.concatenate(
                                (
                                    strata_entities[i * num_partitions + j],
                                    strata_entities[j * num_partitions + i],
                                )
                            )
                        )
                    )
                all_num_entities.append(num_entities)
                if num_entities > max_value:
                    # this will lead to a race condition if we do this in parallel
                    max_value = num_entities
        all_num_entities = np.array(all_num_entities)
        max_value = all_num_entities.max()
        if return_std:
            std = all_num_entities.std()
            return max_value, std
        print("max entities", max_value)
        return max_value

    def _next_work(
        self, rank, machine_id
    ) -> WorkPackage:
        return self._acquire_strata(rank, machine_id)

    def _handle_pre_localize_work(self, rank, machine_id):
        return self._acquire_strata(rank, machine_id, pre_localize=True)

    def _acquire_strata(self, rank, machine_id, pre_localize=False):
        if self._cover_enabled:
            return self._acquire_strata_cover(rank)
        try:
            if len(self.current_iteration) == 0:
                self.current_iteration = set(self.fixed_schedule.pop())
        except IndexError:
            return WorkPackage()
        return self._acquire_strata_by_schedule(
            rank, current_iteration=self.current_iteration, pre_localize=pre_localize
        )

    def _acquire_strata_by_schedule(self, rank, current_iteration, pre_localize=False):
        work_package = WorkPackage()
        try:
            locked_entity_strata = set()
            for locked_dict in [self.running_blocks, self._pre_localized_strata]:
                for running_rank, strata in locked_dict.items():
                    if rank == running_rank:
                        continue
                    locked_entity_strata.add(strata[0])
                    locked_entity_strata.add(strata[1])

            def _strata_locked(strata):
                return (
                    strata[0] in locked_entity_strata
                    or strata[1] in locked_entity_strata
                )

            def _acquire(strata, acquire_pre_localized=False):
                if acquire_pre_localized:
                    del self._pre_localized_strata[rank]
                else:
                    current_iteration.remove(strata)
                strata_data = self.partitions[strata]
                entities_in_strata = self._entities_in_strata.get(strata)
                if self.combine_mirror_blocks and strata_data is not None:
                    if strata[0] == strata[1]:
                        mirror_strata = (strata[0] - 1, strata[1] - 1)
                    else:
                        mirror_strata = (strata[1], strata[0])
                    strata_data = torch.cat(
                        (strata_data, self.partitions[mirror_strata])
                    )
                if not pre_localize:
                    self.running_blocks[rank] = strata
                else:
                    self._pre_localized_strata[rank] = strata
                work_package.partition_id = strata
                partition_slice = self._adaptive_maybe_slice(strata, strata_data)
                if partition_slice is None or len(partition_slice) == 0:
                    work_package.partition_data = strata_data
                else:
                    work_package.partition_data = partition_slice
                work_package.entities_in_partition = entities_in_strata
                return work_package

            # only use pre localized strata, if we are not about to pre-localize a new
            # one --> not pre_localize
            if (
                not pre_localize
                and self._pre_localized_strata.get(rank, None) is not None
            ):
                strata = self._pre_localized_strata[rank]
                if _strata_locked(strata):
                    # we are waiting until the localized strata is free
                    work_package.wait = True
                    return work_package
                return _acquire(strata, acquire_pre_localized=True)

            for strata in current_iteration:
                if _strata_locked(strata):
                    continue
                return _acquire(strata)

            # return wait here
            work_package.wait = True
            return work_package
        except IndexError:
            return work_package

    def _handle_work_done(self, rank):
        super(StratificationWorkScheduler, self)._handle_work_done(rank)
        self.running_blocks.pop(rank, None)
        if self._cover_enabled:
            state = self._cover_active_states.get(rank)
            if state is not None and not state["strata"]:
                self._cover_active_states.pop(rank, None)

    def _repartition_in_background(self):
        self.repartition_future = self.repartition_worker_pool.apply_async(
            self._repartition,
            (
                self.dataset.split("train"),
                self.dataset.num_entities(),
                self.num_partitions,
                self.active_only,
                self.combine_mirror_blocks,
                TORCH_TO_NP_DTYPE[self.data_type],
            ),
        )

    def _refill_work(self):
        if self.repartition_epoch:
            self.partitions, self._entities_in_strata = self.repartition_future.get()
            self._repartition_in_background()
        # Always reset served tracking at epoch boundaries so partitions are re-issued.
        if hasattr(self, "_served_partitions"):
            self._served_partitions = set()
        if self._cover_enabled:
            self._init_cover_schedule()
        else:
            # Recompute schedule every epoch and rebuild the work queue.
            self.fixed_schedule = self.schedule_creator.create_schedule()
            ordered = self._order_by_schedule(deepcopy(self.partitions))
            self.work_to_do = deque(ordered.keys())
            self.config.log(f"Refill work: queued={len(self.work_to_do)}")

    def _order_by_schedule(self, partitions):
        if self.schedule_creator is None:
            return partitions
        schedule = self.schedule_creator.create_schedule()
        if not schedule:
            return partitions
        ordered = {}
        for iteration in schedule:
            for strata in iteration:
                if strata in partitions:
                    ordered[strata] = partitions[strata]
        for strata, data in partitions.items():
            if strata not in ordered:
                ordered[strata] = data
        return ordered

    def _supports_adaptive_feedback(self):
        return True

    def _init_cover_schedule(self):
        if self.num_partitions % self._cover_q != 0:
            raise ValueError(
                "COVER scheduling requires num_partitions divisible by "
                f"q={self._cover_q}, got num_partitions={self.num_partitions}."
            )
        self._cover_groups = self._build_cover_groups(
            self.num_partitions, self._cover_q
        )
        self._cover_group_index = 0
        self._cover_pending_states = deque()
        self._cover_active_states = {}
        if self._cover_log_groups:
            self.config.log(
                "COVER schedule groups="
                f"{len(self._cover_groups)} "
                f"states_per_group={self.num_partitions // self._cover_q}."
            )

    def _build_cover_groups(self, num_partitions, q):
        buckets = {(i, j) for i in range(num_partitions) for j in range(num_partitions)}
        groups = []
        while len(buckets) > num_partitions:
            partitions = list(range(num_partitions))
            group = []
            while partitions:
                state = []
                while len(state) < q:
                    picked = None
                    for idx, pid in enumerate(partitions):
                        ok = True
                        for existing in state:
                            if (existing, pid) not in buckets or (pid, existing) not in buckets:
                                ok = False
                                break
                        if ok:
                            picked = pid
                            partitions.pop(idx)
                            break
                    if picked is None:
                        picked = partitions.pop(0)
                    state.append(picked)
                for a in state:
                    for b in state:
                        if a == b:
                            continue
                        buckets.discard((a, b))
                group.append(state)
            groups.append(group)
        return groups

    def _cover_state_strata(self, state):
        strata = []
        for i in state:
            for j in state:
                if self.combine_mirror_blocks and j < i:
                    continue
                key = (i, j)
                if key in self.partitions:
                    strata.append(key)
        return strata

    def _cover_state_entities(self, strata):
        if not self._entities_in_strata:
            return None
        entity_sets = [
            self._entities_in_strata.get(s)
            for s in strata
            if self._entities_in_strata.get(s) is not None
        ]
        if not entity_sets:
            return None
        union = torch.unique(torch.cat(entity_sets)).to(dtype=self.data_type)
        max_entities = self._get_max_entities()
        if max_entities and union.numel() > max_entities:
            if not hasattr(self, "_cover_entity_overflow_logged"):
                self.config.log(
                    "COVER window entities exceed max_entities; "
                    "falling back to per-strata entities for partition sync."
                )
                self._cover_entity_overflow_logged = True
            return None
        return union

    def _advance_cover_group(self):
        if self._cover_group_index >= len(self._cover_groups):
            return False
        group = self._cover_groups[self._cover_group_index]
        self._cover_group_index += 1
        self._cover_pending_states = deque(group)
        if self._cover_log_groups:
            self.config.log(
                "COVER group "
                f"{self._cover_group_index}/{len(self._cover_groups)} "
                f"states={len(group)}."
            )
        return True

    def _acquire_strata_cover(self, rank):
        work_package = WorkPackage()
        state = self._cover_active_states.get(rank)
        if state is None:
            if not self._cover_pending_states and not self._cover_active_states:
                if not self._advance_cover_group():
                    return work_package
            if not self._cover_pending_states:
                work_package.wait = True
                return work_package
            buffer_state = self._cover_pending_states.popleft()
            strata = deque(self._cover_state_strata(buffer_state))
            entities = self._cover_state_entities(strata)
            state = {"strata": strata, "entities": entities}
            self._cover_active_states[rank] = state
        if not state["strata"]:
            work_package.wait = True
            return work_package
        strata = state["strata"].popleft()
        strata_data = self.partitions.get(strata)
        if self.combine_mirror_blocks and strata_data is not None:
            if strata[0] == strata[1]:
                mirror_strata = (strata[0] - 1, strata[1] - 1)
            else:
                mirror_strata = (strata[1], strata[0])
            mirror_data = self.partitions.get(mirror_strata)
            if mirror_data is not None:
                strata_data = torch.cat((strata_data, mirror_data))
        work_package.partition_id = strata
        partition_slice = self._adaptive_maybe_slice(strata, strata_data)
        if partition_slice is None or len(partition_slice) == 0:
            work_package.partition_data = strata_data
        else:
            work_package.partition_data = partition_slice
        if state["entities"] is None:
            work_package.entities_in_partition = self._entities_in_strata.get(strata)
        else:
            work_package.entities_in_partition = state["entities"]
        return work_package

    def _adaptive_get_partition_length(self, partition_id):
        data = self.partitions.get(partition_id)
        if data is None:
            return None
        length = len(data)
        if not self.combine_mirror_blocks:
            return length
        i, j = partition_id
        if i == j:
            if i % 2 == 0:
                return length
            mirror = (i - 1, j - 1)
        else:
            mirror = (j, i)
        mirror_data = self.partitions.get(mirror)
        if mirror_data is not None:
            length += len(mirror_data)
        return length

    def _adaptive_requeue_partition(self, partition_id):
        if isinstance(self.current_iteration, set):
            self.current_iteration.add(partition_id)
        else:
            super(StratificationWorkScheduler, self)._adaptive_requeue_partition(partition_id)

    def _load_partitions(self, num_partitions):
        start = time.time()
        partition_assignment = self.dataset.load_train_partitions(num_partitions)
        partitions = self._construct_partitions(
            partition_assignment, num_partitions, TORCH_TO_NP_DTYPE[self.data_type]
        )
        print("partition load time", time.time() - start)
        return partitions

    @staticmethod
    def _construct_partitions(partition_assignment, num_partitions, np_type=np.int64):
        (
            partition_indexes,
            partition_data,
        ) = StratificationWorkScheduler._numba_construct_partitions(
            np.ascontiguousarray(partition_assignment), num_partitions
        )
        partition_data = [
            torch.from_numpy(data.astype(np_type)).contiguous() for data in partition_data
        ]
        partitions = dict(zip(partition_indexes, partition_data))
        return partitions

    @staticmethod
    @numba.njit
    def _numba_construct_partitions(partition_assignment, num_partitions):
        partition_indexes = [
            (i, j) for i in range(num_partitions) for j in range(num_partitions)
        ]
        partition_id_lookup: Dict[Tuple[int, int], int] = dict()
        partition_lengths: Dict[int, int] = dict()
        partition_data = []
        for i in range(len(partition_indexes)):
            partition = partition_indexes[i]
            partition_id_lookup[partition] = i
            partition_lengths[i] = 0
            # pre-allocate too much memory, on expectation we would only need
            # len(partition_assignment) / math.pow(num_partitions, 2)
            # this is faster than extending memory later on if a partition is larger
            partition_data.append(
                np.empty(
                    int(len(partition_assignment) / num_partitions), dtype=np.int64
                )
            )

        # iterate over the partition assignments and assign each triple-id to its
        #  corresponding partition
        for i in range(len(partition_assignment)):
            pa = partition_assignment[i]
            pa_tuple = (pa[0], pa[1])
            partition_id = partition_id_lookup[pa_tuple]
            current_partition_size = partition_lengths[partition_id]
            partition_data[partition_id][current_partition_size] = i
            partition_lengths[partition_id] += 1

        # now get correct sizes of partitions
        for i in range(len(partition_data)):
            partition_data[i] = partition_data[i][: partition_lengths[i]]
        return partition_indexes, partition_data


class SchedulerClient:
    def __init__(self, config):
        self.scheduler_rank = get_min_rank(config) - 1
        self.machine_id = config.get("job.distributed.machine_id")
        if config.get("job.distributed.scheduler_data_type") not in ["int", "int32", "int64", "long"]:
            raise ValueError("Only long and int is supported as dtype for the scheduler communication")
        self.data_type = getattr(torch, config.get("job.distributed.scheduler_data_type"))
        try:
            prefetch = int(config.get("job.distributed.scheduler_prefetch"))
        except KeyError:
            prefetch = 1
        self.prefetch_per_request = max(1, prefetch)
        self._prefetched_work = deque()

    def get_init_info(self):
        cmd = torch.tensor([SCHEDULER_CMDS.INIT_INFO, 0], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)
        info_buffer = torch.zeros((2,), dtype=self.data_type)
        dist.recv(info_buffer, src=self.scheduler_rank)
        max_entities = info_buffer[0]
        max_relations = info_buffer[1]
        return max_entities, max_relations

    def _receive_work(
            self, cmd
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[int],
        Optional[int],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """

        Returns:
            work_buffer: tensor containing the triples
            entity_buffer: tensor containing the entities
            relation_buffer: tensor containing the relations
        """
        work_buffer = torch.empty((cmd[1].item(),), dtype=self.data_type)
        dist.recv(work_buffer, src=self.scheduler_rank)
        partition_buffer = torch.empty((1,), dtype=self.data_type)
        dist.recv(partition_buffer, src=self.scheduler_rank)
        partition_id = int(partition_buffer[0].item())
        version_buffer = torch.empty((1,), dtype=self.data_type)
        dist.recv(version_buffer, src=self.scheduler_rank)
        partition_version = int(version_buffer[0].item())
        # get partition entities
        dist.recv(cmd, src=self.scheduler_rank)
        num_entities = cmd[1].item()
        entity_buffer = None
        if num_entities != 0:
            entity_buffer = torch.empty((num_entities,), dtype=self.data_type)
            dist.recv(entity_buffer, src=self.scheduler_rank)
        # get partition relations
        dist.recv(cmd, src=self.scheduler_rank)
        num_relations = cmd[1].item()
        relation_buffer = None
        if num_relations != 0:
            relation_buffer = torch.empty((num_relations,), dtype=self.data_type)
            dist.recv(relation_buffer, src=self.scheduler_rank)
        dist.recv(cmd, src=self.scheduler_rank)
        window_members = None
        if cmd[1].item() > 0:
            window_members = torch.empty((cmd[1].item(),), dtype=self.data_type)
            dist.recv(window_members, src=self.scheduler_rank)
        dist.recv(cmd, src=self.scheduler_rank)
        window_entities = None
        if cmd[1].item() > 0:
            window_entities = torch.empty((cmd[1].item(),), dtype=self.data_type)
            dist.recv(window_entities, src=self.scheduler_rank)
        dist.recv(cmd, src=self.scheduler_rank)
        window_versions = None
        if cmd[1].item() > 0:
            window_versions = torch.empty((cmd[1].item(),), dtype=self.data_type)
            dist.recv(window_versions, src=self.scheduler_rank)
        return (
            work_buffer,
            entity_buffer,
            relation_buffer,
            partition_id,
            partition_version,
            window_members,
            window_entities,
            window_versions,
        )

    def _request_single_work(
        self,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[int],
        Optional[int],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        while True:
            cmd = torch.tensor(
                [SCHEDULER_CMDS.GET_WORK, self.machine_id], dtype=self.data_type
            )
            dist.send(cmd, dst=self.scheduler_rank)
            dist.recv(cmd, src=self.scheduler_rank)
            if cmd[0] == SCHEDULER_CMDS.WORK:
                return self._receive_work(cmd)
            elif cmd[0] == SCHEDULER_CMDS.WAIT:
                wait = max(0.0, float(cmd[1].item()))
                if wait > 0:
                    time.sleep(wait)
            else:
                return None, None, None, None, None, None, None, None

    def _fill_prefetch_queue(self):
        target = self.prefetch_per_request
        while len(self._prefetched_work) < target:
            (
                work,
                entities,
                relations,
                partition_id,
                partition_version,
                window_members,
                window_entities,
                window_versions,
            ) = self._request_single_work()
            if work is None:
                break
            self._prefetched_work.append(
                (
                    work,
                    entities,
                    relations,
                    partition_id,
                    partition_version,
                    window_members,
                    window_entities,
                    window_versions,
                )
            )

    def get_work(
        self,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[int],
        Optional[int],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """

        Returns:
            work_buffer: tensor containing the triples
            entity_buffer: tensor containing the entities
            relation_buffer: tensor containing the relations
        """
        if self.prefetch_per_request <= 1:
            return self._request_single_work()
        if not self._prefetched_work:
            self._fill_prefetch_queue()
        if self._prefetched_work:
            return self._prefetched_work.popleft()
        return None, None, None, None, None, None, None, None

    def get_pre_localize_work(self):
        """

        Returns:
            work: tensor containing the triples
            entities: tensor containing the entities
            relations: tensor containing the relations
            wait: bool indicating whether to wait for others to finish
        """
        cmd = torch.tensor([SCHEDULER_CMDS.PRE_LOCALIZE_WORK, self.machine_id], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)
        dist.recv(cmd, src=self.scheduler_rank)
        if cmd[0] == SCHEDULER_CMDS.WORK:
            (
                work,
                entities,
                relations,
                partition_id,
                partition_version,
                _,
                _,
                _,
            ) = self._receive_work(cmd)
            return work, entities, relations, partition_id, partition_version, False
        elif cmd[0] == SCHEDULER_CMDS.WAIT:
            return None, None, None, None, None, True
        else:
            return None, None, None, None, None, False

    def get_init_work(self, entity_embedder_size):
        """
        Get the entity ids that should be initialized by the worker.
        Receives start and end id from the scheduler
        Args:
            entity_embedder_size: size of the local entity embedding layer

        Returns:
            tensor containing range from start and end entity id

        """
        cmd = torch.tensor([SCHEDULER_CMDS.GET_INIT_WORK, entity_embedder_size], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)
        dist.recv(cmd, src=self.scheduler_rank)
        if cmd[0] > -1:
            return torch.arange(cmd[0], cmd[1], dtype=self.data_type)
        return None

    def register_eval_result(self, hist: dict, hist_filt: dict, hist_filt_test: dict):
        hists = [hist, hist_filt, hist_filt_test]
        cmd = torch.tensor([SCHEDULER_CMDS.REGISTER_EVAL_RESULT, sum(len(h) for h in hists)], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)
        for h in hists:
            for v in h.values():
                ranks = v.cpu()
                dist.send(ranks, dst=self.scheduler_rank)

    def get_eval_result(self, hist: dict, hist_filt: dict, hist_filt_test: dict):
        cmd = torch.tensor([SCHEDULER_CMDS.GET_EVAL_RESULT, -1], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)
        hists = [hist, hist_filt, hist_filt_test]
        for h in hists:
            for key, values in h.items():
                ranks = torch.empty(len(values))
                dist.recv(ranks, src=self.scheduler_rank)
                h[key] = ranks
        return hist, hist_filt, hist_filt_test

    def get_local_entities(self):
        cmd = torch.tensor([SCHEDULER_CMDS.GET_LOCAL_ENT, -1], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)
        dist.recv(cmd, src=self.scheduler_rank)
        if cmd[0] > 0:
            local_entities = torch.empty([cmd[0], ], dtype=self.data_type)
            dist.recv(local_entities, src=self.scheduler_rank)
            return local_entities.long()
        return None

    def work_done(self):
        cmd = torch.tensor([SCHEDULER_CMDS.WORK_DONE, self.machine_id], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)

    def shutdown(self):
        cmd = torch.tensor([SCHEDULER_CMDS.SHUTDOWN, self.machine_id], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)

    def register_partition_result(
        self, step_time, partition_version=None, chunk_size=0
    ):
        cmd = torch.tensor([SCHEDULER_CMDS.REGISTER_PARTITION_RESULT, 0], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)
        payload = torch.tensor([step_time], dtype=torch.float32)
        dist.send(payload, dst=self.scheduler_rank)
        version = -1 if partition_version is None else int(partition_version)
        version_tensor = torch.tensor([version], dtype=self.data_type)
        dist.send(version_tensor, dst=self.scheduler_rank)
        chunk_tensor = torch.tensor([int(chunk_size)], dtype=self.data_type)
        dist.send(chunk_tensor, dst=self.scheduler_rank)
        ack = torch.full((2,), -1, dtype=self.data_type)
        dist.recv(ack, src=self.scheduler_rank)
        return int(ack[0].item()), int(ack[1].item())

    def register_window_result(
        self, step_time, window_members, window_versions, chunk_size=0
    ):
        cmd = torch.tensor([SCHEDULER_CMDS.REGISTER_WINDOW_RESULT, 0], dtype=self.data_type)
        dist.send(cmd, dst=self.scheduler_rank)
        payload = torch.tensor([step_time], dtype=torch.float32)
        dist.send(payload, dst=self.scheduler_rank)
        count = 0 if window_members is None else len(window_members)
        info = torch.tensor([count], dtype=self.data_type)
        dist.send(info, dst=self.scheduler_rank)
        if count:
            members = torch.as_tensor(window_members, dtype=self.data_type)
            versions = torch.as_tensor(window_versions, dtype=self.data_type)
            dist.send(members, dst=self.scheduler_rank)
            dist.send(versions, dst=self.scheduler_rank)
        chunk_tensor = torch.tensor([int(chunk_size)], dtype=self.data_type)
        dist.send(chunk_tensor, dst=self.scheduler_rank)
        ack = torch.full((1,), -1, dtype=self.data_type)
        dist.recv(ack, src=self.scheduler_rank)
        conflict_count = int(ack[0].item())
        if conflict_count <= 0:
            return []
        ids = torch.empty((conflict_count,), dtype=self.data_type)
        orig = torch.empty((conflict_count,), dtype=self.data_type)
        replay = torch.empty((conflict_count,), dtype=self.data_type)
        dist.recv(ids, src=self.scheduler_rank)
        dist.recv(orig, src=self.scheduler_rank)
        dist.recv(replay, src=self.scheduler_rank)
        return [
            (int(pid), int(ov), int(rv))
            for pid, ov, rv in zip(ids.tolist(), orig.tolist(), replay.tolist())
        ]

    def register_partition_gradient(self, partition_id, grad_sum, sample_count):
        if partition_id is None:
            return
        # partition_id is sent by the scheduler as a scalar; complex ids are encoded as negative aliases.
        if isinstance(partition_id, (list, tuple)):
            return
        try:
            partition_id = int(partition_id)
        except (TypeError, ValueError):
            return
        # allow negative partition aliases; only -1 means NO_WORK
        if partition_id == -1:
            return
        cmd = torch.tensor(
            [SCHEDULER_CMDS.REGISTER_PARTITION_GRADIENT, self.machine_id],
            dtype=self.data_type,
        )
        dist.send(cmd, dst=self.scheduler_rank)
        info = torch.tensor([partition_id, sample_count], dtype=self.data_type)
        dist.send(info, dst=self.scheduler_rank)
        payload = torch.tensor([grad_sum], dtype=torch.float32)
        dist.send(payload, dst=self.scheduler_rank)

    def register_partition_relation_gradient(
        self, partition_id, relation_ids, grad_sums, grad_counts
    ):
        if partition_id is None:
            return
        # partition_id is sent by the scheduler as a scalar; complex ids are encoded as negative aliases.
        if isinstance(partition_id, (list, tuple)):
            return
        try:
            partition_id = int(partition_id)
        except (TypeError, ValueError):
            return
        # allow negative partition aliases; only -1 means NO_WORK
        if partition_id == -1:
            return
        relation_ids = torch.as_tensor(
            relation_ids, dtype=self.data_type, device="cpu"
        )
        if relation_ids.numel() == 0:
            return
        grad_counts = torch.as_tensor(grad_counts, dtype=self.data_type, device="cpu")
        grad_sums = torch.as_tensor(grad_sums, dtype=torch.float32, device="cpu")
        cmd = torch.tensor(
            [SCHEDULER_CMDS.REGISTER_PARTITION_RELATION_GRADIENT, self.machine_id],
            dtype=self.data_type,
        )
        dist.send(cmd, dst=self.scheduler_rank)
        info = torch.tensor(
            [partition_id, relation_ids.numel()], dtype=self.data_type
        )
        dist.send(info, dst=self.scheduler_rank)
        dist.send(relation_ids, dst=self.scheduler_rank)
        dist.send(grad_counts, dst=self.scheduler_rank)
        dist.send(grad_sums, dst=self.scheduler_rank)


class LocalSchedulerClient:
    """Single-process scheduler client that bypasses torch.distributed."""

    def __init__(self, config, dataset):
        self.scheduler = WorkScheduler.create(config=config, dataset=dataset)
        self.scheduler._init_in_started_process()
        self.rank = get_min_rank(config)
        self.machine_id = config.get("job.distributed.machine_id")
        if config.get("job.distributed.scheduler_data_type") not in [
            "int",
            "int32",
            "int64",
            "long",
        ]:
            raise ValueError(
                "Only long and int is supported as dtype for the scheduler communication"
            )
        self.data_type = getattr(torch, config.get("job.distributed.scheduler_data_type"))
        try:
            prefetch = int(config.get("job.distributed.scheduler_prefetch"))
        except KeyError:
            prefetch = 1
        self.prefetch_per_request = max(1, prefetch)
        self._prefetched_work = deque()
        self._epoch_start = None
        self._epoch_seen_partitions = set()
        try:
            self._skip_duplicate_partitions = bool(
                config.get("job.distributed.skip_duplicate_partitions")
            )
        except KeyError:
            self._skip_duplicate_partitions = True
        overlap_cfg = (config.get("job.distributed.glow") or {}).get(
            "causal_overlap"
        ) or {}
        self._causal_overlap_enabled = bool(overlap_cfg.get("enable", False))
        if self._causal_overlap_enabled:
            self._skip_duplicate_partitions = False

    def _prepare_work(self, work_package, pre_localize=False):
        if work_package is None or work_package.partition_data is None:
            return None, None, None, None, None, None, None, None
        if work_package.partition_id is not None:
            if work_package.reuse_partition_version:
                if work_package.partition_version is None:
                    work_package.partition_version = (
                        self.scheduler.partition_issue_versions[
                            work_package.partition_id
                        ]
                    )
            else:
                current_version = self.scheduler.partition_issue_versions[
                    work_package.partition_id
                ]
                work_package.partition_version = current_version
                self.scheduler.partition_issue_versions[work_package.partition_id] = (
                    current_version + 1
                )
        if not pre_localize:
            self.scheduler.active_partition_per_worker[self.rank] = (
                work_package.partition_id
            )
            if (
                work_package.window_members is not None
                and work_package.window_versions is not None
            ):
                members = [int(x) for x in work_package.window_members]
                versions = [int(x) for x in work_package.window_versions]
                self.scheduler.active_partition_versions[self.rank] = dict(
                    zip(members, versions)
                )
            else:
                self.scheduler.active_partition_versions[self.rank] = (
                    work_package.partition_version
                )
            self.scheduler.active_partition_chunk_sizes[self.rank] = len(
                work_package.partition_data
            )
            if hasattr(self.scheduler, "previous_partition_per_worker"):
                self.scheduler.previous_partition_per_worker[self.rank] = (
                    work_package.partition_id
                )
        window_members = work_package.window_members
        if window_members is not None and not isinstance(window_members, torch.Tensor):
            window_members = torch.as_tensor(window_members, dtype=self.data_type)
        window_entities = work_package.window_entities
        if window_entities is not None and not isinstance(window_entities, torch.Tensor):
            window_entities = torch.as_tensor(window_entities, dtype=self.data_type)
        window_versions = work_package.window_versions
        if window_versions is not None and not isinstance(window_versions, torch.Tensor):
            window_versions = torch.as_tensor(window_versions, dtype=self.data_type)
        return (
            work_package.partition_data,
            work_package.entities_in_partition,
            work_package.relations_in_partition,
            work_package.partition_id,
            work_package.partition_version,
            window_members,
            window_entities,
            window_versions,
        )

    def _request_single_work(self):
        if self._epoch_start is None:
            self._epoch_start = time.time()
        while True:
            work_package = self.scheduler._next_work(self.rank, self.machine_id)
            if work_package is None:
                return None, None, None, None, None, None, None, None
            if work_package.wait:
                if self.scheduler.wait_time > 0:
                    time.sleep(self.scheduler.wait_time)
                continue
            if work_package.partition_data is None:
                if self._epoch_start is not None:
                    elapsed = time.time() - self._epoch_start
                    self.scheduler.config.log(f"complete_epoch_time: {elapsed}")
                    self._epoch_start = None
                self.scheduler.num_processed_partitions = 0
                self.scheduler._refill_work()
                self.scheduler._on_epoch_completed()
                self._epoch_seen_partitions.clear()
                return None, None, None, None, None, None, None, None
            return self._prepare_work(work_package, pre_localize=False)

    def _next_work(self):
        if self.prefetch_per_request <= 1:
            return self._request_single_work()
        if not self._prefetched_work:
            self._fill_prefetch_queue()
        if self._prefetched_work:
            return self._prefetched_work.popleft()
        return None, None, None, None, None, None, None, None

    def _fill_prefetch_queue(self):
        target = self.prefetch_per_request
        while len(self._prefetched_work) < target:
            work = self._request_single_work()
            if work[0] is None:
                break
            self._prefetched_work.append(work)

    def get_work(self):
        if (
            not self._skip_duplicate_partitions
            or getattr(self.scheduler, "_adaptive_enabled", False)
        ):
            return self._next_work()
        max_attempts = max(8, self.scheduler.num_partitions * 2)
        attempts = 0
        while True:
            work = self._next_work()
            if work[0] is None:
                return work
            partition_id = work[3]
            if partition_id is None:
                return work
            partition_key = partition_id
            if isinstance(partition_key, torch.Tensor):
                partition_key = partition_key.detach().cpu()
                if partition_key.numel() == 1:
                    partition_key = int(partition_key.item())
                else:
                    partition_key = tuple(
                        int(x) for x in partition_key.view(-1).tolist()
                    )
            elif isinstance(partition_key, (list, tuple)):
                partition_key = tuple(int(x) for x in partition_key)
            else:
                try:
                    partition_key = int(partition_key)
                except (TypeError, ValueError):
                    pass
            if partition_key in self._epoch_seen_partitions:
                attempts += 1
                if attempts >= max_attempts:
                    return work
                continue
            self._epoch_seen_partitions.add(partition_key)
            return work

    def get_pre_localize_work(self):
        try:
            work_package = self.scheduler._handle_pre_localize_work(
                rank=self.rank, machine_id=self.machine_id
            )
        except ValueError:
            return None, None, None, None, None, True
        work = self._prepare_work(work_package, pre_localize=True)
        if work[0] is None:
            return None, None, None, None, None, False
        return (
            work[0],
            work[1],
            work[2],
            work[3],
            work[4],
            False,
        )

    def get_init_info(self):
        return self.scheduler._get_max_entities(), self.scheduler._get_max_relations()

    def get_init_work(self, entity_embedder_size):
        if self.scheduler.init_up_to_entity == -1:
            self.scheduler.config.log("initialize parameter server")
        self.scheduler.init_up_to_entity += 1
        if self.scheduler.init_up_to_entity >= self.scheduler.dataset.num_entities():
            return None
        start = self.scheduler.init_up_to_entity
        end = min(
            self.scheduler.dataset.num_entities(),
            self.scheduler.init_up_to_entity + entity_embedder_size,
        )
        if end == self.scheduler.dataset.num_entities():
            self.scheduler.config.log("parameter server initialized")
        self.scheduler.init_up_to_entity += entity_embedder_size
        return torch.arange(start, end, dtype=self.data_type)

    def get_local_entities(self):
        entities = self.scheduler.local_entities.get(self.rank)
        if entities is None:
            return None
        return entities.long()

    def work_done(self):
        self.scheduler._handle_work_done(self.rank)

    def shutdown(self):
        self.scheduler._on_scheduler_shutdown()

    def register_partition_result(self, step_time, partition_version=None, chunk_size=0):
        return self.scheduler._register_partition_result(
            self.rank,
            float(step_time),
            reported_version=partition_version,
            reported_chunk_size=chunk_size,
        )

    def register_window_result(self, step_time, window_members, window_versions, chunk_size=0):
        return self.scheduler._register_window_result(
            self.rank,
            float(step_time),
            window_members,
            window_versions,
            reported_chunk_size=chunk_size,
        )

    def register_partition_gradient(self, partition_id, grad_sum, sample_count):
        if partition_id is None:
            return
        # In local mode we can accept complex ids (e.g., (i,j) strata keys) directly.
        # Also allow negative aliases produced by the scheduler encoder; decode them back.
        if isinstance(partition_id, (list, tuple)):
            partition_id = tuple(int(x) for x in partition_id)
        else:
            try:
                partition_id = int(partition_id)
            except (TypeError, ValueError):
                return
            if partition_id == -1:
                return
            if partition_id < -1:
                partition_id = self.scheduler._decode_partition_id(partition_id)
        self.scheduler._register_partition_gradient(
            partition_id, float(grad_sum), int(sample_count)
        )

    def register_partition_relation_gradient(
        self, partition_id, relation_ids, grad_sums, grad_counts
    ):
        self.scheduler._register_partition_relation_gradient(
            partition_id, relation_ids, grad_sums, grad_counts
        )

    def register_eval_result(self, hist: dict, hist_filt: dict, hist_filt_test: dict):
        self._eval_hists = [hist, hist_filt, hist_filt_test]

    def get_eval_result(self, hist: dict, hist_filt: dict, hist_filt_test: dict):
        return hist, hist_filt, hist_filt_test
