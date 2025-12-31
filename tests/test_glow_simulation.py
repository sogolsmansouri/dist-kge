import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from random import Random

import numpy as np
import torch

from kge import Dataset
from kge.distributed.parameter_client import KgeParameterClient
from kge.distributed.misc import get_min_rank, get_num_keys, get_optimizer_dim
from kge.distributed.work_scheduler import LocalSchedulerClient
from kge.model import KgeModel
from kge.job.train_negative_sampling_distributed import (
    TrainingJobNegativeSamplingDistributed,
)
from tests.util import create_config, get_dataset_folder


def _write_synthetic_dataset(root: str, *, num_entities: int, num_relations: int,
                             num_partitions: int, num_clusters: int,
                             triples_per_partition: int, seed: int):
    rng = np.random.RandomState(seed)
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)

    # Entity/relation maps (id \t name)
    with (root_path / "entity_ids.del").open("w") as fp:
        for eid in range(num_entities):
            fp.write(f"{eid}\te{eid}\n")
    with (root_path / "relation_ids.del").open("w") as fp:
        for rid in range(num_relations):
            fp.write(f"{rid}\tr{rid}\n")

    partitions_per_cluster = num_partitions // num_clusters
    cluster_size = num_entities // num_clusters
    cluster_entities = [
        np.arange(c * cluster_size, (c + 1) * cluster_size)
        for c in range(num_clusters)
    ]

    triples = []
    partition_ids = []
    for pid in range(num_partitions):
        cluster = min(num_clusters - 1, pid // partitions_per_cluster)
        pool = cluster_entities[cluster]
        for _ in range(triples_per_partition):
            s = int(rng.choice(pool))
            o = int(rng.choice(pool))
            p = int(rng.randint(0, num_relations))
            triples.append((s, p, o))
            partition_ids.append(pid)

    triples = np.asarray(triples, dtype=np.int64)
    partition_ids = np.asarray(partition_ids, dtype=np.int64)

    # Train/valid/test splits.
    split_train = triples
    split_valid = triples[: max(1, len(triples) // 20)]
    split_test = triples[max(1, len(triples) // 20): max(2, len(triples) // 10)]

    np.savetxt(root_path / "train.del", split_train, fmt="%d", delimiter="\t")
    np.savetxt(root_path / "valid.del", split_valid, fmt="%d", delimiter="\t")
    np.savetxt(root_path / "test.del", split_test, fmt="%d", delimiter="\t")

    # Partition assignments (stratification base type).
    part_dir = root_path / "partitions" / "stratification" / f"num_{num_partitions}"
    part_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        part_dir / "train_assign_partitions.del",
        partition_ids,
        fmt="%d",
        delimiter="\t",
    )

    dataset_yaml = (
        "dataset:\n"
        "  files.entity_ids.filename: entity_ids.del\n"
        "  files.entity_ids.type: map\n"
        "  files.relation_ids.filename: relation_ids.del\n"
        "  files.relation_ids.type: map\n"
        "  files.train.filename: train.del\n"
        "  files.train.size: {train_size}\n"
        "  files.train.type: triples\n"
        "  files.valid.filename: valid.del\n"
        "  files.valid.size: {valid_size}\n"
        "  files.valid.type: triples\n"
        "  files.test.filename: test.del\n"
        "  files.test.size: {test_size}\n"
        "  files.test.type: triples\n"
        "  name: synthetic_glow\n"
        "  num_entities: {num_entities}\n"
        "  num_relations: {num_relations}\n"
    ).format(
        train_size=len(split_train),
        valid_size=len(split_valid),
        test_size=len(split_test),
        num_entities=num_entities,
        num_relations=num_relations,
    )
    (root_path / "dataset.yaml").write_text(dataset_yaml)


def _avg_window_overlap(order, entity_map, window_size: int) -> float:
    if not order or window_size <= 0:
        return 0.0
    total = 0.0
    count = 0
    for idx in range(0, len(order) - window_size + 1):
        window = order[idx: idx + window_size]
        sets = [set(entity_map[pid].tolist()) for pid in window]
        sum_sizes = sum(len(s) for s in sets)
        if sum_sizes == 0:
            continue
        union = set().union(*sets)
        overlap = 1.0 - (len(union) / float(sum_sizes))
        total += overlap
        count += 1
    return total / max(1, count)

def _avg_unique_entities(order, entity_map, window_size: int) -> float:
    if not order or window_size <= 0:
        return 0.0
    total = 0.0
    count = 0
    for idx in range(0, len(order) - window_size + 1):
        window = order[idx: idx + window_size]
        sets = [set(entity_map[pid].tolist()) for pid in window]
        if not sets:
            continue
        union = set().union(*sets)
        total += len(union)
        count += 1
    return total / max(1, count)

def _make_shared_parameter_client(config, dataset):
    num_keys = get_num_keys(config, dataset)
    embedding_dim = config.get("lookup_embedder.dim")
    optimizer_dim = get_optimizer_dim(config, embedding_dim)
    parameters = torch.zeros(
        (num_keys, embedding_dim + optimizer_dim),
        dtype=torch.float32,
        requires_grad=False,
    )
    return KgeParameterClient.create(
        config=config,
        server_id=0,
        client_id=get_min_rank(config),
        server=parameters,
        num_keys=num_keys,
    )

class TestGlowSimulation(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.mkdtemp(prefix="glow-synth-")
        self.dataset_dir = os.path.join(self._temp_dir, "synthetic_glow")
        _write_synthetic_dataset(
            self.dataset_dir,
            num_entities=60,
            num_relations=6,
            num_partitions=6,
            num_clusters=2,
            triples_per_partition=120,
            seed=7,
        )

        self.config = create_config("synthetic_glow")
        self.config.folder = self._temp_dir
        self.config.set("job.type", "train")
        self.config.set("job.distributed.partition_type", "glow")
        self.config.set("job.distributed.glow.base_partition_type", "stratification")
        self.config.set("job.distributed.num_partitions", 6)
        self.config.set("job.distributed.num_workers", 1)
        self.config.set("job.distributed.num_machines", 1)
        self.config.set("job.distributed.parameter_server", "shared")
        self.config.set("job.distributed.glow.window_size", 2)
        self.config.set("job.distributed.glow.window_overlap", 1)
        self.config.set("job.distributed.glow.window_work", False)
        self.config.set("job.distributed.glow.bandit.enable", False)
        self.dataset = Dataset.create(
            config=self.config, folder=self.dataset_dir, preload_data=True
        )

    def tearDown(self):
        if self._temp_dir and os.path.isdir(self._temp_dir):
            shutil.rmtree(self._temp_dir)

    def test_glow_order_improves_overlap(self):
        client = LocalSchedulerClient(config=self.config, dataset=self.dataset)
        scheduler = client.scheduler

        # Encode synthetic "hotness" to group partitions by cluster.
        cluster_hotness = {0: 2.0, 1: 2.0, 2: 2.0, 3: 1.0, 4: 1.0, 5: 1.0}
        for pid, score in cluster_hotness.items():
            scheduler.partition_gradient_stats[pid] = {"sum": score, "count": 1}
        scheduler._refill_work()

        glow_order = []
        while True:
            work, _ents, _rels, pid, _pver, _wmem, _wents, _wvers = client.get_work()
            if work is None:
                break
            glow_order.append(pid)
            client.work_done()

        entity_map = scheduler._partition_entities_map
        glow_overlap = _avg_window_overlap(
            glow_order, entity_map, window_size=2
        )

        rng = Random(0)
        random_order = list(range(self.config.get("job.distributed.num_partitions")))
        rng.shuffle(random_order)
        random_overlap = _avg_window_overlap(
            random_order, entity_map, window_size=2
        )

        # Glow ordering should increase expected overlap on this synthetic dataset.
        self.assertGreater(glow_overlap, random_overlap)

    def test_glow_order_reduces_unique_entities_per_window(self):
        client = LocalSchedulerClient(config=self.config, dataset=self.dataset)
        scheduler = client.scheduler

        # Same synthetic hotness signal as overlap test.
        cluster_hotness = {0: 2.0, 1: 2.0, 2: 2.0, 3: 1.0, 4: 1.0, 5: 1.0}
        for pid, score in cluster_hotness.items():
            scheduler.partition_gradient_stats[pid] = {"sum": score, "count": 1}
        scheduler._refill_work()

        glow_order = []
        while True:
            work, _ents, _rels, pid, _pver, _wmem, _wents, _wvers = client.get_work()
            if work is None:
                break
            glow_order.append(pid)
            client.work_done()

        entity_map = scheduler._partition_entities_map
        glow_unique = _avg_unique_entities(glow_order, entity_map, window_size=2)

        rng = Random(1)
        random_order = list(range(self.config.get("job.distributed.num_partitions")))
        rng.shuffle(random_order)
        random_unique = _avg_unique_entities(random_order, entity_map, window_size=2)

        # Glow ordering should reduce unique entities per window (IO proxy).
        self.assertLess(glow_unique, random_unique)

    def test_glow_bandit_reorders_windows(self):
        cfg = self.config.clone()
        cfg.set("job.distributed.glow.bandit.enable", True)
        cfg.set("job.distributed.glow.window_work", True)
        cfg.set("job.distributed.glow.bandit.reward_scale", 1.0)
        dataset = Dataset.create(config=cfg, folder=self.dataset_dir, preload_data=True)
        client = LocalSchedulerClient(config=cfg, dataset=dataset)
        scheduler = client.scheduler

        windows = [tuple(entry["key"]) for entry in scheduler._glow_windows]
        self.assertGreaterEqual(len(windows), 2)
        slow = windows[0]
        fast = windows[1]
        scheduler._after_partition_result(slow, avg_time=10.0, chunk_size=100)
        scheduler._after_partition_result(fast, avg_time=1.0, chunk_size=100)
        scheduler._sort_glow_windows()
        top = tuple(scheduler._glow_windows[0]["key"])
        self.assertEqual(top, fast)

    def test_glow_end_to_end_training_runs(self):
        def run_epoch(partition_type: str):
            cfg = create_config("dataset_test")
            cfg.folder = tempfile.mkdtemp(prefix=f"glow-train-{partition_type}-")
            cfg.set("job.type", "train")
            cfg.set("train.type", "distributed_negative_sampling")
            cfg.set("job.device", "cpu")
            cfg.set("job.distributed.single_process", True)
            cfg.set("job.distributed.parameter_server", "shared")
            cfg.set("job.distributed.num_workers", 1)
            cfg.set("job.distributed.num_machines", 1)
            cfg.set("job.distributed.num_partitions", 2)
            cfg.set("job.distributed.partition_type", partition_type)
            cfg.set("train.num_workers", 0)
            cfg.set("train.batch_size", 2)
            cfg.set("train.max_epochs", 1)
            cfg.set("train.optimizer.default.type", "dist_adagrad")
            cfg.set("lookup_embedder.sparse", True)
            cfg.set("negative_sampling.num_samples.s", 1)
            cfg.set("negative_sampling.num_samples.o", 1)
            cfg.set("negative_sampling.num_samples.p", 0)
            if partition_type == "glow":
                cfg.set("job.distributed.glow.base_partition_type", "random")
                cfg.set("job.distributed.glow.window_size", 2)
                cfg.set("job.distributed.glow.window_overlap", 1)
                cfg.set("job.distributed.glow.window_work", False)
                cfg.set("job.distributed.glow.bandit.enable", False)
            dataset = Dataset.create(
                config=cfg, folder=get_dataset_folder("dataset_test"), preload_data=True
            )
            parameter_client = _make_shared_parameter_client(cfg, dataset)
            scheduler_client = LocalSchedulerClient(config=cfg, dataset=dataset)
            max_entities, _max_relations = scheduler_client.get_init_info()
            model = KgeModel.create(
                config=cfg,
                dataset=dataset,
                parameter_client=parameter_client,
                max_partition_entities=max_entities,
            )
            job = TrainingJobNegativeSamplingDistributed(
                config=cfg,
                dataset=dataset,
                model=model,
                parameter_client=parameter_client,
                work_scheduler_client=scheduler_client,
            )
            job._prepare()
            trace = job.run_epoch()
            scheduler_client.shutdown()
            shutil.rmtree(cfg.folder)
            return trace

        glow_trace = run_epoch("glow")
        random_trace = run_epoch("random")
        for trace in (glow_trace, random_trace):
            self.assertTrue(math.isfinite(trace["avg_loss"]))
            self.assertGreater(trace["processed_batches"], 0)


if __name__ == "__main__":
    unittest.main()
