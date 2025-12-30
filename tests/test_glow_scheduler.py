import os
import shutil
import tempfile
import unittest

from kge import Dataset
from kge.distributed.work_scheduler import LocalSchedulerClient
from tests.util import create_config, get_dataset_folder


class TestGlowScheduler(unittest.TestCase):
    def setUp(self):
        self.dataset_name = "dataset_test"
        self.dataset_folder = get_dataset_folder(self.dataset_name)
        self.config = create_config(self.dataset_name)
        self._temp_dir = tempfile.mkdtemp(prefix="glow-scheduler-")
        self.config.folder = self._temp_dir
        self.config.set("job.type", "train")
        self.config.set("job.distributed.partition_type", "glow")
        self.config.set("job.distributed.num_partitions", 2)
        self.config.set("job.distributed.num_workers", 1)
        self.config.set("job.distributed.num_machines", 1)
        self.config.set("job.distributed.parameter_server", "shared")
        self.config.set("job.distributed.glow.window_work", True)
        self.config.set("job.distributed.glow.window_size", 2)
        self.config.set("job.distributed.glow.window_overlap", 1)
        self.config.set("job.distributed.glow.bandit.enable", True)
        self.config.set("job.distributed.glow.bandit.reward_scale", 1.0)
        self.config.set("job.distributed.glow.bandit.conflict_penalty", 0.0)
        self.config.set("job.distributed.glow.bandit.queue_penalty_scale", 0.0)
        self.config.set("job.distributed.glow.bandit.overlap_penalty", 0.0)
        self.dataset = Dataset.create(
            config=self.config, folder=self.dataset_folder, preload_data=True
        )

    def tearDown(self):
        if self._temp_dir and os.path.isdir(self._temp_dir):
            shutil.rmtree(self._temp_dir)

    def test_window_work_bandit_uses_full_key(self):
        client = LocalSchedulerClient(config=self.config, dataset=self.dataset)
        scheduler = client.scheduler
        self.assertTrue(scheduler._glow_windows)
        full_key = tuple(int(x) for x in scheduler._glow_windows[0]["key"])
        scheduler._served_partitions.add(full_key[0])
        (
            work,
            _entities,
            _relations,
            _partition_id,
            _partition_version,
            window_members,
            _window_entities,
            window_versions,
        ) = client.get_work()
        self.assertIsNotNone(work)
        self.assertIsNotNone(window_members)
        members = tuple(int(x) for x in window_members.tolist())
        self.assertNotEqual(members, full_key)
        client.register_window_result(
            1.0, window_members, window_versions, chunk_size=int(work.numel())
        )
        self.assertIn(full_key, scheduler._window_scores)
        self.assertGreater(scheduler._window_scores[full_key], 0.0)
        self.assertNotIn(members, scheduler._window_work_key_map)


if __name__ == "__main__":
    unittest.main()
