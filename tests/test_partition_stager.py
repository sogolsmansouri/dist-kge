import torch

from kge.distributed.partition_stager import PartitionStager
from kge.job.train_negative_sampling_distributed import BatchDataset


def test_partition_stager_stages_partition(tmp_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stager = PartitionStager(device=device)
    triples = torch.arange(12, dtype=torch.long).view(-1, 3)
    stage = stager.stage(partition_id=3, version=7, triples=triples)
    assert stage.partition_id == 3
    assert stage.version == 7
    assert stage.num_triples == triples.size(0)
    assert torch.equal(stage.host_view, triples)
    assert torch.equal(stage.device_view.cpu(), triples)
    assert stager.current() is stage


def test_batch_dataset_uses_staged_views():
    triples = torch.arange(30, dtype=torch.long).view(-1, 3)
    dataset = BatchDataset(
        triples=triples,
        batch_size=2,
        shuffle=False,
        materialize=True,
        materialize_device=None,
    )
    dataset.partition_stager = PartitionStager(device=torch.device("cpu"))
    dataset.set_samples(torch.arange(triples.size(0)), epoch=0, partition_id=0, partition_version=1)

    idx = torch.arange(0, 4, dtype=torch.long)
    host_view = dataset.fetch_triples(idx)
    assert torch.equal(host_view, triples[idx])

    device_view = dataset.fetch_triples_device(idx)
    assert device_view is not None
    assert torch.equal(device_view, triples[idx])
