import pytest
import torch

from kge import Dataset
from kge.distributed.misc import get_optimizer_dim
from kge.job.train_negative_sampling_distributed import (
    TrainingJobNegativeSamplingDistributed,
)
from kge.model.embedder.distributed_lookup_embedder import DistributedLookupEmbedder
from kge.util.sampler import KgeUniformSampler, S
from tests.util import create_config, get_dataset_folder


class _DummyParameterClient:
    def __init__(self, dim: int):
        self.dim = dim


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU sampling test")
def test_uniform_sampler_sample_on_gpu():
    config = create_config("dataset_test")
    config.set("job.device", "cuda:0")
    config.set("job.distributed.sample_on_gpu", True)
    config.set("negative_sampling.sampling_type", "uniform")
    config.set("negative_sampling.implementation", "triple")
    dataset = Dataset.create(
        config=config,
        folder=get_dataset_folder("dataset_test"),
        preload_data=True,
    )
    sampler = KgeUniformSampler(config, "negative_sampling", dataset)
    triples = torch.tensor([[0, 0, 1], [1, 0, 2]], device="cuda")
    assert sampler.supports_device_sampling(triples)
    neg = sampler.sample(triples, S, num_samples=3)
    assert neg.samples().device.type == "cuda"

    config_cpu = config.clone()
    config_cpu.set("job.distributed.sample_on_gpu", False)
    sampler_cpu = KgeUniformSampler(config_cpu, "negative_sampling", dataset)
    assert not sampler_cpu.supports_device_sampling(triples)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU mapping test")
def test_map_ids_on_gpu_device_resolution():
    job = TrainingJobNegativeSamplingDistributed.__new__(
        TrainingJobNegativeSamplingDistributed
    )
    job._map_ids_on_gpu = True
    job.device = torch.device("cuda:0")
    device = TrainingJobNegativeSamplingDistributed._resolve_map_ids_device(job)
    assert device.type == "cuda"

    job._map_ids_on_gpu = False
    device_cpu = TrainingJobNegativeSamplingDistributed._resolve_map_ids_device(job)
    assert device_cpu.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU mapping test")
def test_map_ids_on_gpu_mapper_allocation():
    job = TrainingJobNegativeSamplingDistributed.__new__(
        TrainingJobNegativeSamplingDistributed
    )
    job._map_ids_device = torch.device("cuda:0")
    job._entity_vocab_size = 7
    job._relation_vocab_size = 3
    job._entity_partition_mapper_device = None
    job._relation_partition_mapper_device = None

    entity_mapper = TrainingJobNegativeSamplingDistributed._ensure_partition_mapper_device(
        job, "entity"
    )
    assert entity_mapper.device.type == "cuda"
    assert entity_mapper.numel() == 7
    assert (
        TrainingJobNegativeSamplingDistributed._ensure_partition_mapper_device(
            job, "entity"
        )
        is entity_mapper
    )

    relation_mapper = TrainingJobNegativeSamplingDistributed._ensure_partition_mapper_device(
        job, "relation"
    )
    assert relation_mapper.device.type == "cuda"
    assert relation_mapper.numel() == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU cache test")
def test_gpu_cache_insert_and_lookup():
    config = create_config("dataset_test")
    config.set("job.device", "cuda:0")
    config.set("train.optimizer.default.type", "dist_rowadagrad")
    config.set("job.distributed.gpu_cache.enable", True)
    config.set("job.distributed.gpu_cache.max_entries_entity", 16)
    dataset = Dataset.create(
        config=config,
        folder=get_dataset_folder("dataset_test"),
        preload_data=True,
    )
    optimizer_dim = get_optimizer_dim(config, config.get("lookup_embedder.dim"))
    param_client = _DummyParameterClient(config.get("lookup_embedder.dim") + optimizer_dim)
    embedder = DistributedLookupEmbedder(
        config,
        dataset,
        "complex.entity_embedder",
        vocab_size=dataset.num_entities(),
        parameter_client=param_client,
        complete_vocab_size=dataset.num_entities(),
    )
    embedder._embeddings = embedder._embeddings.to(torch.device("cuda:0"))
    embedder.to_device()

    indexes_cpu = torch.tensor([1, 2, 3], dtype=torch.long)
    emb_gpu = torch.randn((3, embedder.dim), device=embedder._embeddings.weight.device)
    opt_gpu = None
    if embedder.optimizer_dim > 0:
        opt_gpu = torch.randn(
            (3, embedder.optimizer_dim), device=embedder.optimizer_values.device
        )
    embedder._gpu_cache_insert(indexes_cpu, emb_gpu, opt_gpu)
    slots, mask = embedder._gpu_cache_lookup(indexes_cpu)
    assert mask is not None
    assert torch.all(mask)

    slots_device = slots.to(embedder._gpu_cache_embeddings.device)
    assert torch.allclose(embedder._gpu_cache_embeddings[slots_device], emb_gpu)
