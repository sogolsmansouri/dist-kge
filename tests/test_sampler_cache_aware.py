import torch

from kge import Dataset
from kge.util.sampler import KgeSampler, KgeUniformSampler, S, O
from tests.util import create_config, get_dataset_folder


def _create_dataset():
    dataset_name = "dataset_test"
    config = create_config(dataset_name)
    dataset = Dataset.create(
        config=config, folder=get_dataset_folder(dataset_name), preload_data=True
    )
    return config, dataset


def test_cache_aware_resident_only_samples_from_pool():
    config, dataset = _create_dataset()
    config.set("negative_sampling.sampling_type", "cache_aware")
    config.set("negative_sampling.cache_aware.resident_fraction", 1.0)
    config.set("negative_sampling.cache_aware.resident_sampling_type", "pooled")
    config.set("negative_sampling.cache_aware.background_sampling_type", "uniform")
    config.set("negative_sampling.shared", False)
    sampler = KgeSampler.create(config, "negative_sampling", dataset)

    pool = torch.tensor([0, 1], dtype=torch.long)
    sampler.set_pool(pool, S)
    sampler.set_pool(pool, O)

    triples = dataset.split("train")[:4]
    torch.manual_seed(7)
    negatives = sampler.sample(triples, S).samples()

    assert torch.isin(negatives, pool).all()


def test_cache_aware_zero_fraction_matches_uniform():
    config, dataset = _create_dataset()
    config.set("negative_sampling.sampling_type", "cache_aware")
    config.set("negative_sampling.cache_aware.resident_fraction", 0.0)
    config.set("negative_sampling.cache_aware.resident_sampling_type", "pooled")
    config.set("negative_sampling.cache_aware.background_sampling_type", "uniform")
    config.set("negative_sampling.shared", False)
    sampler_cache = KgeSampler.create(config, "negative_sampling", dataset)

    config_uniform = create_config("dataset_test")
    config_uniform.set("negative_sampling.sampling_type", "uniform")
    config_uniform.set("negative_sampling.shared", False)
    sampler_uniform = KgeUniformSampler(config_uniform, "negative_sampling", dataset)

    triples = dataset.split("train")[:4]
    torch.manual_seed(11)
    negatives_cache = sampler_cache.sample(triples, S).samples()
    torch.manual_seed(11)
    negatives_uniform = sampler_uniform.sample(triples, S).samples()

    assert torch.equal(negatives_cache, negatives_uniform)
