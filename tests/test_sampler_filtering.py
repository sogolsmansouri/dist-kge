import torch

from kge import Dataset
from kge.util.sampler import (
    KgeUniformSampler,
    KgeHierarchicalFrequencySampler,
    S,
    P,
    O,
)
from tests.util import create_config, get_dataset_folder


def _create_dataset():
    dataset_name = "dataset_test"
    config = create_config(dataset_name)
    dataset = Dataset.create(
        config=config, folder=get_dataset_folder(dataset_name), preload_data=True
    )
    return config, dataset


def test_uniform_filtering_fast_matches_standard():
    config, dataset = _create_dataset()
    config.set("negative_sampling.filtering.s", True)
    config.set("negative_sampling.filtering.p", False)
    config.set("negative_sampling.filtering.o", False)
    config.set("negative_sampling.filtering.split", "train")
    config.set("negative_sampling.filtering.implementation", "fast")
    sampler = KgeUniformSampler(config, "negative_sampling", dataset)

    positive_triples = dataset.split("train")[:3]
    torch.manual_seed(7)
    negative_samples = sampler._sample(positive_triples, S, 5)
    negative_samples[0, 0] = positive_triples[0, S]
    if positive_triples.size(0) > 1:
        negative_samples[1, 0] = positive_triples[1, S]

    torch.manual_seed(13)
    filtered_standard = sampler._filter_and_resample(
        negative_samples.clone(), S, positive_triples
    )
    torch.manual_seed(13)
    filtered_fast = sampler._filter_and_resample_fast(
        negative_samples.clone(), S, positive_triples
    )
    assert torch.equal(filtered_standard, filtered_fast)

    index = dataset.index("train_po_to_s")
    pairs = positive_triples[:, [P, O]]
    for i in range(pairs.size(0)):
        positives = index.get((pairs[i][0].item(), pairs[i][1].item()))
        if isinstance(positives, list) or positives.numel() == 0:
            continue
        assert not torch.isin(filtered_standard[i], positives).any()


def test_hfrequency_sampling_matches_reference():
    config, dataset = _create_dataset()
    config.set("negative_sampling.sampling_type", "hfrequency")
    sampler = KgeHierarchicalFrequencySampler(config, "negative_sampling", dataset)

    positive_triples = dataset.split("train")[:4]
    num_samples = 6

    torch.manual_seed(123)
    sampled = sampler._sample(positive_triples, S, num_samples)

    torch.manual_seed(123)
    probs = sampler._h2_multinomials[S]
    group_draws = torch.multinomial(
        probs, positive_triples.size(0) * num_samples, replacement=True
    )
    group_counts = sampler._h2_group_counts[S]
    group_offsets = sampler._h2_group_offsets[S]
    sorted_indices = sampler._h2_sorted_indices[S]
    rand = torch.rand(group_draws.numel())
    expected = torch.empty(group_draws.numel(), dtype=torch.long)
    for i, group_idx in enumerate(group_draws):
        count = int(group_counts[group_idx])
        offset = int(group_offsets[group_idx])
        within = int(rand[i].item() * count)
        expected[i] = sorted_indices[offset + within]
    expected = expected.view(positive_triples.size(0), num_samples)

    assert torch.equal(sampled, expected)
