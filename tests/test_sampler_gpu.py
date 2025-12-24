import pytest
import torch

from kge import Config
from kge.util.sampler import DefaultBatchNegativeSample, S


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU sampling test")
def test_default_batch_negative_sample_map_samples_gpu():
    config = Config()
    config.set("negative_sampling.implementation", "triple")
    device = torch.device("cuda")
    triples = torch.arange(6, dtype=torch.long, device=device).view(-1, 3)
    samples = torch.randint(0, 16, (2, 4), device=device)
    neg_sample = DefaultBatchNegativeSample(
        config=config,
        configuration_key="negative_sampling",
        positive_triples=triples,
        slot=S,
        num_samples=4,
        samples=samples.clone(),
    )
    mapper = torch.arange(32, dtype=torch.long)
    mapper = mapper * 2
    neg_sample.map_samples(mapper)
    expected = mapper[samples.cpu()].to(device)
    assert torch.equal(neg_sample.samples(), expected)
