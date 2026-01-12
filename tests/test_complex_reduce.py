import tempfile
import unittest

import torch

from tests.util import create_config, get_dataset_folder
from kge import Dataset
from kge.job.train import TrainingJob
from kge.model import KgeModel
from kge.util.complex_reduce import (
    complex_score_po_reduce,
    complex_score_sp_reduce,
    complex_score_spo_reduce,
)


def _split_complex(emb: torch.Tensor):
    return emb.chunk(2, dim=1)


def _score_spo_baseline(s_emb, p_emb, o_emb):
    p_re, p_im = _split_complex(p_emb)
    o_re, o_im = _split_complex(o_emb)
    s_all = torch.cat((s_emb, s_emb), dim=1)
    r_all = torch.cat((p_re, p_emb, -p_im), dim=1)
    o_all = torch.cat((o_emb, o_im, o_re), dim=1)
    return (s_all * r_all * o_all).sum(dim=1)


def _score_sp_baseline(s_emb, p_emb, o_emb):
    p_re, p_im = _split_complex(p_emb)
    o_re, o_im = _split_complex(o_emb)
    s_all = torch.cat((s_emb, s_emb), dim=1)
    r_all = torch.cat((p_re, p_emb, -p_im), dim=1)
    o_all = torch.cat((o_emb, o_im, o_re), dim=1)
    return (s_all * r_all) @ o_all.transpose(0, 1)


def _score_po_baseline(p_emb, o_emb, s_emb):
    p_re, p_im = _split_complex(p_emb)
    o_re, o_im = _split_complex(o_emb)
    r_all = torch.cat((p_re, p_emb, -p_im), dim=1)
    o_all = torch.cat((o_emb, o_im, o_re), dim=1)
    s_all = torch.cat((s_emb, s_emb), dim=1)
    return (r_all * o_all) @ s_all.transpose(0, 1)


class TestComplexReduceByKey(unittest.TestCase):
    def _dense(self, grad: torch.Tensor) -> torch.Tensor:
        if grad.is_sparse:
            return grad.coalesce().to_dense()
        return grad

    def test_reduce_by_key_matches_baseline(self):
        torch.manual_seed(0)
        num_entities = 4
        num_relations = 3
        dim = 16

        s_weight = torch.randn(num_entities, dim, requires_grad=True)
        p_weight = torch.randn(num_relations, dim, requires_grad=True)
        o_weight = torch.randn(num_entities, dim, requires_grad=True)

        s_weight_r = s_weight.detach().clone().requires_grad_(True)
        p_weight_r = p_weight.detach().clone().requires_grad_(True)
        o_weight_r = o_weight.detach().clone().requires_grad_(True)

        s = torch.tensor([0, 1, 0, 2, 1, 1], dtype=torch.long)
        p = torch.tensor([0, 1, 1, 0, 0, 2], dtype=torch.long)
        o = torch.tensor([1, 2, 1, 0, 2, 2], dtype=torch.long)
        o_targets = torch.tensor([0, 1, 2], dtype=torch.long)
        s_targets = torch.tensor([0, 1], dtype=torch.long)

        s_emb = s_weight.index_select(0, s)
        p_emb = p_weight.index_select(0, p)
        o_emb = o_weight.index_select(0, o)
        o_emb_targets = o_weight.index_select(0, o_targets)
        s_emb_targets = s_weight.index_select(0, s_targets)

        score_spo_base = _score_spo_baseline(s_emb, p_emb, o_emb)
        score_sp_base = _score_sp_baseline(s_emb, p_emb, o_emb_targets)
        score_po_base = _score_po_baseline(p_emb, o_emb, s_emb_targets)

        score_spo_reduce = complex_score_spo_reduce(
            s, p, o, s_weight_r, p_weight_r, o_weight_r, 0.0, 0.0, 0.0, True
        )
        score_sp_reduce = complex_score_sp_reduce(
            s,
            p,
            o_targets,
            s_weight_r,
            p_weight_r,
            o_weight_r,
            0.0,
            0.0,
            0.0,
            True,
        )
        score_po_reduce = complex_score_po_reduce(
            p,
            o,
            s_targets,
            p_weight_r,
            o_weight_r,
            s_weight_r,
            0.0,
            0.0,
            0.0,
            True,
        )

        self.assertTrue(
            torch.allclose(score_spo_base, score_spo_reduce, atol=1e-5, rtol=1e-4)
        )
        self.assertTrue(
            torch.allclose(score_sp_base, score_sp_reduce, atol=1e-5, rtol=1e-4)
        )
        self.assertTrue(
            torch.allclose(score_po_base, score_po_reduce, atol=1e-5, rtol=1e-4)
        )

        loss_base = score_spo_base.sum() + score_sp_base.sum() + score_po_base.sum()
        loss_reduce = (
            score_spo_reduce.sum() + score_sp_reduce.sum() + score_po_reduce.sum()
        )
        loss_base.backward()
        loss_reduce.backward()

        self.assertTrue(
            torch.allclose(
                self._dense(s_weight.grad),
                self._dense(s_weight_r.grad),
                atol=1e-5,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            torch.allclose(
                self._dense(p_weight.grad),
                self._dense(p_weight_r.grad),
                atol=1e-5,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            torch.allclose(
                self._dense(o_weight.grad),
                self._dense(o_weight_r.grad),
                atol=1e-5,
                rtol=1e-4,
            )
        )


class TestComplexReduceByKeyModel(unittest.TestCase):
    def _dense(self, grad: torch.Tensor) -> torch.Tensor:
        if grad.is_sparse:
            return grad.coalesce().to_dense()
        return grad

    def test_reduce_by_key_matches_model(self):
        torch.manual_seed(0)
        config = create_config("dataset_test", model="complex")
        config.set_all(
            {
                "lookup_embedder.dim": 16,
                "lookup_embedder.sparse": True,
                "lookup_embedder.dropout": 0.0,
                "complex.reduce_by_key": False,
            }
        )
        dataset = Dataset.create(config, folder=get_dataset_folder("dataset_test"))

        model_base = KgeModel.create(config, dataset)
        config_reduce = config.clone()
        config_reduce.set("complex.reduce_by_key", True)
        model_reduce = KgeModel.create(config_reduce, dataset)
        model_reduce.load_state_dict(model_base.state_dict())

        model_base.train()
        model_reduce.train()

        triples = dataset.split("train")
        subset = triples[: min(6, triples.size(0))]
        s = subset[:, 0]
        p = subset[:, 1]
        o = subset[:, 2]
        num_entities = dataset.num_entities()
        o_targets = torch.arange(min(4, num_entities))
        s_targets = torch.arange(min(3, num_entities))

        score_spo_base = model_base.score_spo(s, p, o)
        score_sp_base = model_base.score_sp(s, p, o_targets)
        score_po_base = model_base.score_po(p, o, s_targets)

        score_spo_reduce = model_reduce.score_spo(s, p, o)
        score_sp_reduce = model_reduce.score_sp(s, p, o_targets)
        score_po_reduce = model_reduce.score_po(p, o, s_targets)

        self.assertTrue(
            torch.allclose(score_spo_base, score_spo_reduce, atol=1e-5, rtol=1e-4)
        )
        self.assertTrue(
            torch.allclose(score_sp_base, score_sp_reduce, atol=1e-5, rtol=1e-4)
        )
        self.assertTrue(
            torch.allclose(score_po_base, score_po_reduce, atol=1e-5, rtol=1e-4)
        )

        model_base.zero_grad(set_to_none=True)
        model_reduce.zero_grad(set_to_none=True)
        loss_base = score_spo_base.sum() + score_sp_base.sum() + score_po_base.sum()
        loss_reduce = (
            score_spo_reduce.sum() + score_sp_reduce.sum() + score_po_reduce.sum()
        )
        loss_base.backward()
        loss_reduce.backward()

        s_grad_base = model_base.get_s_embedder()._embeddings.weight.grad
        s_grad_reduce = model_reduce.get_s_embedder()._embeddings.weight.grad
        p_grad_base = model_base.get_p_embedder()._embeddings.weight.grad
        p_grad_reduce = model_reduce.get_p_embedder()._embeddings.weight.grad
        o_grad_base = model_base.get_o_embedder()._embeddings.weight.grad
        o_grad_reduce = model_reduce.get_o_embedder()._embeddings.weight.grad

        self.assertTrue(
            torch.allclose(
                self._dense(s_grad_base),
                self._dense(s_grad_reduce),
                atol=1e-5,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            torch.allclose(
                self._dense(p_grad_base),
                self._dense(p_grad_reduce),
                atol=1e-5,
                rtol=1e-4,
            )
        )
        self.assertTrue(
            torch.allclose(
                self._dense(o_grad_base),
                self._dense(o_grad_reduce),
                atol=1e-5,
                rtol=1e-4,
            )
        )


class TestComplexReduceByKeyTraining(unittest.TestCase):
    def test_training_one_epoch(self):
        config = create_config("dataset_test", model="complex")
        config.set("job.type", "train")
        config.set("train.type", "negative_sampling")
        config.set_all(
            {
                "lookup_embedder.dim": 16,
                "lookup_embedder.sparse": True,
                "lookup_embedder.dropout": 0.0,
                "train.batch_size": 4,
                "train.num_workers": 0,
                "negative_sampling.implementation": "triple",
                "complex.reduce_by_key": True,
            }
        )
        dataset = Dataset.create(config, folder=get_dataset_folder("dataset_test"))

        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = config.clone()
            run_config.folder = tmpdir
            job = TrainingJob.create(run_config, dataset)
            job._prepare()
            trace = job.run_epoch()

        avg_loss = trace.get("avg_loss")
        self.assertIsNotNone(avg_loss)
        self.assertTrue(torch.isfinite(torch.tensor(avg_loss)))
