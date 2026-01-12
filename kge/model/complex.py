import torch
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from kge.util.triton_complex import (
    fused_complex_requested,
    complex_matmul,
    complex_score_spo_triton,
    has_triton_complex,
)
from kge.util.complex_reduce import (
    complex_score_po_reduce,
    complex_score_sp_reduce,
    complex_score_spo_reduce,
)


class ComplExScorer(RelationalScorer):
    r"""Implementation of the ComplEx KGE scorer with optional fused variant."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self._use_fused = fused_complex_requested()
        self._use_triton = has_triton_complex()
        if self.config is not None:
            self.config.log(
                "ComplEx fused path: "
                f"requested={self._use_fused} triton_available={self._use_triton}"
            )

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        if self._use_fused:
            try:
                return self._score_emb_complex(s_emb, p_emb, o_emb, combine)
            except Exception:
                self._use_fused = False

        n = p_emb.size(0)
        p_emb_re, p_emb_im = (t.contiguous() for t in p_emb.chunk(2, dim=1))
        o_emb_re, o_emb_im = (t.contiguous() for t in o_emb.chunk(2, dim=1))

        s_all = torch.cat((s_emb, s_emb), dim=1)  # re, im, re, im
        r_all = torch.cat((p_emb_re, p_emb, -p_emb_im), dim=1)  # re, re, im, -im
        o_all = torch.cat((o_emb, o_emb_im, o_emb_re), dim=1)  # re, im, im, re

        if combine == "spo":
            out = (s_all * o_all * r_all).sum(dim=1)
        elif combine == "sp_":
            out = (s_all * r_all).mm(o_all.transpose(0, 1))
        elif combine == "_po":
            out = (r_all * o_all).mm(s_all.transpose(0, 1))
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)

        return out.view(n, -1)

    @staticmethod
    def _as_complex(emb: torch.Tensor) -> torch.Tensor:
        re, im = emb.chunk(2, dim=-1)
        return torch.complex(re, im)

    def _score_emb_complex(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)
        s = self._as_complex(s_emb)
        p = self._as_complex(p_emb)
        o = self._as_complex(o_emb)

        if combine == "spo":
            if self._use_triton and s_emb.is_cuda:
                out = complex_score_spo_triton(s_emb, p_emb, o_emb).view(n, 1)
            else:
                out = torch.real((s * p * o.conj()).sum(dim=-1, keepdim=True))
        elif combine == "sp_":
            out = torch.real(complex_matmul(s * p, o, conj_right=True))
        elif combine == "_po":
            out = torch.real(complex_matmul(p * o.conj(), s, conj_right=False))
        else:
            raise ValueError('cannot handle combine="{}"'.format(combine))

        return out.view(n, -1)


class ComplEx(KgeModel):
    r"""Implementation of the ComplEx KGE model."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key=None,
        init_for_load_only=False,
        create_embedders=True,
        parameter_client=None,
        max_partition_entities=0,
    ):
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=ComplExScorer,
            configuration_key=configuration_key,
            init_for_load_only=init_for_load_only,
            create_embedders=create_embedders,
            parameter_client=parameter_client,
            max_partition_entities=max_partition_entities,
        )
        self._reduce_by_key = False
        self._cache_row_grads = False
        if config is not None:
            try:
                self._reduce_by_key = bool(config.get("complex.reduce_by_key"))
            except Exception:
                self._reduce_by_key = False
            try:
                reduce_by_key_row_grads = bool(
                    config.get("complex.reduce_by_key_row_grads")
                )
            except Exception:
                reduce_by_key_row_grads = False
            if reduce_by_key_row_grads and not self._reduce_by_key:
                if config.get("train.auto_correct"):
                    config.set(
                        "complex.reduce_by_key",
                        True,
                        overwrite=Config.Overwrite.Yes,
                        log=True,
                    )
                    self._reduce_by_key = True
                else:
                    config.log(
                        "complex.reduce_by_key_row_grads is enabled while "
                        "complex.reduce_by_key is disabled; row-grad cache will be "
                        "inactive."
                    )
            if self._reduce_by_key:
                if reduce_by_key_row_grads:
                    self._cache_row_grads = True
                else:
                    try:
                        opt_type = config.get("train.optimizer.default.type")
                    except Exception:
                        opt_type = None
                    if opt_type in ("dist_adagrad", "dist_rowadagrad"):
                        self._cache_row_grads = True

    def _reduce_by_key_enabled(self) -> bool:
        if not self._reduce_by_key:
            return False
        if not self.training or not torch.is_grad_enabled():
            return False
        s_embedder = self.get_s_embedder()
        p_embedder = self.get_p_embedder()
        o_embedder = self.get_o_embedder()
        if not getattr(s_embedder, "sparse", False):
            return False
        if not getattr(p_embedder, "sparse", False):
            return False
        if not getattr(o_embedder, "sparse", False):
            return False
        return True

    def score_spo(self, s: torch.Tensor, p: torch.Tensor, o: torch.Tensor, direction=None) -> torch.Tensor:
        if self._reduce_by_key_enabled():
            s_embedder = self.get_s_embedder()
            p_embedder = self.get_p_embedder()
            o_embedder = self.get_o_embedder()
            return complex_score_spo_reduce(
                s,
                p,
                o,
                s_embedder._embeddings.weight,
                p_embedder._embeddings.weight,
                o_embedder._embeddings.weight,
                s_embedder.dropout.p,
                p_embedder.dropout.p,
                o_embedder.dropout.p,
                self.training,
                self._cache_row_grads,
            )
        return super().score_spo(s, p, o, direction)

    def score_sp(self, s: torch.Tensor, p: torch.Tensor, o: torch.Tensor = None) -> torch.Tensor:
        if self._reduce_by_key_enabled() and o is not None:
            s_embedder = self.get_s_embedder()
            p_embedder = self.get_p_embedder()
            o_embedder = self.get_o_embedder()
            return complex_score_sp_reduce(
                s,
                p,
                o,
                s_embedder._embeddings.weight,
                p_embedder._embeddings.weight,
                o_embedder._embeddings.weight,
                s_embedder.dropout.p,
                p_embedder.dropout.p,
                o_embedder.dropout.p,
                self.training,
                self._cache_row_grads,
            )
        return super().score_sp(s, p, o)

    def score_po(self, p: torch.Tensor, o: torch.Tensor, s: torch.Tensor = None) -> torch.Tensor:
        if self._reduce_by_key_enabled() and s is not None:
            p_embedder = self.get_p_embedder()
            o_embedder = self.get_o_embedder()
            s_embedder = self.get_s_embedder()
            return complex_score_po_reduce(
                p,
                o,
                s,
                p_embedder._embeddings.weight,
                o_embedder._embeddings.weight,
                s_embedder._embeddings.weight,
                p_embedder.dropout.p,
                o_embedder.dropout.p,
                s_embedder.dropout.p,
                self.training,
                self._cache_row_grads,
            )
        return super().score_po(p, o, s)
