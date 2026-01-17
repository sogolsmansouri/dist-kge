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
        self._reduce_by_key = False
        self._cache_row_grads = False
        self._reduce_by_key_logged = False
        self._reduce_by_key_inactive_logged = False
        if self.config is not None:
            self.config.log(
                "ComplEx fused path: "
                f"requested={self._use_fused} triton_available={self._use_triton}"
            )
        # Enable reduce-by-key scoring when configured for this model instance.
        # Note: in distributed training, the scorer's configuration key is
        # `distributed_model.base_model`, so this picks up CLI overrides like
        # `--distributed_model.base_model.reduce_by_key true`.
        try:
            self._reduce_by_key = self._get_bool_option("reduce_by_key", default=False)
        except Exception:
            self._reduce_by_key = False
        try:
            self._cache_row_grads = self._get_bool_option(
                "reduce_by_key_row_grads", default=False
            )
        except Exception:
            self._cache_row_grads = False
        if self._cache_row_grads and not self._reduce_by_key:
            self._cache_row_grads = False

    def _maybe_log_reduce_by_key_inactive(self, reasons) -> None:
        if (
            not self._reduce_by_key
            or self._reduce_by_key_inactive_logged
            or self.config is None
        ):
            return
        if not reasons:
            return
        self.config.log(
            "ComplEx reduce_by_key configured but inactive: "
            + ", ".join(str(r) for r in reasons)
            + "."
        )
        self._reduce_by_key_inactive_logged = True

    def _get_bool_option(self, key: str, default: bool = False) -> bool:
        value = None
        try:
            value = self.get_option(key)
        except Exception:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(int(value))
        if isinstance(value, str):
            raw = value.strip().strip("'\"").lower()
            if raw in ("1", "true", "yes", "y", "on"):
                return True
            if raw in ("0", "false", "no", "n", "off", ""):
                return False
        return bool(value)

    def score_spo_indexed(
        self,
        s_idx: torch.Tensor,
        p_idx: torch.Tensor,
        o_idx: torch.Tensor,
        s_embedder,
        p_embedder,
        o_embedder,
        training: bool,
    ):
        if not self._reduce_by_key:
            return None
        reasons = []
        if not training:
            reasons.append("not training")
        if not torch.is_grad_enabled():
            reasons.append("grad disabled")
        if not getattr(s_embedder, "sparse", False):
            reasons.append("s_embedder.sparse=False")
        if not getattr(p_embedder, "sparse", False):
            reasons.append("p_embedder.sparse=False")
        if not getattr(o_embedder, "sparse", False):
            reasons.append("o_embedder.sparse=False")
        # Needs direct access to embedding weights.
        if not hasattr(s_embedder, "_embeddings") or not hasattr(
            getattr(s_embedder, "_embeddings", None), "weight"
        ):
            reasons.append("s_embedder missing _embeddings.weight")
        if not hasattr(p_embedder, "_embeddings") or not hasattr(
            getattr(p_embedder, "_embeddings", None), "weight"
        ):
            reasons.append("p_embedder missing _embeddings.weight")
        if not hasattr(o_embedder, "_embeddings") or not hasattr(
            getattr(o_embedder, "_embeddings", None), "weight"
        ):
            reasons.append("o_embedder missing _embeddings.weight")
        if reasons:
            self._maybe_log_reduce_by_key_inactive(reasons)
            return None
        if self.config is not None and not self._reduce_by_key_logged:
            self.config.log(
                "ComplEx reduce_by_key active "
                f"(cache_row_grads={self._cache_row_grads})."
            )
            self._reduce_by_key_logged = True
        return complex_score_spo_reduce(
            s_idx,
            p_idx,
            o_idx,
            s_embedder._embeddings.weight,
            p_embedder._embeddings.weight,
            o_embedder._embeddings.weight,
            s_embedder.dropout.p,
            p_embedder.dropout.p,
            o_embedder.dropout.p,
            training,
            self._cache_row_grads,
        )

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        if self._use_fused:
            try:
                return self._score_emb_complex(s_emb, p_emb, o_emb, combine)
            except Exception:
                self._use_fused = False

        n = p_emb.size(0)
        s_emb_re, s_emb_im = (t.contiguous() for t in s_emb.chunk(2, dim=1))
        p_emb_re, p_emb_im = (t.contiguous() for t in p_emb.chunk(2, dim=1))
        o_emb_re, o_emb_im = (t.contiguous() for t in o_emb.chunk(2, dim=1))

        if combine == "spo":
            a = s_emb_re * p_emb_re - s_emb_im * p_emb_im
            b = s_emb_re * p_emb_im + s_emb_im * p_emb_re
            out = (a * o_emb_re + b * o_emb_im).sum(dim=1, keepdim=True)
        elif combine == "sp_":
            a = s_emb_re * p_emb_re - s_emb_im * p_emb_im
            b = s_emb_re * p_emb_im + s_emb_im * p_emb_re
            out = a.mm(o_emb_re.transpose(0, 1))
            out.addmm_(b, o_emb_im.transpose(0, 1))
        elif combine == "_po":
            a = p_emb_re * o_emb_re + p_emb_im * o_emb_im
            b = p_emb_im * o_emb_re - p_emb_re * o_emb_im
            out = a.mm(s_emb_re.transpose(0, 1))
            out.addmm_(b, s_emb_im.transpose(0, 1), alpha=-1.0)
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
        self._reduce_by_key_logged = False
        self._reduce_by_key_inactive_logged = False
        if config is not None:
            # Resolve reduce-by-key options via the model's configuration key so
            # distributed base-model overrides are honored.
            try:
                self._reduce_by_key = self._get_bool_option(
                    "reduce_by_key", default=False
                )
            except Exception:
                self._reduce_by_key = False
            try:
                reduce_by_key_row_grads = self._get_bool_option(
                    "reduce_by_key_row_grads", default=False
                )
            except Exception:
                reduce_by_key_row_grads = False
            if reduce_by_key_row_grads and not self._reduce_by_key:
                if config.get("train.auto_correct"):
                    self.set_option(
                        "reduce_by_key",
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

    def _maybe_log_reduce_by_key_inactive(self, reasons) -> None:
        if (
            not self._reduce_by_key
            or self._reduce_by_key_inactive_logged
            or self.config is None
        ):
            return
        if not reasons:
            return
        self.config.log(
            "ComplEx reduce_by_key configured but inactive: "
            + ", ".join(str(r) for r in reasons)
            + "."
        )
        self._reduce_by_key_inactive_logged = True

    def _get_bool_option(self, key: str, default: bool = False) -> bool:
        value = None
        try:
            value = self.get_option(key)
        except Exception:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(int(value))
        if isinstance(value, str):
            raw = value.strip().strip("'\"").lower()
            if raw in ("1", "true", "yes", "y", "on"):
                return True
            if raw in ("0", "false", "no", "n", "off", ""):
                return False
        return bool(value)

    def _reduce_by_key_enabled(self) -> bool:
        if not self._reduce_by_key:
            return False
        reasons = []
        if not self.training:
            reasons.append("not training")
        if not torch.is_grad_enabled():
            reasons.append("grad disabled")
        s_embedder = self.get_s_embedder()
        p_embedder = self.get_p_embedder()
        o_embedder = self.get_o_embedder()
        if not getattr(s_embedder, "sparse", False):
            reasons.append("s_embedder.sparse=False")
        if not getattr(p_embedder, "sparse", False):
            reasons.append("p_embedder.sparse=False")
        if not getattr(o_embedder, "sparse", False):
            reasons.append("o_embedder.sparse=False")
        if not hasattr(s_embedder, "_embeddings") or not hasattr(
            getattr(s_embedder, "_embeddings", None), "weight"
        ):
            reasons.append("s_embedder missing _embeddings.weight")
        if not hasattr(p_embedder, "_embeddings") or not hasattr(
            getattr(p_embedder, "_embeddings", None), "weight"
        ):
            reasons.append("p_embedder missing _embeddings.weight")
        if not hasattr(o_embedder, "_embeddings") or not hasattr(
            getattr(o_embedder, "_embeddings", None), "weight"
        ):
            reasons.append("o_embedder missing _embeddings.weight")
        if reasons:
            self._maybe_log_reduce_by_key_inactive(reasons)
            return False
        if not self._reduce_by_key_logged and self.config is not None:
            self.config.log(
                "ComplEx reduce_by_key active "
                f"(cache_row_grads={self._cache_row_grads})."
            )
            self._reduce_by_key_logged = True
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
