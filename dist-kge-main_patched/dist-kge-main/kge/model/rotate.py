import torch
import math
from kge import Config, Dataset
from kge.job import Job
from kge.model.kge_model import RelationalScorer, KgeModel
from torch.nn import functional as F


# TODO sp_ and _po scoring with RotatE leads to *large* intermediate results. It's
# unclear whether this can be fixed. Expect out-of-memory errors when using RotatE with
# 1vsAll or KvsAll training. To do validation/evaluation, you may want to set
# eval.chunk_size.
class RotatEScorer(RelationalScorer):
    r"""Implementation of the RotatE KGE scorer."""

    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self._norm = self.get_option("l_norm")

    def _as_complex_entity(self, emb: torch.Tensor) -> torch.Tensor:
        """Reshape entity embeddings into complex tensors without copying."""
        new_shape = emb.shape[:-1] + (-1, 2)
        return torch.view_as_complex(emb.reshape(new_shape))

    def _relation_to_complex(self, p_emb: torch.Tensor) -> torch.Tensor:
        """Interpret relation embeddings as rotation phases on the complex unit circle."""
        return torch.polar(torch.ones_like(p_emb), p_emb)

    def score_emb(self, s_emb, p_emb, o_emb, combine: str):
        n = p_emb.size(0)
        s_complex = self._as_complex_entity(s_emb)
        o_complex = self._as_complex_entity(o_emb)
        p_complex = self._relation_to_complex(p_emb)

        if combine == "spo":
            diff = s_complex * p_complex - o_complex
            out = -torch.linalg.vector_norm(diff.abs(), ord=self._norm, dim=1)
        elif combine == "sp_":
            sp = s_complex * p_complex
            diff = sp.unsqueeze(1) - o_complex.unsqueeze(0)
            out = -torch.linalg.vector_norm(diff.abs(), ord=self._norm, dim=2)
        elif combine == "_po":
            po = torch.conj(p_complex) * o_complex
            diff = po.unsqueeze(1) - s_complex.unsqueeze(0)
            out = -torch.linalg.vector_norm(diff.abs(), ord=self._norm, dim=2)
        else:
            return super().score_emb(s_emb, p_emb, o_emb, combine)

        return out.view(n, -1)


class RotatE(KgeModel):
    r"""Implementation of the RotatE KGE model."""

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
        self._init_configuration(config, configuration_key)
        if self.get_option("entity_embedder.dim") % 2 != 0:
            raise ValueError(
                "RotatE requires embeddings of even dimensionality"
                " (got {})".format(self.get_option("entity_embedder.dim"))
            )
        if self.get_option("relation_embedder.dim") < 0:
            self.set_option(
                "relation_embedder.dim",
                self.get_option("entity_embedder.dim") // 2,
                log=True,
            )
        super().__init__(
            config=config,
            dataset=dataset,
            scorer=RotatEScorer,
            configuration_key=self.configuration_key,
            init_for_load_only=init_for_load_only,
            create_embedders=create_embedders,
            parameter_client=parameter_client,
            max_partition_entities=max_partition_entities,
        )
        self._normalize_phases = self.get_option("normalize_phases")

    @torch.no_grad()
    def normalize_phases(self):
        out = self.get_p_embedder()._embeddings.weight.data

        # normalize phases so that they lie in [-pi,pi]
        # TODO this is a hack that assumes that we use a lookup embedder

        # first shift phases by pi
        out = out + math.pi

        # compute the modulo (result then in [0,2*pi))
        out = torch.remainder(out, 2.0 * math.pi)

        # shift back
        out = out - math.pi

        # write back the updated embeddings
        self.get_p_embedder()._embeddings.weight.data[:] = out[:]

    def prepare_job(self, job: Job, **kwargs):
        from kge.job import TrainingJob

        super().prepare_job(job, **kwargs)

        if self._normalize_phases and isinstance(job, TrainingJob):
            from kge.model import LookupEmbedder

            if not isinstance(self.get_p_embedder(), LookupEmbedder):
                raise ValueError(
                    "RotatE currently supports normalize_phases=True "
                    "only when a lookup embedder is used for relations; "
                    "current relation embedder is "
                    f"{self.get_option('relation_embedder.type')} "
                    "however"
                )

            # just to be sure it's right initially
            job.pre_run_hooks.append(lambda job: self.normalize_phases())

            # normalize after each batch
            job.post_batch_hooks.append(lambda job: self.normalize_phases())
