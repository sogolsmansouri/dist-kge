import torch
from torch import Tensor
import torch.nn.functional as F

from kge import Config, Dataset, Configurable
from kge.model import KgeEmbedder

from typing import Tuple


class NaryTuckerModel(torch.nn.Module, Configurable):
    """Tucker-style model for n-ary facts with shared projections per argument slot."""

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str = None,
        init_for_load_only: bool = False,
    ):
        if configuration_key is None:
            configuration_key = config.get("model")
        Configurable.__init__(self, config, configuration_key)
        torch.nn.Module.__init__(self)

        self.dataset = dataset
        self.max_arity = max(
            dataset.nary_max_arity(), self.get_option("max_supported_arity")
        )
        if self.max_arity <= 0:
            raise ValueError(
                "Dataset does not expose n-ary facts. "
                "Ensure dataset.files.*.type is set to 'nary_facts'."
            )
        self.rank = self.get_option("rank")
        self.dropout = self.get_option("dropout")

        self._entity_embedder = KgeEmbedder.create(
            config,
            dataset,
            self.configuration_key + ".entity_embedder",
            dataset.num_entities(),
            init_for_load_only=init_for_load_only,
        )
        self._relation_embedder = KgeEmbedder.create(
            config,
            dataset,
            self.configuration_key + ".relation_embedder",
            dataset.num_relations(),
            init_for_load_only=init_for_load_only,
        )
        self.relation_projection = torch.nn.Linear(
            self._relation_embedder.dim, self.rank, bias=False
        )
        self.argument_projections = torch.nn.ModuleList(
            [
                torch.nn.Linear(self._entity_embedder.dim, self.rank, bias=False)
                for _ in range(self.max_arity)
            ]
        )
        if not init_for_load_only:
            torch.nn.init.xavier_normal_(self.relation_projection.weight)
            for layer in self.argument_projections:
                torch.nn.init.xavier_normal_(layer.weight)

    def parameters(self, recurse: bool = True):
        return super().parameters(recurse=recurse)

    def get_entity_embedder(self):
        return self._entity_embedder

    def get_relation_embedder(self):
        return self._relation_embedder

    def _project_relations(self, relation_ids: Tensor) -> Tensor:
        rel_emb = self._relation_embedder.embed(relation_ids)
        rel_proj = self.relation_projection(rel_emb)
        if self.dropout > 0:
            rel_proj = F.dropout(rel_proj, p=self.dropout, training=self.training)
        return rel_proj

    def _project_argument(
        self, argument_ids: Tensor, position: int, mask: Tensor
    ) -> Tensor:
        safe_ids = argument_ids.clone()
        if mask is not None:
            safe_ids = torch.where(mask, safe_ids, torch.zeros_like(safe_ids))
        arg_emb = self._entity_embedder.embed(safe_ids)
        proj = self.argument_projections[position](arg_emb)
        if mask is not None:
            proj = torch.where(mask.unsqueeze(-1), proj, torch.ones_like(proj))
        return proj

    def score_facts(
        self, relation_ids: Tensor, argument_ids: Tensor, argument_mask: Tensor
    ) -> Tensor:
        """Score n-ary facts.

        Args:
            relation_ids: Tensor of shape [batch]
            argument_ids: Tensor of shape [batch, max_arity]
            argument_mask: Bool tensor of shape [batch, max_arity]
        """
        batch = relation_ids.shape[0]
        if argument_ids.size(1) != self.max_arity:
            pad_cols = self.max_arity - argument_ids.size(1)
            if pad_cols < 0:
                raise ValueError(
                    f"argument_ids has width {argument_ids.size(1)} "
                    f"but model max_arity is {self.max_arity}"
                )
            if pad_cols > 0:
                pad_tensor = argument_ids.new_full((batch, pad_cols), 0)
                argument_ids = torch.cat((argument_ids, pad_tensor), dim=1)
                mask_pad = argument_mask.new_zeros((batch, pad_cols))
                argument_mask = torch.cat((argument_mask, mask_pad), dim=1)

        score_vec = self._project_relations(relation_ids)
        for pos in range(self.max_arity):
            mask = argument_mask[:, pos]
            if not mask.any():
                continue
            slot_scores = self._project_argument(
                argument_ids[:, pos], pos, mask
            )
            score_vec = score_vec * slot_scores
        return score_vec.sum(dim=-1)

    def save(self):
        return self.state_dict()

    def load(self, state_dict):
        self.load_state_dict(state_dict)

    def ensure_max_arity(self, target_arity: int):
        if target_arity <= self.max_arity:
            return
        for _ in range(self.max_arity, target_arity):
            layer = torch.nn.Linear(self._entity_embedder.dim, self.rank, bias=False)
            torch.nn.init.xavier_normal_(layer.weight)
            self.argument_projections.append(layer)
        self.max_arity = target_arity
