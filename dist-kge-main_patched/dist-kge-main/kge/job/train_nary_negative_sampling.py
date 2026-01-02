import torch
import torch.nn.functional as F
import torch.utils.data
from typing import Dict, Any, Tuple

from kge.job import TrainingOrEvaluationJob
from kge.model.nary_tucker import NaryTuckerModel
from kge.util import KgeOptimizer
from kge.job.train import _generate_worker_init_fn


class TrainingJobNaryNegativeSampling(TrainingOrEvaluationJob):
    """Training job for n-ary knowledge graphs using the NaryTuckerModel."""

    def __init__(
        self,
        config,
        dataset,
        parent_job=None,
        model=None,
        forward_only=False,
        parameter_client=None,
        work_scheduler_client=None,
        init_for_load_only=False,
    ):
        super().__init__(config, dataset, parent_job)
        if not dataset.has_nary_facts():
            raise ValueError(
                "Dataset does not define n-ary facts. "
                "Set dataset.files.<split>.type to 'nary_facts'."
            )
        self.forward_only = forward_only
        self.device = config.get("job.device")
        self.batch_size = config.get("train.batch_size")
        self.max_epochs = config.get("train.max_epochs")
        self.epoch = 0
        self.train_split = config.get("train.split")
        self.num_negatives = config.get("nary_negative_sampling.num_negatives")
        self.corrupt_all_arguments = config.get(
            "nary_negative_sampling.corrupt_all_arguments"
        )
        if model is None:
            model = NaryTuckerModel(
                config, dataset, init_for_load_only=init_for_load_only
            )
        self.model = model.to(self.device)
        if not self.forward_only:
            self.model.train()
            self.optimizer = KgeOptimizer.create(config, self.model)
        else:
            self.optimizer = None
        self.pad_id = dataset.nary_pad_id()
        self.current_trace: Dict[str, Any] = dict()
        self.loader = None
        self._facts = None
        self._masks = None

    def _prepare(self):
        super()._prepare()
        facts, masks, _ = self.dataset.nary_split(self.train_split)
        self.model.ensure_max_arity(self.dataset.nary_max_arity())
        self._facts = facts
        self._masks = masks
        self.num_examples = len(facts)
        self.loader = torch.utils.data.DataLoader(
            torch.arange(self.num_examples),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _generate_negatives(
        self, arguments: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.num_negatives <= 0:
            return None, None
        batch, width = arguments.size()
        neg_args = arguments.unsqueeze(1).repeat(1, self.num_negatives, 1).clone()
        neg_mask = mask.unsqueeze(1).repeat(1, self.num_negatives, 1)
        num_entities = self.dataset.num_entities()
        device = arguments.device
        arange_neg = torch.arange(self.num_negatives, device=device)
        for row in range(batch):
            valid_slots = torch.nonzero(mask[row], as_tuple=False).view(-1)
            if valid_slots.numel() == 0:
                continue
            if self.corrupt_all_arguments and valid_slots.numel() > 0:
                repeats = (
                    self.num_negatives + valid_slots.numel() - 1
                ) // valid_slots.numel()
                slot_choices = valid_slots.repeat(repeats)[: self.num_negatives]
            else:
                choice_idx = torch.randint(
                    0, valid_slots.numel(), (self.num_negatives,), device=device
                )
                slot_choices = valid_slots[choice_idx]
            replacements = torch.randint(
                0, num_entities, (self.num_negatives,), device=device
            )
            neg_args[row, arange_neg, slot_choices] = replacements
        return neg_args, neg_mask

    def _process_batch(self, batch_indices: torch.Tensor) -> Dict[str, Any]:
        relations = self._facts[batch_indices, 0].to(
            self.device, non_blocking=True
        )
        arguments = self._facts[batch_indices, 1:].to(
            self.device, non_blocking=True
        )
        argument_mask = self._masks[batch_indices].to(
            self.device, non_blocking=True
        )
        result = {}
        pos_scores = self.model.score_facts(relations, arguments, argument_mask)
        pos_loss = F.softplus(-pos_scores).mean()
        loss = pos_loss
        if self.num_negatives > 0:
            neg_args, neg_mask = self._generate_negatives(arguments, argument_mask)
            neg_scores = self.model.score_facts(
                relations.unsqueeze(1).repeat(1, self.num_negatives).view(-1),
                neg_args.view(-1, arguments.size(1)),
                neg_mask.view(-1, argument_mask.size(1)),
            ).view(-1, self.num_negatives)
            neg_loss = F.softplus(neg_scores).mean()
            loss = loss + neg_loss
            result["avg_negative_score"] = neg_scores.mean().item()
        result["avg_positive_score"] = pos_scores.mean().item()
        result["loss"] = loss
        return result

    def _run(self):
        if self.forward_only:
            raise ValueError("Forward-only mode not supported for n-ary training.")
        traces = []
        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            for batch_indices in self.loader:
                batch_info = self._process_batch(batch_indices)
                loss = batch_info["loss"]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
                self.config.print(
                    (
                        f"\r{self.config.log_prefix}"
                        f"epoch {epoch+1}/{self.max_epochs} "
                        f"batch {num_batches}/{len(self.loader)} "
                        f"loss {loss.item():.4f}"
                    ),
                    end="",
                    flush=True,
                )
            avg_loss = epoch_loss / max(1, num_batches)
            trace_entry = dict(
                type="nary_negative_sampling",
                scope="epoch",
                epoch=epoch + 1,
                split=self.train_split,
                avg_loss=avg_loss,
                size=self.num_examples,
            )
            traces.append(trace_entry)
            self.trace(event="train_epoch", **trace_entry)
            self.config.log(
                f"Epoch {epoch+1}/{self.max_epochs}: avg_loss={avg_loss:.4f}"
            )
        self.config.print("")
        return traces
