import warnings
from kge import Config, Configurable, Dataset

import random
import torch
from typing import Optional
import numpy as np
import numba
import time

SLOTS = [0, 1, 2]
SLOT_STR = ["s", "p", "o"]
S, P, O = SLOTS


class KgeSampler(Configurable):
    """Negative sampler. """

    def __init__(self, config: Config, configuration_key: str, dataset: Dataset):
        super().__init__(config, configuration_key)

        # load config
        self.num_samples = torch.zeros(3, dtype=torch.int)
        self.filter_positives = torch.zeros(3, dtype=torch.bool)
        self.vocabulary_size = torch.zeros(3, dtype=torch.int)
        self.shared = self.get_option("shared")
        self.shared_type = self.check_option("shared_type", ["naive", "default"])
        self.with_replacement = self.get_option("with_replacement")
        if not self.with_replacement and not self.shared:
            raise ValueError(
                "Without replacement sampling is only supported when "
                "shared negative sampling is enabled."
            )
        self.filtering_split = config.get("negative_sampling.filtering.split")
        if self.filtering_split == "":
            self.filtering_split = config.get("train.split")
        for slot in SLOTS:
            slot_str = SLOT_STR[slot]
            self.num_samples[slot] = self.get_option(f"num_samples.{slot_str}")
            self.filter_positives[slot] = self.get_option(f"filtering.{slot_str}")
            self.vocabulary_size[slot] = (
                dataset.num_relations() if slot == P else dataset.num_entities()
            )
            # create indices for filtering here already if needed and not existing
            # otherwise every worker would create every index again and again
            if self.filter_positives[slot]:
                pair = ["po", "so", "sp"][slot]
                dataset.index(f"{self.filtering_split}_{pair}_to_{slot_str}")
        if any(self.filter_positives):
            if self.shared:
                raise ValueError(
                    "Filtering is not supported when shared negative sampling is enabled."
                )
            self.filter_implementation = self.check_option(
                "filtering.implementation", ["standard", "fast", "fast_if_available"]
            )
        self.dataset = dataset
        # auto config
        for slot, copy_from in [(S, O), (P, None), (O, S)]:
            if self.num_samples[slot] < 0:
                if copy_from is not None and self.num_samples[copy_from] > 0:
                    self.num_samples[slot] = self.num_samples[copy_from]
                else:
                    self.num_samples[slot] = 0

    def supports_device_sampling(self, positive_triples: torch.Tensor) -> bool:
        """Returns True if the sampler can operate directly on the triples' device."""
        return False

    def uses_pool(self) -> bool:
        """Returns True if the sampler relies on a resident entity pool."""
        return False

    @staticmethod
    def create(
        config: Config, configuration_key: str, dataset: Dataset
    ) -> "KgeSampler":
        if config.get(configuration_key + ".combined"):
            return KgeCombinedSampler(config, configuration_key, dataset)
        else:
            return KgeSampler._create(config, configuration_key, dataset)

    @staticmethod
    def _create(
        config: Config, configuration_key: str, dataset: Dataset
    ) -> "KgeSampler":
        """Factory method for sampler creation."""
        sampling_type = config.get(configuration_key + ".sampling_type")
        if sampling_type == "uniform":
            return KgeUniformSampler(config, configuration_key, dataset)
        elif sampling_type == "frequency":
            return KgeFrequencySampler(config, configuration_key, dataset)
        elif sampling_type == "hfrequency":
            return KgeHierarchicalFrequencySampler(config, configuration_key, dataset)
        elif sampling_type == "pooled":
            return KgePooledSampler(config, configuration_key, dataset)
        elif sampling_type == "batch":
            return KgeBatchSampler(config, configuration_key, dataset)
        elif sampling_type == "cache_aware":
            return KgeCacheAwareSampler(config, configuration_key, dataset)
        else:
            # perhaps TODO: try class with specified name -> extensibility
            raise ValueError(configuration_key + ".sampling_type")

    def sample(
        self,
        positive_triples: torch.Tensor,
        slot: int,
        num_samples: Optional[int] = None,
    ) -> "BatchNegativeSample":
        """Obtain a set of negative samples for a specified slot.

        `positive_triples` is a batch_size x 3 tensor of positive triples. `slot` is
        either 0 (subject), 1 (predicate), or 2 (object). If `num_samples` is `None`,
        it is set to the default value for the slot configured in this sampler.

        Returns a `BatchNegativeSample` data structure that allows to retrieve or score
        all negative samples. In the simplest setting, this data structure holds a
        batch_size x num_samples tensor with the negative sample indexes (see
        `DefaultBatchNegativeSample`), but more efficient approaches may be used by
        certain samplers.

        """
        if num_samples is None:
            num_samples = self.num_samples[slot].item()

        if self.shared:
            # for shared sampling, we do not post-process; return right away
            return self._sample_shared(positive_triples, slot, num_samples)
        else:
            negative_samples = self._sample(positive_triples, slot, num_samples)

        # for non-shared smaples, we filter the positives (if set in config)
        if self.filter_positives[slot]:
            if self.filter_implementation == "fast":
                negative_samples = self._filter_and_resample_fast(
                    negative_samples, slot, positive_triples
                )
            elif self.filter_implementation == "standard":
                negative_samples = self._filter_and_resample(
                    negative_samples, slot, positive_triples
                )
            else:  # fast_if_available
                try:
                    negative_samples = self._filter_and_resample_fast(
                        negative_samples, slot, positive_triples
                    )
                    self.filter_implementation = "fast"
                except NotImplementedError:
                    negative_samples = self._filter_and_resample(
                        negative_samples, slot, positive_triples
                    )
                    self.filter_implementation = "standard"

        return DefaultBatchNegativeSample(
            self.config,
            self.configuration_key,
            positive_triples,
            slot,
            num_samples,
            negative_samples,
        )

    def _sample(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ) -> torch.Tensor:
        """Sample negative examples.

        This methods returns a tensor of size batch_size x num_samples holding the
        indexes for the sample. The method is also used to resample filtered positives.

        """
        raise NotImplementedError("The selected sampler is not implemented.")

    def _sample_shared(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ) -> "BatchNegativeSample":
        """Sample negative examples with sharing.

        This methods directly returns a BatchNegativeSample data structure for
        efficiency.

        """
        raise NotImplementedError(
            "The selected sampler does not support shared negative samples."
        )

    def _filter_and_resample(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ) -> torch.Tensor:
        """Filter and resample indices until only negatives have been created. """
        pair_str = ["po", "so", "sp"][slot]
        # holding the positive indices for the respective pair
        index = self.dataset.index(
            f"{self.filtering_split}_{pair_str}_to_{SLOT_STR[slot]}"
        )
        cols = [[P, O], [S, O], [S, P]][slot]
        pairs = positive_triples[:, cols]
        if pairs.device.type != "cpu":
            pairs = pairs.cpu()
        target_device = negative_samples.device
        for i in range(positive_triples.size(0)):
            positives = index.get((pairs[i][0].item(), pairs[i][1].item()))
            if isinstance(positives, list):
                if not positives:
                    continue
                positives = torch.tensor(positives, device=target_device)
            else:
                if positives.numel() == 0:
                    continue
                if positives.device != target_device:
                    positives = positives.to(target_device)
            if positives.dtype != torch.long:
                positives = positives.long()
            row = negative_samples[i]
            # indices of samples that have to be sampled again
            resample_mask = torch.isin(row, positives)
            if not resample_mask.any():
                continue
            resample_idx = resample_mask.nonzero(as_tuple=False).view(-1)
            # number of new samples needed
            num_new = resample_idx.numel()
            # number already found of the new samples needed
            num_found = 0
            num_remaining = num_new - num_found
            while num_remaining:
                new_samples = self._sample(
                    positive_triples[i, None], slot, num_remaining
                ).view(-1)
                if new_samples.device != target_device:
                    new_samples = new_samples.to(target_device)
                if new_samples.dtype != positives.dtype:
                    new_samples = new_samples.to(positives.dtype)
                # indices of the true negatives
                tn_mask = ~torch.isin(new_samples, positives)
                # write the true negatives found
                if tn_mask.any():
                    tn = new_samples[tn_mask]
                    take = min(tn.numel(), num_remaining)
                    row[resample_idx[num_found : num_found + take]] = tn[:take]
                    num_found += take
                    num_remaining = num_new - num_found
        return negative_samples

    def _filter_and_resample_fast(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ) -> torch.Tensor:
        """Filter and resample indices.

        Samplers can override this method when their sampling strategy allows for a
        more efficient filtering method than the generic standard method or when their
        code can be optimized by tools such as Numba.

        """
        raise NotImplementedError(
            "Use filtering.implementation=standard for this sampler."
        )


class BatchNegativeSample(Configurable):
    """Abstract superclass for a negative sample of a batch.

    Provides methods to access the negative samples and to score them using a model.
    """

    def __init__(
        self,
        config: Config,
        configuration_key: str,
        positive_triples: torch.Tensor,
        slot: int,
        num_samples: int,
    ):
        super().__init__(config, configuration_key)
        self.positive_triples = positive_triples
        self.slot = slot
        self.num_samples = num_samples
        self._implementation = self.check_option(
            "implementation", ["triple", "batch", "all"]
        )
        self.forward_time = 0.0
        self.prepare_time = 0.0
        self._lookahead_payload = None

    def samples(self, indexes=None) -> torch.Tensor:
        """Returns a tensor holding the indexes of the negative samples.

        If `indexes` is provided, only score the corresponding subset of the batch.

        Returns a chunk_size x num_samples tensor of indexes. Here chunk_size corresponds
        the batch size (if `indexes=None`) or to the number of specified indexes (otherwise).
        """
        raise NotImplementedError

    def unique_samples(self, indexes=None, return_inverse=False, remove_dropped=True):
        """Returns the unique negative samples.

        If `indexes` is provided, only consider the corresponding subset of the batch.
        Optionally, also returns the indexes of each unqiue sample in the flattened
        negative-sampling tensor (i.e., in `self.samples(indexes).view(-1)`).

        """
        samples = self.samples(indexes)
        flat = samples.view(-1)
        if remove_dropped and not return_inverse:
            # Some sampling/mapping paths can mark invalid samples with negative ids
            # (e.g., -1). These must never reach prefetch/pull/indexing paths.
            #
            # IMPORTANT: do NOT filter when return_inverse=True, because that would
            # make the returned inverse mapping inconsistent with the original
            # samples tensor.
            valid = flat >= 0
            if not torch.all(valid):
                flat = flat[valid]
        return torch.unique(flat, return_inverse=return_inverse)

    def to(self, device, non_blocking: bool = False) -> "BatchNegativeSample":
        """Move the negative samples to the specified device."""
        self.positive_triples = self.positive_triples.to(
            device, non_blocking=non_blocking
        )
        return self

    def pin_memory(self) -> "BatchNegativeSample":
        """Pin underlying CPU buffers to enable non-blocking transfers."""
        if (
            self.positive_triples.device.type == "cpu"
            and not self.positive_triples.is_pinned()
        ):
            self.positive_triples = self.positive_triples.pin_memory()
        return self

    @staticmethod
    def _device_key(device) -> str:
        if isinstance(device, torch.device):
            if device.index is None:
                return device.type
            return f"{device.type}:{device.index}"
        return str(device)

    def attach_lookahead(self, payload: Optional[dict]):
        """Attach prefetched buffers that can be consumed later."""
        if payload is None:
            return
        self._lookahead_payload = payload

    def _consume_lookahead(self, device):
        if self._lookahead_payload is None:
            return None
        payload = self._lookahead_payload
        device_key = payload.get("device")
        if device_key is not None and device_key != self._device_key(device):
            return None
        self._lookahead_payload = None
        return payload

    def build_lookahead_payload(self, device, non_blocking: bool = False):
        """Create a device-resident copy of the samples for reuse."""
        try:
            samples = self.samples().to(device, non_blocking=non_blocking)
        except Exception:
            return None
        return {"device": self._device_key(device), "samples": samples}

    def map_samples(self, mapper):
        """Maps samples to new ids"""
        raise NotImplementedError

    def score(self, model, indexes=None) -> torch.Tensor:
        """Score the negative samples for the batch with the provided model.

        If `indexes` is provided, only score the corresponding subset of the batch.

        Returns a chunk_size x num_samples tensor of scores. Here chunk_size corresponds
        the batch size (if `indexes=None`) or to the number of specified indexes (otherwise).

        Sets the `forward_time` and `prepare_time` attributes.
        """
        self.forward_time = 0.0
        self.prepare_time = 0.0

        # the default implementation here is based on the set of all samples as provided
        # by self.samples(); get the relevant data
        slot = self.slot
        self.prepare_time -= time.time()
        negative_samples = self.samples(indexes)
        num_samples = self.num_samples
        triples = (
            self.positive_triples[indexes, :] if indexes else self.positive_triples
        )
        self.prepare_time += time.time()

        # go ahead and score
        device = self.positive_triples.device
        chunk_size = len(negative_samples)
        scores = None
        if self._implementation == "triple":
            # construct triples
            self.prepare_time -= time.time()
            triples_to_score = triples.repeat(1, num_samples).view(-1, 3)
            triples_to_score[:, slot] = negative_samples.contiguous().view(-1)
            self.prepare_time += time.time()

            # and score them
            self.forward_time -= time.time()
            scores = model.score_spo(
                triples_to_score[:, S],
                triples_to_score[:, P],
                triples_to_score[:, O],
                direction=SLOT_STR[slot],
            ).view(chunk_size, -1)
            self.forward_time += time.time()
        elif self._implementation in ["batch", "all"]:
            # Score each triples against all unique possible targets, then pick out the
            # actual scores.
            self.prepare_time -= time.time()
            if self._implementation == "all":
                unique_targets = None  # means all
                column_indexes = negative_samples.contiguous().view(-1)
            else:
                unique_targets, column_indexes = self.unique_samples(
                    indexes, return_inverse=True
                )
            self.prepare_time += time.time()

            # compute all scores for slot
            self.forward_time -= time.time()
            use_nvtx = False
            try:
                use_nvtx = bool(self.config.get("train.profile_nvtx")) and device.type == "cuda"
            except Exception:
                use_nvtx = False
            if use_nvtx:
                try:
                    torch.cuda.nvtx.range_push(f"ns/{SLOT_STR[slot]}/score_unique_targets")
                except Exception:
                    use_nvtx = False
            try:
                all_scores = self._score_unique_targets(
                    model, slot, triples, unique_targets
                )
            finally:
                if use_nvtx:
                    try:
                        torch.cuda.nvtx.range_pop()
                    except Exception:
                        pass
            self.forward_time += time.time()

            # determine indexes of relevant scores in scoring matrix
            # and pick the scores we need (avoid advanced indexing; use gather)
            self.forward_time -= time.time()
            if use_nvtx:
                try:
                    torch.cuda.nvtx.range_push(f"ns/{SLOT_STR[slot]}/gather_scores")
                except Exception:
                    use_nvtx = False
            try:
                if column_indexes.dtype != torch.long:
                    column_indexes = column_indexes.long()
                column_indexes_2d = column_indexes.view(chunk_size, num_samples)
                scores = all_scores.gather(1, column_indexes_2d)
            finally:
                if use_nvtx:
                    try:
                        torch.cuda.nvtx.range_pop()
                    except Exception:
                        pass
            self.forward_time += time.time()
        else:
            raise ValueError

        return scores

    @staticmethod
    def _score_unique_targets(model, slot, triples, unique_targets) -> torch.Tensor:
        if slot == S:
            all_scores = model.score_po(triples[:, P], triples[:, O], unique_targets)
        elif slot == P:
            all_scores = model.score_so(triples[:, S], triples[:, O], unique_targets)
        elif slot == O:
            all_scores = model.score_sp(triples[:, S], triples[:, P], unique_targets)
        else:
            raise NotImplementedError
        return all_scores


class DefaultBatchNegativeSample(BatchNegativeSample):
    """Default implementation that stores all negative samples as a tensor."""

    def __init__(
        self,
        config: Config,
        configuration_key: str,
        positive_triples: torch.Tensor,
        slot: int,
        num_samples: int,
        samples: torch.Tensor,
    ):
        super().__init__(config, configuration_key, positive_triples, slot, num_samples)
        self._samples = samples

    def samples(self, indexes=None) -> torch.Tensor:
        return self._samples if indexes is None else self._samples[indexes]

    def to(self, device, non_blocking: bool = False) -> "DefaultBatchNegativeSample":
        payload = self._consume_lookahead(device)
        super().to(device, non_blocking=non_blocking)
        if payload and payload.get("samples") is not None:
            self._samples = payload["samples"]
            payload["samples"] = None
        else:
            self._samples = self._samples.to(device, non_blocking=non_blocking)
        return self

    def pin_memory(self) -> "DefaultBatchNegativeSample":
        super().pin_memory()
        if self._samples.device.type == "cpu" and not self._samples.is_pinned():
            self._samples = self._samples.pin_memory()
        return self

    def map_samples(self, mapper):
        device = self._samples.device
        mapper_device = mapper.device if isinstance(mapper, torch.Tensor) else device
        if mapper_device is None:
            mapper_device = torch.device("cpu")
        indexes = self._samples
        if indexes.device != mapper_device:
            indexes_cpu = indexes.to(mapper_device)
        else:
            indexes_cpu = indexes
        mapped = mapper[indexes_cpu]
        if mapped.device != device:
            mapped = mapped.to(device)
        self._samples = mapped


class NaiveSharedNegativeSample(BatchNegativeSample):
    """Implementation for naive shared sampling.

    Here all triples use exactly the same negatives samples.

    """

    def __init__(
        self,
        config: Config,
        configuration_key: str,
        positive_triples: torch.Tensor,
        slot: int,
        num_samples: int,
        unique_samples: torch.Tensor,
        repeat_indexes: torch.Tensor,
    ):
        super().__init__(config, configuration_key, positive_triples, slot, num_samples)
        self._unique_samples = unique_samples
        self._repeat_indexes = repeat_indexes
        self._unique_samples_cpu = None
        self._repeat_indexes_cpu = None
        # Cached column index used when sampling with replacement requires repeating
        # targets. Avoids re-materializing torch.cat(...) every batch.
        self._score_col_index = None
        self._score_col_index_cpu = None

    def _tensor_for_device(self, tensor, cache_attr, device):
        if tensor.device == device:
            return tensor
        if device.type == "cpu":
            cached = getattr(self, cache_attr)
            if (
                cached is None
                or cached.device != device
                or cached.size() != tensor.size()
            ):
                cached = tensor.to(device)
                setattr(self, cache_attr, cached)
            return cached
        return tensor.to(device)

    def unique_samples(self, indexes=None, return_inverse=False, remove_dropped=True) -> torch.Tensor:
        if return_inverse:
            # slow but probably rarely used anyway
            samples = self.samples(indexes)
            return torch.unique(samples.contiguous().view(-1), return_inverse=True)
        else:
            device = self.positive_triples.device
            return self._tensor_for_device(
                self._unique_samples, "_unique_samples_cpu", device
            )

    def samples(self, indexes=None) -> torch.Tensor:
        # create one row, then expand to chunk size
        if type(indexes) == slice:
            chunk_size = len(range(*indexes.indices(len(self.positive_triples))))
        else:
            chunk_size = len(indexes) if indexes else len(self.positive_triples)
        device = self.positive_triples.device
        unique_samples = self._tensor_for_device(
            self._unique_samples, "_unique_samples_cpu", device
        )
        num_unique = len(unique_samples)
        if num_unique == self.num_samples:
            negative_samples1 = unique_samples
        else:
            negative_samples1 = torch.empty(
                self.num_samples, dtype=torch.long, device=device
            )
            negative_samples1[:num_unique] = unique_samples
            repeat_indexes = self._tensor_for_device(
                self._repeat_indexes, "_repeat_indexes_cpu", device
            )
            if repeat_indexes.numel() > 0:
                negative_samples1[num_unique:] = unique_samples[repeat_indexes]

        return negative_samples1.unsqueeze(0).expand((chunk_size, -1))

    def map_samples(self, mapper):
        cpu_samples = self._tensor_for_device(
            self._unique_samples, "_unique_samples_cpu", mapper.device
        )
        mapped = mapper[cpu_samples]
        self._unique_samples = mapped.to(self._unique_samples.device)
        self._unique_samples_cpu = None

    def score(self, model, indexes=None) -> torch.Tensor:
        if self._implementation != "batch":
            return super().score(model, indexes)

        # for batch, we have a faster implementation that avoids creating the full
        # sample tensor
        self.prepare_time = 0.0
        self.forward_time = 0.0
        slot = self.slot
        unique_targets = self._unique_samples
        num_unique = len(unique_targets)
        triples = (
            self.positive_triples
            if indexes is None
            else self.positive_triples[indexes, :]
        )
        chunk_size = len(triples)

        # compute scores for all unique targets for slot
        self.forward_time -= time.time()
        scores = self._score_unique_targets(model, slot, triples, unique_targets)

        # repeat scores as needed for WR sampling
        if num_unique != self.num_samples:
            device = scores.device
            col_index = self._tensor_for_device(
                self._score_col_index,
                "_score_col_index_cpu",
                device,
            )
            if (
                col_index is None
                or col_index.device != device
                or col_index.numel() != self.num_samples
            ):
                repeat_indexes = self._tensor_for_device(
                    self._repeat_indexes, "_repeat_indexes_cpu", device
                )
                col_index = torch.cat(
                    (
                        torch.arange(num_unique, device=device),
                        repeat_indexes,
                    )
                )
                self._score_col_index = col_index if device.type != "cpu" else None
                self._score_col_index_cpu = col_index if device.type == "cpu" else None
            scores = scores[:, col_index]
        self.forward_time += time.time()

        return scores

    def build_lookahead_payload(self, device, non_blocking: bool = False):
        try:
            unique = self._unique_samples.to(device, non_blocking=non_blocking)
            repeat = self._repeat_indexes.to(device, non_blocking=non_blocking)
        except Exception:
            return None
        return {
            "device": self._device_key(device),
            "unique_samples": unique,
            "repeat_indexes": repeat,
        }

    def to(self, device, non_blocking: bool = False) -> "NaiveSharedNegativeSample":
        payload = self._consume_lookahead(device)
        super().to(device, non_blocking=non_blocking)
        if payload and "unique_samples" in payload:
            self._unique_samples = payload["unique_samples"]
            self._repeat_indexes = payload.get("repeat_indexes", self._repeat_indexes)
        else:
            self._unique_samples = self._unique_samples.to(
                device, non_blocking=non_blocking
            )
            self._repeat_indexes = self._repeat_indexes.to(
                device, non_blocking=non_blocking
            )
        self._unique_samples_cpu = None
        self._repeat_indexes_cpu = None
        self._score_col_index = None
        self._score_col_index_cpu = None
        return self

    def pin_memory(self) -> "NaiveSharedNegativeSample":
        super().pin_memory()
        if (
            self._unique_samples.device.type == "cpu"
            and not self._unique_samples.is_pinned()
        ):
            self._unique_samples = self._unique_samples.pin_memory()
            self._unique_samples_cpu = None
        if (
            self._repeat_indexes.device.type == "cpu"
            and not self._repeat_indexes.is_pinned()
        ):
            self._repeat_indexes = self._repeat_indexes.pin_memory()
            self._repeat_indexes_cpu = None
        return self


class DefaultSharedNegativeSample(BatchNegativeSample):
    def __init__(
        self,
        config: Config,
        configuration_key: str,
        positive_triples: torch.Tensor,
        slot: int,
        num_samples: int,
        unique_samples: torch.Tensor,
        drop_index: torch.Tensor,
        repeat_indexes: torch.Tensor,
    ):
        super().__init__(config, configuration_key, positive_triples, slot, num_samples)
        self._unique_samples = unique_samples
        self._drop_index = drop_index
        self._repeat_indexes = repeat_indexes
        self._unique_samples_cpu = None
        self._drop_index_cpu = None
        self._repeat_indexes_cpu = None

    def _tensor_for_device(self, tensor, cache_attr, device):
        if tensor.device == device:
            return tensor
        if device.type == "cpu":
            cached = getattr(self, cache_attr)
            if (
                cached is None
                or cached.device != device
                or cached.size() != tensor.size()
            ):
                cached = tensor.to(device)
                setattr(self, cache_attr, cached)
            return cached
        return tensor.to(device)

    def unique_samples(self, indexes=None, return_inverse=False, remove_dropped=True) -> torch.Tensor:
        if return_inverse:
            # slow but probably rarely used anyway
            return super(DefaultSharedNegativeSample, self).unique_samples(
                indexes=indexes, return_inverse=return_inverse
            )
        if remove_dropped:
            drop_index = self._tensor_for_device(
                self._drop_index, "_drop_index_cpu", self.positive_triples.device
            )
            drop_index = drop_index if indexes is None else drop_index[indexes]
            if torch.all(drop_index == drop_index[0]).item():
                # same sample dropped for every triple in the batch
                not_drop_mask = torch.ones(len(self._unique_samples), dtype=torch.bool)
                not_drop_mask[drop_index[0]] = False
                unique_samples = self._tensor_for_device(
                    self._unique_samples,
                    "_unique_samples_cpu",
                    self.positive_triples.device,
                )
                return unique_samples[not_drop_mask]
        return self._tensor_for_device(
            self._unique_samples, "_unique_samples_cpu", self.positive_triples.device
        )

    def map_samples(self, mapper):
        cpu_samples = self._tensor_for_device(
            self._unique_samples, "_unique_samples_cpu", mapper.device
        )
        mapped = mapper[cpu_samples]
        self._unique_samples = mapped.to(self._unique_samples.device)
        self._unique_samples_cpu = None

    def samples(self, indexes=None) -> torch.Tensor:
        num_samples = self.num_samples
        triples = (
            self.positive_triples
            if indexes is None
            else self.positive_triples[indexes, :]
        )
        drop_index = self._tensor_for_device(
            self._drop_index, "_drop_index_cpu", triples.device
        )
        drop_index = drop_index if indexes is None else drop_index[indexes]
        chunk_size = len(triples)

        # create output tensor
        device = self.positive_triples.device
        unique_samples = self._tensor_for_device(
            self._unique_samples, "_unique_samples_cpu", device
        )
        num_unique = len(unique_samples) - 1
        negative_samples = torch.empty(
            chunk_size, num_unique, dtype=torch.long, device=device
        )

        # Add the first num_distinct samples for each positive. Dropping is
        # performed by copying the last shared sample over the dropped sample
        negative_samples[:, :] = unique_samples[:-1]
        drop_rows = torch.nonzero(drop_index != num_unique, as_tuple=False).squeeze()
        negative_samples[drop_rows, drop_index[drop_rows]] = unique_samples[-1]

        # repeat indexes as needed for WR sampling
        if num_unique != num_samples:
            repeat_indexes = self._tensor_for_device(
                self._repeat_indexes, "_repeat_indexes_cpu", device
            )
            negative_samples = negative_samples[
                :,
                torch.cat((torch.arange(num_unique, device=device), repeat_indexes)),
            ]

        return negative_samples

    def score(self, model, indexes=None) -> torch.Tensor:
        if self._implementation != "batch":
            return super().score(model, indexes)

        # for batch, we have a faster implementation that avoids creating the full
        # sample tensor
        self.prepare_time = 0.0
        self.forward_time = 0.0
        slot = self.slot
        unique_targets = self._tensor_for_device(
            self._unique_samples, "_unique_samples_cpu", self.positive_triples.device
        )
        num_unique = len(unique_targets) - 1
        triples = (
            self.positive_triples[indexes, :] if indexes else self.positive_triples
        )
        drop_index = self._tensor_for_device(
            self._drop_index, "_drop_index_cpu", triples.device
        )
        drop_index = drop_index[indexes] if indexes else drop_index
        drop_rows = torch.nonzero(drop_index != num_unique, as_tuple=False).squeeze()
        chunk_size = len(triples)

        # compute scores for all unique targets for slot
        self.forward_time -= time.time()
        all_scores = self._score_unique_targets(model, slot, triples, unique_targets)

        # create the complete scoring matrix
        device = self.positive_triples.device
        scores = torch.empty(chunk_size, num_unique, device=device)

        # fill in the unique negative scores. first column is left empty
        # to hold positive scores
        scores[:, :] = all_scores[:, :-1]
        scores[drop_rows, drop_index[drop_rows]] = all_scores[drop_rows, -1]

        # repeat scores as needed for WR sampling
        if num_unique != self.num_samples:
            repeat_indexes = self._tensor_for_device(
                self._repeat_indexes, "_repeat_indexes_cpu", device
            )
            scores = scores[
                :,
                torch.cat((torch.arange(num_unique, device=device), repeat_indexes)),
            ]
        self.forward_time += time.time()

        return scores

    def to(self, device, non_blocking: bool = False):
        super().to(device, non_blocking=non_blocking)
        self._unique_samples = self._unique_samples.to(
            device, non_blocking=non_blocking
        )
        self._drop_index = self._drop_index.to(device, non_blocking=non_blocking)
        self._repeat_indexes = self._repeat_indexes.to(
            device, non_blocking=non_blocking
        )
        self._unique_samples_cpu = None
        self._drop_index_cpu = None
        self._repeat_indexes_cpu = None
        return self

    def pin_memory(self) -> "DefaultSharedNegativeSample":
        super().pin_memory()
        if (
            self._unique_samples.device.type == "cpu"
            and not self._unique_samples.is_pinned()
        ):
            self._unique_samples = self._unique_samples.pin_memory()
            self._unique_samples_cpu = None
        if self._drop_index.device.type == "cpu" and not self._drop_index.is_pinned():
            self._drop_index = self._drop_index.pin_memory()
            self._drop_index_cpu = None
        if (
            self._repeat_indexes.device.type == "cpu"
            and not self._repeat_indexes.is_pinned()
        ):
            self._repeat_indexes = self._repeat_indexes.pin_memory()
            self._repeat_indexes_cpu = None
        return self


class CombinedSharedBatchNegativeSample(BatchNegativeSample):
    def __init__(
        self,
        config: Config,
        configuration_key: str,
        positive_triples: torch.Tensor,
        slot: int,
        num_samples: int,
        batch_negative_sample_1: BatchNegativeSample,
        batch_negative_sample_2: BatchNegativeSample,
    ):
        super().__init__(config, configuration_key, positive_triples, slot, num_samples)
        self.batch_negative_sample_1 = batch_negative_sample_1
        self.batch_negative_sample_2 = batch_negative_sample_2

    def unique_samples(self, indexes=None, return_inverse=False, remove_dropped=True):
        if return_inverse:
            # slow but probably rarely used anyway
            samples = self.samples(indexes)
            return torch.unique(samples.contiguous().view(-1), return_inverse=True)
        else:
            unique_samples_1 = self.batch_negative_sample_1.unique_samples(
                indexes, return_inverse, remove_dropped
            )
            unique_samples_2 = self.batch_negative_sample_2.unique_samples(
                indexes, return_inverse, remove_dropped
            )
            if unique_samples_1.numel() == 0:
                return unique_samples_2
            elif unique_samples_2.numel() == 0:
                return unique_samples_1
            return torch.unique(torch.cat([unique_samples_1, unique_samples_2]))

    def map_samples(self, mapper):
        self.batch_negative_sample_1.map_samples(mapper)
        self.batch_negative_sample_2.map_samples(mapper)

    def samples(self, indexes=None) -> torch.Tensor:
        samples_1 = self.batch_negative_sample_1.samples(indexes)
        samples_2 = self.batch_negative_sample_2.samples(indexes)
        if samples_1.numel() == 0:
            return samples_2
        elif samples_2.numel() == 0:
            return samples_1
        return torch.cat((samples_1, samples_2), dim=1)

    def score(self, model, indexes=None) -> torch.Tensor:
        if type(self.batch_negative_sample_1) in [NaiveSharedNegativeSample, BatchNegativeSample] and type(self.batch_negative_sample_2) in [NaiveSharedNegativeSample, BatchNegativeSample]:
            # lets just concat the scoring here
            # not as flexible but faster
            combined_batch_negative = NaiveSharedNegativeSample(
                self.config,
                self.configuration_key,
                self.positive_triples,
                self.slot,
                self.num_samples,
                torch.cat((
                    self.batch_negative_sample_1.unique_samples(),
                    self.batch_negative_sample_2.unique_samples()
                )),
                repeat_indexes=torch.empty(0, dtype=torch.long, device=self.positive_triples.device),
            )
            scores = combined_batch_negative.score(model, indexes)
            return scores
        scores_1 = self.batch_negative_sample_1.score(model, indexes)
        scores_2 = self.batch_negative_sample_2.score(model, indexes)
        # don't concat empty tensors due to pytorch bug
        if scores_1.numel() == 0:
            return scores_2
        elif scores_2.numel() == 0:
            return scores_1
        return torch.cat((scores_1, scores_2), dim=1)

    def to(
        self, device, non_blocking: bool = False
    ) -> "CombinedSharedBatchNegativeSample":
        self.batch_negative_sample_1 = self.batch_negative_sample_1.to(
            device, non_blocking=non_blocking
        )
        self.batch_negative_sample_2 = self.batch_negative_sample_2.to(
            device, non_blocking=non_blocking
        )
        self.positive_triples = self.positive_triples.to(
            device, non_blocking=non_blocking
        )
        return self

    def pin_memory(self) -> "CombinedSharedBatchNegativeSample":
        super().pin_memory()
        self.batch_negative_sample_1 = self.batch_negative_sample_1.pin_memory()
        self.batch_negative_sample_2 = self.batch_negative_sample_2.pin_memory()
        return self


class KgeUniformSampler(KgeSampler):
    def __init__(self, config: Config, configuration_key: str, dataset: Dataset):
        super().__init__(config, configuration_key, dataset)

    def supports_device_sampling(self, positive_triples: torch.Tensor) -> bool:
        if positive_triples.device.type != "cuda":
            return False
        if not self.config.get("job.distributed.sample_on_gpu"):
            return False
        if any(self.filter_positives):
            return False
        if self.shared and self.shared_type != "naive":
            return False
        return True

    def _sample(self, positive_triples: torch.Tensor, slot: int, num_samples: int):
        device = (
            positive_triples.device
            if positive_triples.device.type == "cuda"
            else torch.device("cpu")
        )
        return torch.randint(
            self.vocabulary_size[slot],
            (positive_triples.size(0), num_samples),
            device=device,
        )

    def _sample_shared(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ):
        batch_size = len(positive_triples)
        vocab_size = int(self.vocabulary_size[slot])
        target_device = (
            positive_triples.device
            if positive_triples.device.type == "cuda"
            else torch.device("cpu")
        )

        # determine number of distinct negative samples for each positive
        if self.with_replacement and num_samples > 0:
            population = vocab_size - (0 if self.shared_type == "naive" else 1)
            population = max(population, 1)
            draws = torch.randint(
                population,
                (num_samples,),
                device=target_device if target_device.type != "cuda" else "cpu",
            )
            num_unique = int(torch.unique(draws, sorted=False).numel())
        else:  # WOR -> all samples distinct
            num_unique = num_samples

        samples_needed = num_unique if self.shared_type == "naive" else num_unique + 1
        if samples_needed > vocab_size:
            raise ValueError(
                "Requested more unique negative samples than vocabulary size."
            )
        if samples_needed > 0:
            # Avoid large randperm allocations on huge vocabularies.
            if vocab_size > 1_000_000:
                unique_samples = torch.tensor(
                    random.sample(range(vocab_size), samples_needed),
                    dtype=torch.long,
                )
            else:
                unique_samples = torch.randperm(vocab_size, device="cpu")[:samples_needed]
            if target_device.type == "cuda":
                unique_samples = unique_samples.to(target_device)
        else:
            unique_samples = torch.empty(0, dtype=torch.long)

        if num_unique != num_samples and num_unique > 0:
            repeat_indexes = torch.randint(
                0,
                num_unique,
                (num_samples - num_unique,),
                dtype=torch.long,
                device=target_device,
            )
        else:
            repeat_indexes = torch.empty(0, dtype=torch.long, device=target_device)

        if self.shared_type == "naive":
            return NaiveSharedNegativeSample(
                self.config,
                self.configuration_key,
                positive_triples,
                slot,
                num_samples,
                unique_samples.long(),
                repeat_indexes,
            )

        positives = positive_triples[:, slot].to("cpu").long()
        drop_index = KgeUniformSampler._create_shared_drop_index(
            positives, unique_samples.long(), num_unique
        )

        return DefaultSharedNegativeSample(
            self.config,
            self.configuration_key,
            positive_triples,
            slot,
            num_samples,
            unique_samples.long(),
            drop_index,
            repeat_indexes,
        )

    @staticmethod
    def _create_shared_drop_index(
        positives: torch.Tensor, unique_samples: torch.Tensor, num_unique: int
    ) -> torch.Tensor:
        batch_size = positives.numel()
        if num_unique == 0:
            return torch.zeros(batch_size, dtype=torch.long)

        drop_index = torch.randint(0, num_unique, (batch_size,), dtype=torch.long)
        active = unique_samples[:num_unique]
        sorted_vals, order = torch.sort(active)
        positions = torch.searchsorted(sorted_vals, positives, right=False)
        valid = positions < sorted_vals.numel()
        if valid.any():
            pos_valid = positions[valid]
            matches = sorted_vals[pos_valid] == positives[valid]
            if matches.any():
                replacement = order[pos_valid[matches]]
                tmp = drop_index[valid]
                tmp[matches] = replacement
                drop_index[valid] = tmp
        return drop_index

    def _filter_and_resample_fast(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ):
        return super()._filter_and_resample(negative_samples, slot, positive_triples)

class KgeFrequencySampler(KgeSampler):
    """
    Sample negatives based on their relative occurrence in the slot in the train set.
    Sample frequency based in hierarchical fashion
    Can be smoothed with a symmetric prior.
    """

    def __init__(self, config, configuration_key, dataset):
        super().__init__(config, configuration_key, dataset)
        self._multinomials = []
        alpha = self.get_option("frequency.smoothing")
        for slot in SLOTS:
            counts = torch.bincount(
                dataset.split(config.get("train.split"))[:, slot].long(),
                minlength=self.vocabulary_size[slot].item(),
            ).float()
            smoothed_counts = counts + float(alpha)
            probs = smoothed_counts / smoothed_counts.sum()
            if self.with_replacement:
                self._multinomials.append(
                    torch._multinomial_alias_setup(
                        probs
                    )
                )
            else:
                self._multinomials.append(probs)

    def _sample(self, positive_triples: torch.Tensor, slot: int, num_samples: int):
        if num_samples is None:
            num_samples = self.num_samples[slot].item()

        if num_samples == 0:
            result = torch.empty([positive_triples.size(0), num_samples])
        else:
            if self.with_replacement:
                result = torch._multinomial_alias_draw(
                            self._multinomials[slot][1],
                            self._multinomials[slot][0],
                            positive_triples.size(0) * num_samples,
                        ).view(positive_triples.size(0), num_samples)
            else:
                result = torch.multinomial(
                            self._multinomials[slot],
                            positive_triples.size(0) * num_samples,
                            replacement=False,
                        ).view(positive_triples.size(0), num_samples)

        return result

    def _sample_shared(
            self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ):
        batch_size = len(positive_triples)

        # note: those are not unique. This is just a quick implementation for evaluation
        unique_samples = self._sample(
            positive_triples[0].view(1, -1),
            slot,
            num_samples if self.shared_type == "naive" else num_samples + 1,
        ).view(-1)
        repeat_indexes = torch.empty(0)

        # for naive shared sampling, we are done
        if self.shared_type == "naive":
            return NaiveSharedNegativeSample(
                self.config,
                self.configuration_key,
                positive_triples,
                slot,
                num_samples,
                unique_samples.long(),
                repeat_indexes,
            )

        # For default, we now filter the positives. For each row i (positive triple),
        # select a sample to drop. For rows that contain its positive as a negative
        # example, drop that positive. For all other rows, drop a random position. Here
        # we start with random drop position for each row and then update the ones that
        # contain its positive in the negative samples
        positives = positive_triples[:, slot].long()
        num_unique = unique_samples.numel()
        drop_index = self._create_shared_drop_index(
            positives, unique_samples.long(), num_unique
        )

        # now we are done for default
        return DefaultSharedNegativeSample(
            self.config,
            self.configuration_key,
            positive_triples,
            slot,
            num_samples,
            unique_samples.long(),
            drop_index,
            repeat_indexes,
        )


class KgeHierarchicalFrequencySampler(KgeSampler):
    """
    Sample negatives based on their relative occurrence in the slot in the train set.
    Can be smoothed with a symmetric prior.
    """

    def __init__(self, config, configuration_key, dataset):
        super().__init__(config, configuration_key, dataset)
        self._multinomials = []
        self._h2_multinomials = []
        self._h2_sorted_indices = []
        self._h2_group_offsets = []
        self._h2_group_counts = []
        alpha = self.get_option("frequency.smoothing")
        for slot in SLOTS:
            counts = torch.bincount(
                dataset.split(config.get("train.split"))[:, slot].long(),
                minlength=self.vocabulary_size[slot].item(),
            ).float()
            smoothed_counts = counts + float(alpha)
            sorted_counts, sorted_indices = torch.sort(smoothed_counts)
            h2_unique_counts, h2_counts_counts = torch.unique_consecutive(
                sorted_counts, return_counts=True
            )
            h2_group_offsets = torch.cumsum(h2_counts_counts, dim=0) - h2_counts_counts
            self._h2_sorted_indices.append(sorted_indices)
            self._h2_group_offsets.append(h2_group_offsets)
            self._h2_group_counts.append(h2_counts_counts)
            probs = h2_counts_counts.float() / h2_counts_counts.sum()
            self._h2_multinomials.append(probs)

    def _sample(self, positive_triples: torch.Tensor, slot: int, num_samples: int):
        if num_samples is None:
            num_samples = self.num_samples[slot].item()

        if num_samples == 0:
            result = torch.empty([positive_triples.size(0), num_samples])
        else:
            result_1 = torch.multinomial(
                self._h2_multinomials[slot],
                positive_triples.size(0) * num_samples,
                replacement=True,
            ).view(positive_triples.size(0), num_samples)
            flat_groups = result_1.view(-1)
            group_counts = self._h2_group_counts[slot][flat_groups]
            group_offsets = self._h2_group_offsets[slot][flat_groups]
            rand = torch.rand(group_counts.numel()) * group_counts.float()
            within = rand.to(torch.long)
            flat_result = self._h2_sorted_indices[slot][group_offsets + within]
            result = flat_result.view(positive_triples.size(0), num_samples)

        return result

    def _sample_shared(
            self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ):
        # note: those are not unique. This is just a quick implementation for evaluation
        unique_samples = self._sample(
            positive_triples[0].view(1, -1),
            slot,
            num_samples if self.shared_type == "naive" else num_samples + 1,
        ).view(-1)
        repeat_indexes = torch.empty(0)

        # for naive shared sampling, we are done
        if self.shared_type == "naive":
            return NaiveSharedNegativeSample(
                self.config,
                self.configuration_key,
                positive_triples,
                slot,
                num_samples,
                unique_samples.long(),
                repeat_indexes,
            )
        else:
            raise NotImplementedError(
                "shared hierarchical frequency sampling is not yet supported")


class KgeBatchSampler(KgeSampler):
    def __init__(self, config, configuration_key, dataset):
        super().__init__(config, configuration_key, dataset)
        if self.get_option("shared"):
            if not self.get_option("shared_type") == "naive":
                raise ValueError("only shared_type naive supported with batch sampling")
            if not self.get_option("with_replacement"):
                raise ValueError(
                    "without replacement sampling not supported with batch sampling"
                )

    def supports_device_sampling(self, positive_triples: torch.Tensor) -> bool:
        if positive_triples.device.type != "cuda":
            return False
        if not self.config.get("job.distributed.sample_on_gpu"):
            return False
        if any(self.filter_positives):
            return False
        return True

    def _sample(self, positive_triples: torch.Tensor, slot: int, num_samples: int):
        device = positive_triples.device
        return positive_triples[:, slot][
            torch.randint(
                len(positive_triples),
                [len(positive_triples), num_samples],
                dtype=torch.long,
                device=device,
            )
        ]

    def _sample_shared(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ):
        device = positive_triples.device
        batch_samples = positive_triples[:, slot][
            torch.randint(
                len(positive_triples), (num_samples,), dtype=torch.long, device=device
            )
        ]
        return NaiveSharedNegativeSample(
            self.config,
            self.configuration_key,
            positive_triples,
            slot,
            num_samples,
            batch_samples,
            torch.empty(0),
        )

        # don't use repeat index as it is faster without
        # unique_samples, counts = torch.unique(batch_samples, return_counts=True)
        # repeat_indexes = torch.from_numpy(
        #     self._create_repeat_index_from_counts(
        #         unique_samples.numpy(), counts.numpy()
        #     )
        # ).long()

        # return NaiveSharedNegativeSample(
        #     self.config,
        #     self.configuration_key,
        #     positive_triples,
        #     slot,
        #     num_samples,
        #     unique_samples,
        #     repeat_indexes,
        # )

    @staticmethod
    @numba.njit
    def _create_repeat_index_from_counts(unique_samples: np.array, counts: np.array):
        """
        Creates the repeat index needed for the shared negative sample object.
        Calculates based on the counts of the unique samples
        Args:
            unique_samples: unique negative samples
            counts: count of each unique negative sample

        Returns:
            returns a 1d-tensor with len(sum(counts-1)) containing the ids of entities
            to repeat
        """
        len_repeat_index = np.sum(counts - 1)
        repeat_index = np.zeros((len_repeat_index,))
        repeat_position = 0
        for i in range(len(unique_samples)):
            for j in range(counts[i] - 1):
                repeat_index[repeat_position] = i
                repeat_position += 1
        return repeat_index


class KgeCombinedSampler(KgeSampler):
    def __init__(self, config, configuration_key, dataset):
        super().__init__(config, configuration_key, dataset)
        self.sampler_1: KgeSampler = KgeSampler._create(
            config, configuration_key, dataset
        )
        self.sampler_2: KgeSampler = KgeSampler._create(
            self._create_second_sampler_config(), configuration_key, dataset
        )
        self.sampler_2_percentage = self.get_option(
            "combined_options.negatives_percentage"
        )
        if config.get("negative_sampling.shared_type") == "naive" and config.get("negative_sampling.shared") and type(self.sampler_2) is KgeBatchSampler:
            # enforce more efficient scoring
            # here we avoid that a repeat index is used in the naive shared sampler
            warnings.warn("setting with replacement to true to sampler 1. This allows for more efficient scoring. Only used in the combination of naive shared sampling with batch sampling.")
            self.sampler_1.with_replacement = False

    def supports_device_sampling(self, positive_triples: torch.Tensor) -> bool:
        return (
            self.sampler_1.supports_device_sampling(positive_triples)
            and self.sampler_2.supports_device_sampling(positive_triples)
        )

    def uses_pool(self) -> bool:
        return self.sampler_1.uses_pool() or self.sampler_2.uses_pool()

    def _create_second_sampler_config(self):
        """
        Creates config object for the second sampler based on the options defined
        under the key combined_options
        Returns:
            Config object
        """
        sampler_2_config = Config()
        sampler_2_options = {
            self.configuration_key: self.config.get(self.configuration_key)
        }
        combined_options = self.get_option("combined_options")
        for key, option in combined_options.items():
            if key == "negatives_percentage":
                continue
            sampler_2_options[self.configuration_key][key] = option
        sampler_2_config.set_all(sampler_2_options, create=True)
        return sampler_2_config

    def _sample(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ) -> torch.Tensor:
        num_samples_2 = int(num_samples * self.sampler_2_percentage)
        num_samples_1 = num_samples - num_samples_2
        negatives_1 = self.sampler_1._sample(positive_triples, slot, num_samples_1)
        negatives_2 = self.sampler_2._sample(positive_triples, slot, num_samples_2)
        return torch.cat((negatives_1, negatives_2), dim=1)

    def _sample_shared(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ) -> "BatchNegativeSample":
        num_samples_2 = int(num_samples * self.sampler_2_percentage)
        num_samples_1 = num_samples - num_samples_2
        batch_negative_sample_1 = self.sampler_1._sample_shared(
            positive_triples, slot, num_samples_1
        )
        batch_negative_sample_2 = self.sampler_2._sample_shared(
            positive_triples, slot, num_samples_2
        )
        return CombinedSharedBatchNegativeSample(
            config=self.config,
            configuration_key=self.configuration_key,
            positive_triples=positive_triples,
            slot=slot,
            num_samples=num_samples,
            batch_negative_sample_1=batch_negative_sample_1,
            batch_negative_sample_2=batch_negative_sample_2,
        )

    def _filter_and_resample(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ) -> torch.Tensor:
        return self._handle_filtering(
            negative_samples, slot, positive_triples, implementation="standard"
        )

    def _filter_and_resample_fast(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ) -> torch.Tensor:
        return self._handle_filtering(
            negative_samples, slot, positive_triples, implementation="fast"
        )

    def _handle_filtering(
        self,
        negative_samples: torch.Tensor,
        slot: int,
        positive_triples: torch.Tensor,
        implementation="standard",
    ) -> torch.Tensor:
        if implementation == "fast":
            filter_function_name = "_filter_and_resample_fast"
        else:
            filter_function_name = "_filter_and_resample"
        num_samples = negative_samples.shape[1]
        num_samples_2 = int(num_samples * self.sampler_2_percentage)
        num_samples_1 = num_samples - num_samples_2
        negative_samples_1 = self.sampler_1.__getattribute__(filter_function_name)(
            negative_samples[:, :num_samples_1], slot, positive_triples
        )
        negative_samples_2 = self.sampler_2.__getattribute__(filter_function_name)(
            negative_samples[:, num_samples_1:], slot, positive_triples
        )
        return torch.cat((negative_samples_1, negative_samples_2), dim=1)

    def set_pool(self, pool: torch.Tensor, slot: int):
        if type(self.sampler_1) is KgePooledSampler:
            self.sampler_1.set_pool(pool, slot)
        if type(self.sampler_2) is KgePooledSampler:
            self.sampler_2.set_pool(pool, slot)


class KgeCacheAwareSampler(KgeSampler):
    def __init__(self, config, configuration_key, dataset):
        super().__init__(config, configuration_key, dataset)
        resident_fraction = float(self.get_option("cache_aware.resident_fraction"))
        if resident_fraction < 0.0 or resident_fraction > 1.0:
            raise ValueError(
                "negative_sampling.cache_aware.resident_fraction must be in [0, 1]"
            )
        self.resident_fraction = resident_fraction
        self.resident_sampling_type = self.get_option(
            "cache_aware.resident_sampling_type"
        )
        self.background_sampling_type = self.get_option(
            "cache_aware.background_sampling_type"
        )
        if (
            config.get("job.distributed.entity_sync_level") == "partition"
            and self.background_sampling_type != "pooled"
        ):
            raise ValueError(
                "cache_aware sampling with partition sync requires "
                "cache_aware.background_sampling_type == 'pooled'."
            )
        self.resident_sampler = KgeSampler._create(
            self._create_sampler_config(self.resident_sampling_type),
            configuration_key,
            dataset,
        )
        self.background_sampler = KgeSampler._create(
            self._create_sampler_config(self.background_sampling_type),
            configuration_key,
            dataset,
        )

    def _create_sampler_config(self, sampling_type: str) -> Config:
        sampler_config = Config()
        sampler_options = {
            self.configuration_key: self.config.get(self.configuration_key)
        }
        sampler_options[self.configuration_key]["sampling_type"] = sampling_type
        sampler_options[self.configuration_key]["combined"] = False
        sampler_config.set_all(sampler_options, create=True)
        return sampler_config

    def supports_device_sampling(self, positive_triples: torch.Tensor) -> bool:
        return (
            self.resident_sampler.supports_device_sampling(positive_triples)
            and self.background_sampler.supports_device_sampling(positive_triples)
        )

    def uses_pool(self) -> bool:
        return self.resident_sampler.uses_pool() or self.background_sampler.uses_pool()

    def _sample(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ) -> torch.Tensor:
        num_resident = int(num_samples * self.resident_fraction)
        num_background = num_samples - num_resident
        if num_resident <= 0:
            return self.background_sampler._sample(
                positive_triples, slot, num_background
            )
        if num_background <= 0:
            return self.resident_sampler._sample(
                positive_triples, slot, num_resident
            )
        negatives_resident = self.resident_sampler._sample(
            positive_triples, slot, num_resident
        )
        negatives_background = self.background_sampler._sample(
            positive_triples, slot, num_background
        )
        return torch.cat((negatives_resident, negatives_background), dim=1)

    def _sample_shared(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ) -> "BatchNegativeSample":
        num_resident = int(num_samples * self.resident_fraction)
        num_background = num_samples - num_resident
        if num_resident <= 0:
            return self.background_sampler._sample_shared(
                positive_triples, slot, num_background
            )
        if num_background <= 0:
            return self.resident_sampler._sample_shared(
                positive_triples, slot, num_resident
            )
        batch_negative_resident = self.resident_sampler._sample_shared(
            positive_triples, slot, num_resident
        )
        batch_negative_background = self.background_sampler._sample_shared(
            positive_triples, slot, num_background
        )
        return CombinedSharedBatchNegativeSample(
            config=self.config,
            configuration_key=self.configuration_key,
            positive_triples=positive_triples,
            slot=slot,
            num_samples=num_samples,
            batch_negative_sample_1=batch_negative_resident,
            batch_negative_sample_2=batch_negative_background,
        )

    def _filter_and_resample(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ) -> torch.Tensor:
        return self._handle_filtering(
            negative_samples, slot, positive_triples, implementation="standard"
        )

    def _filter_and_resample_fast(
        self, negative_samples: torch.Tensor, slot: int, positive_triples: torch.Tensor
    ) -> torch.Tensor:
        return self._handle_filtering(
            negative_samples, slot, positive_triples, implementation="fast"
        )

    def _handle_filtering(
        self,
        negative_samples: torch.Tensor,
        slot: int,
        positive_triples: torch.Tensor,
        implementation="standard",
    ) -> torch.Tensor:
        if implementation == "fast":
            filter_function_name = "_filter_and_resample_fast"
        else:
            filter_function_name = "_filter_and_resample"
        num_samples = negative_samples.shape[1]
        num_resident = int(num_samples * self.resident_fraction)
        num_background = num_samples - num_resident
        if num_resident <= 0:
            return self.background_sampler.__getattribute__(filter_function_name)(
                negative_samples, slot, positive_triples
            )
        if num_background <= 0:
            return self.resident_sampler.__getattribute__(filter_function_name)(
                negative_samples, slot, positive_triples
            )
        negative_resident = self.resident_sampler.__getattribute__(filter_function_name)(
            negative_samples[:, :num_resident], slot, positive_triples
        )
        negative_background = self.background_sampler.__getattribute__(
            filter_function_name
        )(negative_samples[:, num_resident:], slot, positive_triples)
        return torch.cat((negative_resident, negative_background), dim=1)

    def set_pool(self, pool: torch.Tensor, slot: int):
        if hasattr(self.resident_sampler, "set_pool"):
            self.resident_sampler.set_pool(pool, slot)
        if hasattr(self.background_sampler, "set_pool"):
            self.background_sampler.set_pool(pool, slot)


class KgePooledSampler(KgeSampler):
    def __init__(self, config, configuration_key, dataset):
        super().__init__(config, configuration_key, dataset)
        # these tensors need to be shared since we are keeping the data loader workers
        # alive. Otherwise pools won't be updated in all workers
        self.sample_pools = dict()
        self.sample_pools[S] = torch.randperm(self.vocabulary_size[S]).share_memory_()
        self.sample_pools[P] = torch.randperm(self.vocabulary_size[P]).share_memory_()
        self.sample_pools[O] = self.sample_pools[S]
        self.sample_pools_device = {S: None, P: None, O: None}
        self.sample_pool_sizes = dict()
        self.sample_pool_sizes[S] = torch.zeros([1, ], dtype=torch.int).share_memory_()
        self.sample_pool_sizes[P] = torch.zeros([1, ], dtype=torch.int).share_memory_()
        self.sample_pool_sizes[O] = self.sample_pool_sizes[S]

    def supports_device_sampling(self, positive_triples: torch.Tensor) -> bool:
        if positive_triples.device.type != "cuda":
            return False
        if not self.config.get("job.distributed.sample_on_gpu"):
            return False
        if any(self.filter_positives):
            return False
        if self.shared and self.shared_type != "naive":
            return False
        for slot in SLOTS:
            if int(self.num_samples[slot].item()) <= 0:
                continue
            if self.sample_pools_device.get(slot) is None:
                return False
        return True

    def uses_pool(self) -> bool:
        return True

    def _sample(self, positive_triples: torch.Tensor, slot: int, num_samples: int):
        pool = self.sample_pools_device.get(slot)
        if pool is not None:
            idx = torch.randint(
                pool.size(0),
                (positive_triples.size(0), num_samples),
                device=pool.device,
            )
            return pool.index_select(0, idx.view(-1)).view_as(idx)
        cpu_pool = self.sample_pools[slot]
        return cpu_pool[
            torch.randint(
                len(cpu_pool), (positive_triples.size(0), num_samples)
            )
        ]

    def _sample_shared(
        self, positive_triples: torch.Tensor, slot: int, num_samples: int
    ):
        # if not self.shared_type == "naive":
        #     raise NotImplementedError("currently only naive shared samping supported for pooled")
        # # determine number of distinct negative samples for each positive

        batch_size = len(positive_triples)
        pool_size = self.sample_pool_sizes[slot].item()

        if self.with_replacement:
            # Simple way to get a sample from the distribution of number of distinct
            # values in the negative sample (for "default" type: WR sampling except the
            # positive, hence the - 1)
            if pool_size > 0 and num_samples > 0:
                draws = torch.randint(pool_size, (num_samples,), dtype=torch.long)
                num_unique = int(torch.unique(draws, sorted=False).numel())
            else:
                num_unique = 0
        else:  # WOR -> all samples distinct
            num_unique = num_samples

        # Take the WOR sample. For default, take one more WOR sample than necessary
        # (used to replace sampled positives). Numpy is horribly slow for large
        # vocabulary sizes, so we use random.sample instead.
        #
        # SLOW:
        # unique_samples = np.random.choice(
        #     self.vocabulary_size[slot], num_unique, replace=False
        # )

        device_pool = self.sample_pools_device.get(slot)
        if (
            device_pool is not None
            and self.shared_type == "naive"
            and device_pool.size(0) > 0
        ):
            return self._sample_shared_device(
                positive_triples, slot, num_samples, device_pool
            )

        # set pool size to ensure it does not fail in P-slot
        pool_size = max(1, pool_size)
        sample_count = num_unique if self.shared_type == "naive" else num_unique + 1
        if sample_count <= pool_size:
            sample_index = torch.tensor(
                random.sample(range(pool_size), sample_count),
                dtype=torch.long,
            )
        else:
            sample_index = torch.randint(
                0, pool_size, (sample_count,), dtype=torch.long
            )
        unique_samples = self.sample_pools[slot][sample_index]

        # For WR, we need to upsample. To do so, we compute the set of additional
        # (repeated) sample indexes.
        if num_unique != num_samples:  # only happens with WR
            repeat_indexes = torch.randint(
                0, num_unique, (num_samples - num_unique,), dtype=torch.long
            )
        else:
            repeat_indexes = torch.empty(0)  # WOR or WR when all samples unique
        # for naive shared sampling, we are done
        if self.shared_type == "naive":
            return NaiveSharedNegativeSample(
                self.config,
                self.configuration_key,
                positive_triples,
                slot,
                num_samples,
                unique_samples,
                # torch.tensor(unique_samples, dtype=torch.long),
                repeat_indexes,
            )

        # For default, we now filter the positives. For each row i (positive triple),
        # select a sample to drop. For rows that contain its positive as a negative
        # example, drop that positive. For all other rows, drop a random position. Here
        # we start with random drop position for each row and then update the ones that
        # contain its positive in the negative samples
        positives = positive_triples[:, slot].to("cpu").long()
        drop_index = KgeUniformSampler._create_shared_drop_index(
            positives, unique_samples.long(), unique_samples.numel()
        )

        # now we are done for default
        return DefaultSharedNegativeSample(
            self.config,
            self.configuration_key,
            positive_triples,
            slot,
            num_samples,
            unique_samples,
            # torch.tensor(unique_samples, dtype=torch.long),
            drop_index,
            repeat_indexes,
        )

    def set_pool(self, pool: torch.Tensor, slot: int):
        pool_len = len(pool)
        self.sample_pool_sizes[slot][0] = pool_len
        if pool.is_cuda:
            device_pool = self.sample_pools_device.get(slot)
            needs_alloc = (
                device_pool is None
                or device_pool.device != pool.device
                or device_pool.size(0) < pool_len
            )
            if needs_alloc:
                device_pool = torch.empty(
                    (pool_len,), dtype=pool.dtype, device=pool.device
                )
            device_pool[:pool_len] = pool
            self.sample_pools_device[slot] = device_pool
            return

        self.sample_pools[slot][:pool_len] = pool
        self.sample_pools_device[slot] = None

    def _sample_shared_device(
        self,
        positive_triples: torch.Tensor,
        slot: int,
        num_samples: int,
        pool: torch.Tensor,
    ):
        pool_size = int(self.sample_pool_sizes[slot].item())
        pool_size = min(pool_size, pool.size(0))
        pool_size = max(pool_size, 1)
        device = pool.device
        if self.with_replacement:
            draws = torch.randint(pool_size, (num_samples,), device=device)
            unique_indexes = torch.unique(draws, sorted=False)
        else:
            if num_samples <= pool_size:
                unique_indexes = torch.empty(0, dtype=torch.long, device=device)
                while unique_indexes.numel() < num_samples:
                    remaining = num_samples - unique_indexes.numel()
                    candidates = torch.randint(
                        pool_size, (max(remaining * 2, remaining),), device=device
                    )
                    unique_indexes = torch.unique(
                        torch.cat((unique_indexes, candidates)), sorted=False
                    )
                    unique_indexes = unique_indexes[:num_samples]
            else:
                draws = torch.randint(pool_size, (num_samples,), device=device)
                unique_indexes = torch.unique(draws, sorted=False)
        num_unique = unique_indexes.numel()
        if num_unique == 0:
            unique_indexes = torch.zeros(1, dtype=torch.long, device=device)
            num_unique = 1
        unique_samples = pool.index_select(0, unique_indexes)
        if num_unique != num_samples:
            repeat_indexes = torch.randint(
                num_unique,
                (num_samples - num_unique,),
                dtype=torch.long,
                device=device,
            )
        else:
            repeat_indexes = torch.empty(0, dtype=torch.long, device=device)
        return NaiveSharedNegativeSample(
            self.config,
            self.configuration_key,
            positive_triples,
            slot,
            num_samples,
            unique_samples,
            repeat_indexes,
        )
