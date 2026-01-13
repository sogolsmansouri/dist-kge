"""Row-sparse gradient cache for reduce-by-key optimizers."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import threading

import torch

_ROW_GRAD_CACHE: Dict[
    int, Tuple[torch.Tensor, torch.Tensor, Optional[torch.cuda.Event]]
] = {}
_ROW_GRAD_LOCK = threading.Lock()


def _reduce_by_key_dense(
    indices: torch.Tensor, values: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reduce-by-key on 1D indices and 2D values -> (unique_indices, summed_values)."""
    if indices.numel() == 0:
        return indices, values
    if indices.dim() != 1:
        indices = indices.view(-1)
    if values.dim() != 2:
        raise ValueError("row_values must be 2D [nnz, dim].")
    perm = torch.argsort(indices)
    idx_sorted = indices.index_select(0, perm)
    val_sorted = values.index_select(0, perm)
    boundary = torch.ones_like(idx_sorted, dtype=torch.bool)
    if idx_sorted.numel() > 1:
        boundary[1:] = idx_sorted[1:] != idx_sorted[:-1]
    segment_id = torch.cumsum(boundary, dim=0) - 1
    num_unique = int(segment_id[-1].item()) + 1
    out = torch.zeros(
        (num_unique, values.size(1)),
        device=values.device,
        dtype=values.dtype,
    )
    out.index_add_(0, segment_id, val_sorted)
    unique = idx_sorted[boundary]
    return unique, out


def store_row_grad(
    param: torch.Tensor, row_ids: torch.Tensor, row_values: torch.Tensor
) -> None:
    if row_ids.device != row_values.device:
        raise RuntimeError("Row grad IDs/values must be on the same device.")
    if row_ids.dtype != torch.long:
        row_ids = row_ids.long()
    # Make sure shapes are consistent
    row_ids = row_ids.view(-1)
    if row_values.dim() != 2:
        raise ValueError("row_values must be 2D [nnz, dim].")

    # IMPORTANT: accumulation-safe behavior:
    # If store_row_grad() is called multiple times before optimizer.step(),
    # we must not overwrite and drop earlier gradients.
    with _ROW_GRAD_LOCK:
        prev = _ROW_GRAD_CACHE.get(id(param))
        if prev is not None:
            prev_ids, prev_vals, prev_event = prev
            # Ensure producer stream is complete before we read/concat previous tensors
            if prev_event is not None:
                torch.cuda.current_stream(device=prev_ids.device).wait_event(prev_event)
            # concat + reduce
            row_ids = torch.cat([prev_ids, row_ids], dim=0)
            row_values = torch.cat([prev_vals, row_values], dim=0)
            row_ids, row_values = _reduce_by_key_dense(row_ids, row_values)

        event = None
        if row_ids.is_cuda:
            event = torch.cuda.Event(enable_timing=False, blocking=False)
            event.record(torch.cuda.current_stream(device=row_ids.device))

        _ROW_GRAD_CACHE[id(param)] = (row_ids, row_values, event)


def pop_row_grad(
    param: torch.Tensor,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    with _ROW_GRAD_LOCK:
        entry = _ROW_GRAD_CACHE.pop(id(param), None)
    if entry is None:
        return None
    row_ids, row_values, event = entry
    if event is not None:
        torch.cuda.current_stream(device=row_ids.device).wait_event(event)
    return row_ids, row_values
