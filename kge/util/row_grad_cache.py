"""Row-sparse gradient cache for reduce-by-key optimizers."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import threading

import torch

_ROW_GRAD_CACHE: Dict[
    int, Tuple[torch.Tensor, torch.Tensor, Optional[torch.cuda.Event]]
] = {}
_ROW_GRAD_LOCK = threading.Lock()


def store_row_grad(
    param: torch.Tensor, row_ids: torch.Tensor, row_values: torch.Tensor
) -> None:
    event = None
    if row_ids.is_cuda or row_values.is_cuda:
        if row_ids.device != row_values.device:
            raise RuntimeError("Row grad IDs/values must be on the same device.")
        event = torch.cuda.Event(enable_timing=False, blocking=False)
        event.record(torch.cuda.current_stream(device=row_ids.device))
    with _ROW_GRAD_LOCK:
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
