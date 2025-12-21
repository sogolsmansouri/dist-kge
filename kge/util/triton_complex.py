"""Optional helpers for fused ComplEx computations.

By default the safe PyTorch implementation is used. Setting the environment
variable ``KGE_COMPLEX_FUSED`` to ``1`` (or another truthy value such as
``true``/``yes``) enables the fused path. When Triton is available we can add
specialised kernels here in the future; at the moment the fused path relies on
PyTorch's complex matmul which is numerically stable and still benefits from a
single GEMM per slot.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

_USE_FUSED = os.getenv("KGE_COMPLEX_FUSED", "").lower() in ("1", "true", "yes")


def fused_complex_requested() -> bool:
    """Return whether the fused ComplEx scorer is requested via env flag."""

    return _USE_FUSED


def has_triton_complex() -> bool:
    """Placeholder for future Triton kernels (currently unused)."""

    return False


def complex_matmul(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    conj_right: bool,
) -> torch.Tensor:
    """Multiply complex matrices where right-hand side may be conjugated."""

    if left.dim() != 2 or right.dim() != 2:
        raise ValueError("complex_matmul expects 2-D tensors")
    if left.size(1) != right.size(1):
        raise ValueError("last dimension mismatch in complex_matmul")
    rhs = right.conj() if conj_right else right
    return left @ rhs.transpose(0, 1)
