"""Optional helpers for fused ComplEx computations.

By default the safe PyTorch implementation is used. Setting the environment
variable ``KGE_COMPLEX_FUSED`` to ``1`` (or another truthy value such as
``true``/``yes``) enables the fused path. When Triton is available we use a
specialised kernel for the SPO path to reduce Python overhead and improve
throughput while keeping accumulation in float32.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

try:  # pragma: no-cover - optional dependency
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no-cover - import guard
    triton = None
    tl = None
    _HAS_TRITON = False

_USE_FUSED = os.getenv("KGE_COMPLEX_FUSED", "").lower() in ("1", "true", "yes")


def fused_complex_requested() -> bool:
    """Return whether the fused ComplEx scorer is requested via env flag."""

    return _USE_FUSED


def has_triton_complex() -> bool:
    """Return whether Triton-based ComplEx kernels are available."""

    return _HAS_TRITON


def _complex_score_spo_torch(
    s_emb: torch.Tensor, p_emb: torch.Tensor, o_emb: torch.Tensor
) -> torch.Tensor:
    s_re, s_im = s_emb.chunk(2, dim=1)
    p_re, p_im = p_emb.chunk(2, dim=1)
    o_re, o_im = o_emb.chunk(2, dim=1)

    t_re = s_re * p_re - s_im * p_im
    t_im = s_re * p_im + s_im * p_re
    out = t_re * o_re + t_im * o_im
    return out.sum(dim=1)


BLOCK_K = 256


if _HAS_TRITON:

    @triton.jit
    def _complex_score_spo_kernel(
        s_ptr,
        p_ptr,
        o_ptr,
        out_ptr,
        stride_s,
        stride_p,
        stride_o,
        dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        row_s = s_ptr + pid_m * stride_s
        row_p = p_ptr + pid_m * stride_p
        row_o = o_ptr + pid_m * stride_o

        offs = pid_k * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < dim

        s_re = tl.load(row_s + offs, mask=mask, other=0.0).to(tl.float32)
        s_im = tl.load(row_s + offs + dim, mask=mask, other=0.0).to(tl.float32)
        p_re = tl.load(row_p + offs, mask=mask, other=0.0).to(tl.float32)
        p_im = tl.load(row_p + offs + dim, mask=mask, other=0.0).to(tl.float32)
        o_re = tl.load(row_o + offs, mask=mask, other=0.0).to(tl.float32)
        o_im = tl.load(row_o + offs + dim, mask=mask, other=0.0).to(tl.float32)

        t_re = s_re * p_re - s_im * p_im
        t_im = s_re * p_im + s_im * p_re
        u_re = t_re * o_re + t_im * o_im
        partial = tl.sum(u_re, axis=0)

        tl.atomic_add(out_ptr + pid_m, partial)


def complex_score_spo_triton(
    s_emb: torch.Tensor, p_emb: torch.Tensor, o_emb: torch.Tensor
) -> torch.Tensor:
    """Compute ComplEx SPO scores via Triton when available."""

    if not _HAS_TRITON or not s_emb.is_cuda or not p_emb.is_cuda or not o_emb.is_cuda:
        return _complex_score_spo_torch(s_emb, p_emb, o_emb)

    if s_emb.dim() != 2 or p_emb.dim() != 2 or o_emb.dim() != 2:
        return _complex_score_spo_torch(s_emb, p_emb, o_emb)
    if s_emb.size(1) != p_emb.size(1) or s_emb.size(1) != o_emb.size(1):
        return _complex_score_spo_torch(s_emb, p_emb, o_emb)
    if s_emb.size(1) % 2 != 0:
        return _complex_score_spo_torch(s_emb, p_emb, o_emb)

    s_emb = s_emb.contiguous()
    p_emb = p_emb.contiguous()
    o_emb = o_emb.contiguous()

    dim = s_emb.size(1) // 2
    out = torch.zeros((s_emb.size(0),), device=s_emb.device, dtype=torch.float32)
    grid = (s_emb.size(0), triton.cdiv(dim, BLOCK_K))
    _complex_score_spo_kernel[grid](
        s_emb,
        p_emb,
        o_emb,
        out,
        s_emb.stride(0),
        p_emb.stride(0),
        o_emb.stride(0),
        dim,
        BLOCK_SIZE=BLOCK_K,
    )
    return out


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
