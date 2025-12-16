"""
Optional Triton-based fused kernels used to update embeddings and optimizer
states in a single pass. Falls back to standard PyTorch ops when Triton or CUDA
is not available.
"""

from typing import Optional

import torch

try:  # pragma: no-cover - optional dependency
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no-cover - import guard
    triton = None
    tl = None
    _HAS_TRITON = False


MAX_BLOCK_SIZE = 512


if _HAS_TRITON:

    @triton.jit
    def _fused_scatter_kernel(
        param_ptr,
        optimizer_ptr,
        emb_update_ptr,
        opt_update_ptr,
        indices_ptr,
        dim,
        opt_dim,
        num_rows,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= num_rows:
            return
        row_idx = tl.load(indices_ptr + pid)
        row_idx = row_idx.to(tl.int64)
        offs = tl.arange(0, BLOCK_SIZE)

        # embedding updates
        if dim > 0:
            emb_row_offset = row_idx * dim
            emb_update_offset = pid * dim
            mask_emb = offs < dim
            emb_vals = tl.load(param_ptr + emb_row_offset + offs, mask=mask_emb, other=0.0)
            upd_vals = tl.load(emb_update_ptr + emb_update_offset + offs, mask=mask_emb, other=0.0)
            emb_vals += upd_vals
            tl.store(param_ptr + emb_row_offset + offs, emb_vals, mask=mask_emb)

        # optimizer updates (overwrite with new accumulator)
        if opt_dim > 0:
            opt_row_offset = row_idx * opt_dim
            opt_update_offset = pid * opt_dim
            mask_opt = offs < opt_dim
            opt_vals = tl.load(opt_update_ptr + opt_update_offset + offs, mask=mask_opt, other=0.0)
            tl.store(optimizer_ptr + opt_row_offset + offs, opt_vals, mask=mask_opt)


def fused_embedding_optimizer_update(
    embeddings: torch.Tensor,
    optimizer_values: torch.Tensor,
    indices: torch.Tensor,
    embedding_updates: torch.Tensor,
    optimizer_updates: torch.Tensor,
) -> None:
    """
    Applies embedding and optimizer updates for multiple rows in a single fused
    Triton kernel. Falls back to vanilla PyTorch ops when Triton is unavailable
    or tensors are on CPU.
    """

    if indices.numel() == 0:
        return

    if (
        not _HAS_TRITON
        or not embeddings.is_cuda
        or not embedding_updates.is_cuda
        or embeddings.shape[1] > MAX_BLOCK_SIZE
        or optimizer_values.shape[1] > MAX_BLOCK_SIZE
    ):
        embeddings.index_add_(0, indices, embedding_updates)
        optimizer_values.index_copy_(0, indices, optimizer_updates)
        return

    if not embeddings.is_contiguous():
        raise ValueError("Embeddings tensor must be contiguous for Triton kernel.")
    if not optimizer_values.is_contiguous():
        raise ValueError("Optimizer tensor must be contiguous for Triton kernel.")

    dim = embeddings.shape[1]
    opt_dim = optimizer_values.shape[1]

    emb_updates = embedding_updates.contiguous()
    opt_updates = optimizer_updates.contiguous()
    flat_embeddings = embeddings.view(-1)
    flat_optimizer = optimizer_values.view(-1)
    flat_emb_updates = emb_updates.view(-1)
    flat_opt_updates = opt_updates.view(-1) if opt_dim > 0 else torch.empty(0, device=embeddings.device)
    flat_indices = indices.contiguous().to(torch.int32)
    num_rows = flat_indices.numel()
    grid = (triton.cdiv(num_rows, 1),)
    _fused_scatter_kernel[grid](
        flat_embeddings,
        flat_optimizer,
        flat_emb_updates,
        flat_opt_updates,
        flat_indices,
        dim,
        opt_dim,
        num_rows,
        BLOCK_SIZE=MAX_BLOCK_SIZE,
    )
