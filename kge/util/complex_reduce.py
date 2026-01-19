"""ComplEx reduce-by-key scoring for sparse embedding updates.

Forward is SDDMM-like (edge scoring on indexed triples or candidate blocks);
backward aggregates gradients with reduce-by-key (sort + segment reduce), which
is not a fused segmented-reduce kernel and returns indices in sorted order.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from kge.util.row_grad_cache import store_row_grad


def _apply_dropout(
    emb: torch.Tensor, p: float, training: bool
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if not training or p <= 0.0:
        return emb, None
    keep_prob = 1.0 - p
    mask = (torch.rand_like(emb) < keep_prob).to(emb.dtype) / keep_prob
    return emb * mask, mask


def _reduce_by_key(
    indices: torch.Tensor,
    values: torch.Tensor,
    num_rows: int,
    return_sparse: bool = True,
):
    if indices.numel() == 0:
        if return_sparse:
            return torch.sparse_coo_tensor(
                torch.empty((1, 0), device=values.device, dtype=torch.long),
                torch.empty(
                    (0, values.size(1)), device=values.device, dtype=values.dtype
                ),
                size=(num_rows, values.size(1)),
            )
        return (
            torch.empty((0,), device=values.device, dtype=torch.long),
            torch.empty((0, values.size(1)), device=values.device, dtype=values.dtype),
        )
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
    if return_sparse:
        # Indices are strictly increasing and unique here, so the tensor is already
        # coalesced. Use is_coalesced=True when supported to avoid an extra
        # coalesce() (which shows up as coalesceValuesKernel + sort in profiles).
        try:
            return torch.sparse_coo_tensor(
                unique.unsqueeze(0),
                out,
                size=(num_rows, values.size(1)),
                is_coalesced=True,
            )
        except TypeError:
            return torch.sparse_coo_tensor(
                unique.unsqueeze(0),
                out,
                size=(num_rows, values.size(1)),
            ).coalesce()
    return unique, out


def _split_complex(emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return emb.chunk(2, dim=1)


def _score_parts(
    s_re: torch.Tensor,
    s_im: torch.Tensor,
    p_re: torch.Tensor,
    p_im: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    a = s_re * p_re - s_im * p_im
    b = s_im * p_re + s_re * p_im
    return a, b


class _ComplExScoreSpoReduceFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        s_idx: torch.Tensor,
        p_idx: torch.Tensor,
        o_idx: torch.Tensor,
        s_weight: torch.Tensor,
        p_weight: torch.Tensor,
        o_weight: torch.Tensor,
        s_dropout: float,
        p_dropout: float,
        o_dropout: float,
        training: bool,
        cache_row_grads: bool,
    ) -> torch.Tensor:
        s_idx = s_idx.reshape(-1).long()
        p_idx = p_idx.reshape(-1).long()
        o_idx = o_idx.reshape(-1).long()
        if s_idx.device != s_weight.device:
            s_idx = s_idx.to(s_weight.device)
        if p_idx.device != p_weight.device:
            p_idx = p_idx.to(p_weight.device)
        if o_idx.device != o_weight.device:
            o_idx = o_idx.to(o_weight.device)

        s_emb = s_weight.index_select(0, s_idx)
        p_emb = p_weight.index_select(0, p_idx)
        o_emb = o_weight.index_select(0, o_idx)

        s_emb, s_mask = _apply_dropout(s_emb, s_dropout, training)
        p_emb, p_mask = _apply_dropout(p_emb, p_dropout, training)
        o_emb, o_mask = _apply_dropout(o_emb, o_dropout, training)

        s_re, s_im = _split_complex(s_emb)
        p_re, p_im = _split_complex(p_emb)
        o_re, o_im = _split_complex(o_emb)
        a, b = _score_parts(s_re, s_im, p_re, p_im)
        scores = (a * o_re + b * o_im).sum(dim=1, keepdim=True)

        ctx.save_for_backward(
            s_idx,
            p_idx,
            o_idx,
            s_emb,
            p_emb,
            o_emb,
            s_mask if s_mask is not None else torch.tensor([], device=s_emb.device),
            p_mask if p_mask is not None else torch.tensor([], device=p_emb.device),
            o_mask if o_mask is not None else torch.tensor([], device=o_emb.device),
        )
        ctx.s_weight_shape = s_weight.shape
        ctx.p_weight_shape = p_weight.shape
        ctx.o_weight_shape = o_weight.shape
        ctx.s_dropout = s_dropout
        ctx.p_dropout = p_dropout
        ctx.o_dropout = o_dropout
        ctx.training = training
        ctx.cache_row_grads = bool(cache_row_grads)
        ctx.s_weight_ref = s_weight
        ctx.p_weight_ref = p_weight
        ctx.o_weight_ref = o_weight
        return scores

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (
            s_idx,
            p_idx,
            o_idx,
            s_emb,
            p_emb,
            o_emb,
            s_mask,
            p_mask,
            o_mask,
        ) = ctx.saved_tensors
        grad_out = grad_out.view(-1, 1)

        s_re, s_im = _split_complex(s_emb)
        p_re, p_im = _split_complex(p_emb)
        o_re, o_im = _split_complex(o_emb)
        a, b = _score_parts(s_re, s_im, p_re, p_im)

        grad_s_re = grad_out * (p_re * o_re + p_im * o_im)
        grad_s_im = grad_out * (-p_im * o_re + p_re * o_im)
        grad_p_re = grad_out * (s_re * o_re + s_im * o_im)
        grad_p_im = grad_out * (-s_im * o_re + s_re * o_im)
        grad_o_re = grad_out * a
        grad_o_im = grad_out * b

        grad_s = torch.cat((grad_s_re, grad_s_im), dim=1)
        grad_p = torch.cat((grad_p_re, grad_p_im), dim=1)
        grad_o = torch.cat((grad_o_re, grad_o_im), dim=1)

        if ctx.training and s_mask.numel() > 0:
            grad_s = grad_s * s_mask
        if ctx.training and p_mask.numel() > 0:
            grad_p = grad_p * p_mask
        if ctx.training and o_mask.numel() > 0:
            grad_o = grad_o * o_mask

        if ctx.cache_row_grads:
            s_rows, s_values = _reduce_by_key(
                s_idx, grad_s, ctx.s_weight_shape[0], return_sparse=False
            )
            p_rows, p_values = _reduce_by_key(
                p_idx, grad_p, ctx.p_weight_shape[0], return_sparse=False
            )
            o_rows, o_values = _reduce_by_key(
                o_idx, grad_o, ctx.o_weight_shape[0], return_sparse=False
            )
            store_row_grad(ctx.s_weight_ref, s_rows, s_values)
            store_row_grad(ctx.p_weight_ref, p_rows, p_values)
            store_row_grad(ctx.o_weight_ref, o_rows, o_values)
            grad_s_weight = None
            grad_p_weight = None
            grad_o_weight = None
        else:
            grad_s_weight = _reduce_by_key(s_idx, grad_s, ctx.s_weight_shape[0])
            grad_p_weight = _reduce_by_key(p_idx, grad_p, ctx.p_weight_shape[0])
            grad_o_weight = _reduce_by_key(o_idx, grad_o, ctx.o_weight_shape[0])

        return (
            None,  # s_idx
            None,  # p_idx
            None,  # o_idx
            grad_s_weight,  # s_weight
            grad_p_weight,  # p_weight
            grad_o_weight,  # o_weight
            None,  # s_dropout
            None,  # p_dropout
            None,  # o_dropout
            None,  # training
            None,  # cache_row_grads
        )


class _ComplExScoreSpReduceFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        s_idx: torch.Tensor,
        p_idx: torch.Tensor,
        o_idx: torch.Tensor,
        s_weight: torch.Tensor,
        p_weight: torch.Tensor,
        o_weight: torch.Tensor,
        s_dropout: float,
        p_dropout: float,
        o_dropout: float,
        training: bool,
        cache_row_grads: bool,
    ) -> torch.Tensor:
        s_idx = s_idx.reshape(-1).long()
        p_idx = p_idx.reshape(-1).long()
        o_idx = o_idx.reshape(-1).long()
        if s_idx.device != s_weight.device:
            s_idx = s_idx.to(s_weight.device)
        if p_idx.device != p_weight.device:
            p_idx = p_idx.to(p_weight.device)
        if o_idx.device != o_weight.device:
            o_idx = o_idx.to(o_weight.device)

        s_emb = s_weight.index_select(0, s_idx)
        p_emb = p_weight.index_select(0, p_idx)
        o_emb = o_weight.index_select(0, o_idx)

        s_emb, s_mask = _apply_dropout(s_emb, s_dropout, training)
        p_emb, p_mask = _apply_dropout(p_emb, p_dropout, training)
        o_emb, o_mask = _apply_dropout(o_emb, o_dropout, training)

        s_re, s_im = _split_complex(s_emb)
        p_re, p_im = _split_complex(p_emb)
        o_re, o_im = _split_complex(o_emb)
        a, b = _score_parts(s_re, s_im, p_re, p_im)
        scores = a @ o_re.transpose(0, 1) + b @ o_im.transpose(0, 1)

        ctx.save_for_backward(
            s_idx,
            p_idx,
            o_idx,
            s_emb,
            p_emb,
            o_emb,
            s_mask if s_mask is not None else torch.tensor([], device=s_emb.device),
            p_mask if p_mask is not None else torch.tensor([], device=p_emb.device),
            o_mask if o_mask is not None else torch.tensor([], device=o_emb.device),
        )
        ctx.s_weight_shape = s_weight.shape
        ctx.p_weight_shape = p_weight.shape
        ctx.o_weight_shape = o_weight.shape
        ctx.training = training
        ctx.cache_row_grads = bool(cache_row_grads)
        ctx.s_weight_ref = s_weight
        ctx.p_weight_ref = p_weight
        ctx.o_weight_ref = o_weight
        return scores

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (
            s_idx,
            p_idx,
            o_idx,
            s_emb,
            p_emb,
            o_emb,
            s_mask,
            p_mask,
            o_mask,
        ) = ctx.saved_tensors

        s_re, s_im = _split_complex(s_emb)
        p_re, p_im = _split_complex(p_emb)
        o_re, o_im = _split_complex(o_emb)
        a, b = _score_parts(s_re, s_im, p_re, p_im)

        grad_a = grad_out @ o_re
        grad_b = grad_out @ o_im

        grad_s_re = grad_a * p_re + grad_b * p_im
        grad_s_im = grad_a * (-p_im) + grad_b * p_re
        grad_p_re = grad_a * s_re + grad_b * s_im
        grad_p_im = grad_a * (-s_im) + grad_b * s_re
        grad_o_re = grad_out.transpose(0, 1) @ a
        grad_o_im = grad_out.transpose(0, 1) @ b

        grad_s = torch.cat((grad_s_re, grad_s_im), dim=1)
        grad_p = torch.cat((grad_p_re, grad_p_im), dim=1)
        grad_o = torch.cat((grad_o_re, grad_o_im), dim=1)

        if ctx.training and s_mask.numel() > 0:
            grad_s = grad_s * s_mask
        if ctx.training and p_mask.numel() > 0:
            grad_p = grad_p * p_mask
        if ctx.training and o_mask.numel() > 0:
            grad_o = grad_o * o_mask

        if ctx.cache_row_grads:
            s_rows, s_values = _reduce_by_key(
                s_idx, grad_s, ctx.s_weight_shape[0], return_sparse=False
            )
            p_rows, p_values = _reduce_by_key(
                p_idx, grad_p, ctx.p_weight_shape[0], return_sparse=False
            )
            o_rows, o_values = _reduce_by_key(
                o_idx, grad_o, ctx.o_weight_shape[0], return_sparse=False
            )
            store_row_grad(ctx.s_weight_ref, s_rows, s_values)
            store_row_grad(ctx.p_weight_ref, p_rows, p_values)
            store_row_grad(ctx.o_weight_ref, o_rows, o_values)
            grad_s_weight = None
            grad_p_weight = None
            grad_o_weight = None
        else:
            grad_s_weight = _reduce_by_key(s_idx, grad_s, ctx.s_weight_shape[0])
            grad_p_weight = _reduce_by_key(p_idx, grad_p, ctx.p_weight_shape[0])
            grad_o_weight = _reduce_by_key(o_idx, grad_o, ctx.o_weight_shape[0])

        return (
            None,  # s_idx
            None,  # p_idx
            None,  # o_idx
            grad_s_weight,  # s_weight
            grad_p_weight,  # p_weight
            grad_o_weight,  # o_weight
            None,  # s_dropout
            None,  # p_dropout
            None,  # o_dropout
            None,  # training
            None,  # cache_row_grads
        )


class _ComplExScorePoReduceFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        p_idx: torch.Tensor,
        o_idx: torch.Tensor,
        s_idx: torch.Tensor,
        p_weight: torch.Tensor,
        o_weight: torch.Tensor,
        s_weight: torch.Tensor,
        p_dropout: float,
        o_dropout: float,
        s_dropout: float,
        training: bool,
        cache_row_grads: bool,
    ) -> torch.Tensor:
        p_idx = p_idx.reshape(-1).long()
        o_idx = o_idx.reshape(-1).long()
        s_idx = s_idx.reshape(-1).long()
        if p_idx.device != p_weight.device:
            p_idx = p_idx.to(p_weight.device)
        if o_idx.device != o_weight.device:
            o_idx = o_idx.to(o_weight.device)
        if s_idx.device != s_weight.device:
            s_idx = s_idx.to(s_weight.device)

        p_emb = p_weight.index_select(0, p_idx)
        o_emb = o_weight.index_select(0, o_idx)
        s_emb = s_weight.index_select(0, s_idx)

        p_emb, p_mask = _apply_dropout(p_emb, p_dropout, training)
        o_emb, o_mask = _apply_dropout(o_emb, o_dropout, training)
        s_emb, s_mask = _apply_dropout(s_emb, s_dropout, training)

        s_re, s_im = _split_complex(s_emb)
        p_re, p_im = _split_complex(p_emb)
        o_re, o_im = _split_complex(o_emb)
        c = p_re * o_re + p_im * o_im
        d = p_re * o_im - p_im * o_re
        scores = c @ s_re.transpose(0, 1) + d @ s_im.transpose(0, 1)

        ctx.save_for_backward(
            p_idx,
            o_idx,
            s_idx,
            p_emb,
            o_emb,
            s_emb,
            p_mask if p_mask is not None else torch.tensor([], device=p_emb.device),
            o_mask if o_mask is not None else torch.tensor([], device=o_emb.device),
            s_mask if s_mask is not None else torch.tensor([], device=s_emb.device),
        )
        ctx.p_weight_shape = p_weight.shape
        ctx.o_weight_shape = o_weight.shape
        ctx.s_weight_shape = s_weight.shape
        ctx.training = training
        ctx.cache_row_grads = bool(cache_row_grads)
        ctx.p_weight_ref = p_weight
        ctx.o_weight_ref = o_weight
        ctx.s_weight_ref = s_weight
        return scores

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (
            p_idx,
            o_idx,
            s_idx,
            p_emb,
            o_emb,
            s_emb,
            p_mask,
            o_mask,
            s_mask,
        ) = ctx.saved_tensors

        s_re, s_im = _split_complex(s_emb)
        p_re, p_im = _split_complex(p_emb)
        o_re, o_im = _split_complex(o_emb)
        c = p_re * o_re + p_im * o_im
        d = p_re * o_im - p_im * o_re

        grad_c = grad_out @ s_re
        grad_d = grad_out @ s_im

        grad_p_re = grad_c * o_re + grad_d * o_im
        grad_p_im = grad_c * o_im + grad_d * (-o_re)
        grad_o_re = grad_c * p_re + grad_d * (-p_im)
        grad_o_im = grad_c * p_im + grad_d * p_re
        grad_s_re = grad_out.transpose(0, 1) @ c
        grad_s_im = grad_out.transpose(0, 1) @ d

        grad_p = torch.cat((grad_p_re, grad_p_im), dim=1)
        grad_o = torch.cat((grad_o_re, grad_o_im), dim=1)
        grad_s = torch.cat((grad_s_re, grad_s_im), dim=1)

        if ctx.training and p_mask.numel() > 0:
            grad_p = grad_p * p_mask
        if ctx.training and o_mask.numel() > 0:
            grad_o = grad_o * o_mask
        if ctx.training and s_mask.numel() > 0:
            grad_s = grad_s * s_mask

        if ctx.cache_row_grads:
            p_rows, p_values = _reduce_by_key(
                p_idx, grad_p, ctx.p_weight_shape[0], return_sparse=False
            )
            o_rows, o_values = _reduce_by_key(
                o_idx, grad_o, ctx.o_weight_shape[0], return_sparse=False
            )
            s_rows, s_values = _reduce_by_key(
                s_idx, grad_s, ctx.s_weight_shape[0], return_sparse=False
            )
            store_row_grad(ctx.p_weight_ref, p_rows, p_values)
            store_row_grad(ctx.o_weight_ref, o_rows, o_values)
            store_row_grad(ctx.s_weight_ref, s_rows, s_values)
            grad_p_weight = None
            grad_o_weight = None
            grad_s_weight = None
        else:
            grad_p_weight = _reduce_by_key(p_idx, grad_p, ctx.p_weight_shape[0])
            grad_o_weight = _reduce_by_key(o_idx, grad_o, ctx.o_weight_shape[0])
            grad_s_weight = _reduce_by_key(s_idx, grad_s, ctx.s_weight_shape[0])

        return (
            None,  # p_idx
            None,  # o_idx
            None,  # s_idx
            grad_p_weight,  # p_weight
            grad_o_weight,  # o_weight
            grad_s_weight,  # s_weight
            None,  # p_dropout
            None,  # o_dropout
            None,  # s_dropout
            None,  # training
            None,  # cache_row_grads
        )


def complex_score_spo_reduce(
    s_idx: torch.Tensor,
    p_idx: torch.Tensor,
    o_idx: torch.Tensor,
    s_weight: torch.Tensor,
    p_weight: torch.Tensor,
    o_weight: torch.Tensor,
    s_dropout: float,
    p_dropout: float,
    o_dropout: float,
    training: bool,
    cache_row_grads: bool = False,
) -> torch.Tensor:
    return _ComplExScoreSpoReduceFn.apply(
        s_idx,
        p_idx,
        o_idx,
        s_weight,
        p_weight,
        o_weight,
        float(s_dropout),
        float(p_dropout),
        float(o_dropout),
        training,
        cache_row_grads,
    ).view(-1)


def complex_score_sp_reduce(
    s_idx: torch.Tensor,
    p_idx: torch.Tensor,
    o_idx: torch.Tensor,
    s_weight: torch.Tensor,
    p_weight: torch.Tensor,
    o_weight: torch.Tensor,
    s_dropout: float,
    p_dropout: float,
    o_dropout: float,
    training: bool,
    cache_row_grads: bool = False,
) -> torch.Tensor:
    return _ComplExScoreSpReduceFn.apply(
        s_idx,
        p_idx,
        o_idx,
        s_weight,
        p_weight,
        o_weight,
        float(s_dropout),
        float(p_dropout),
        float(o_dropout),
        training,
        cache_row_grads,
    )


def complex_score_po_reduce(
    p_idx: torch.Tensor,
    o_idx: torch.Tensor,
    s_idx: torch.Tensor,
    p_weight: torch.Tensor,
    o_weight: torch.Tensor,
    s_weight: torch.Tensor,
    p_dropout: float,
    o_dropout: float,
    s_dropout: float,
    training: bool,
    cache_row_grads: bool = False,
) -> torch.Tensor:
    return _ComplExScorePoReduceFn.apply(
        p_idx,
        o_idx,
        s_idx,
        p_weight,
        o_weight,
        s_weight,
        float(p_dropout),
        float(o_dropout),
        float(s_dropout),
        training,
        cache_row_grads,
    )
