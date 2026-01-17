#!/usr/bin/env python3
"""
Sanity-check ComplEx scoring algebra against the previous "4d cat trick".

This script does not depend on the kge package; it only requires PyTorch.
Run it inside the same environment you use for training (where torch is installed).
"""

from __future__ import annotations

import argparse
import sys


def _require_torch():
    try:
        import torch  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "PyTorch is required. Run inside your training conda env.\n"
            f"Import error: {exc}"
        ) from exc


def score_cat_trick(s_emb, p_emb, o_emb, combine: str):
    import torch

    n = p_emb.size(0)
    p_emb_re, p_emb_im = (t.contiguous() for t in p_emb.chunk(2, dim=1))
    o_emb_re, o_emb_im = (t.contiguous() for t in o_emb.chunk(2, dim=1))

    s_all = torch.cat((s_emb, s_emb), dim=1)  # re, im, re, im
    r_all = torch.cat((p_emb_re, p_emb, -p_emb_im), dim=1)  # re, re, im, -im
    o_all = torch.cat((o_emb, o_emb_im, o_emb_re), dim=1)  # re, im, im, re

    if combine == "spo":
        out = (s_all * o_all * r_all).sum(dim=1, keepdim=True)
    elif combine == "sp_":
        out = (s_all * r_all).mm(o_all.transpose(0, 1))
    elif combine == "_po":
        out = (r_all * o_all).mm(s_all.transpose(0, 1))
    else:
        raise ValueError(f"unsupported combine={combine!r}")

    return out.view(n, -1)


def score_formula(s_emb, p_emb, o_emb, combine: str):
    import torch

    n = p_emb.size(0)
    s_re, s_im = (t.contiguous() for t in s_emb.chunk(2, dim=1))
    p_re, p_im = (t.contiguous() for t in p_emb.chunk(2, dim=1))
    o_re, o_im = (t.contiguous() for t in o_emb.chunk(2, dim=1))

    if combine == "spo":
        a = s_re * p_re - s_im * p_im
        b = s_re * p_im + s_im * p_re
        out = (a * o_re + b * o_im).sum(dim=1, keepdim=True)
    elif combine == "sp_":
        a = s_re * p_re - s_im * p_im
        b = s_re * p_im + s_im * p_re
        out = a.mm(o_re.transpose(0, 1))
        out.addmm_(b, o_im.transpose(0, 1))
    elif combine == "_po":
        a = p_re * o_re + p_im * o_im
        b = p_im * o_re - p_re * o_im
        out = a.mm(s_re.transpose(0, 1))
        out.addmm_(b, s_im.transpose(0, 1), alpha=-1.0)
    else:
        raise ValueError(f"unsupported combine={combine!r}")

    return out.view(n, -1)


def main(argv: list[str]) -> int:
    _require_torch()
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=7, help="real embedding dim")
    parser.add_argument("--n", type=int, default=5, help="batch size (n)")
    parser.add_argument("--m", type=int, default=11, help="candidate size (m)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu|cuda (if available in your env)",
    )
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    d = args.d
    n = args.n
    m = args.m

    def rand(shape):
        return torch.randn(*shape, device=device, dtype=torch.float64, requires_grad=True)

    # spo: n x 2d, n x 2d, n x 2d
    s = rand((n, 2 * d))
    p = rand((n, 2 * d))
    o = rand((n, 2 * d))
    for combine, (s_emb, p_emb, o_emb) in (
        ("spo", (s, p, o)),
        ("sp_", (rand((n, 2 * d)), rand((n, 2 * d)), rand((m, 2 * d)))),
        ("_po", (rand((m, 2 * d)), rand((n, 2 * d)), rand((n, 2 * d)))),
    ):
        out_old = score_cat_trick(s_emb, p_emb, o_emb, combine)
        out_new = score_formula(s_emb, p_emb, o_emb, combine)
        max_abs = (out_old - out_new).abs().max().item()
        denom = out_old.abs().max().clamp_min(1e-12)
        max_rel = ((out_old - out_new).abs() / denom).max().item()
        print(f"{combine}: max_abs={max_abs:.3e} max_rel={max_rel:.3e} shape={tuple(out_old.shape)}")

        # Backward check (sum is enough to touch all entries)
        loss_old = out_old.sum()
        loss_new = out_new.sum()
        grads_old = torch.autograd.grad(loss_old, (s_emb, p_emb, o_emb), retain_graph=True)
        grads_new = torch.autograd.grad(loss_new, (s_emb, p_emb, o_emb), retain_graph=True)
        for name, go, gn in zip(("s", "p", "o"), grads_old, grads_new):
            g_abs = (go - gn).abs().max().item()
            g_denom = go.abs().max().clamp_min(1e-12)
            g_rel = ((go - gn).abs() / g_denom).max().item()
            print(f"  grad {name}: max_abs={g_abs:.3e} max_rel={g_rel:.3e}")

        # Clean up graph references to avoid accidental re-use in next loop.
        del out_old, out_new, grads_old, grads_new, loss_old, loss_new

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

