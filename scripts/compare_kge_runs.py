#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(str(x).strip())
    except Exception:
        return None


def _fmt_s(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    if abs(x) >= 100:
        return f"{x:.1f}s"
    if abs(x) >= 10:
        return f"{x:.2f}s"
    if abs(x) >= 1:
        return f"{x:.3f}s"
    return f"{x:.4f}s"


def _fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{int(round(x)):,}"
    except Exception:
        return str(x)


def _fmt_bytes(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    v = float(x)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(v) < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(v)}{unit}"
            return f"{v:.2f}{unit}"
        v /= 1024.0
    return f"{v:.2f}TB"


def _delta(a: Optional[float], b: Optional[float], fmt=_fmt_s) -> str:
    if a is None or b is None:
        return "n/a"
    return fmt(b - a)


def _iter_trace_files(run_dir: Path) -> List[Path]:
    files = list(run_dir.glob("worker-*/trace.yaml"))
    if (run_dir / "trace.yaml").is_file():
        files.append(run_dir / "trace.yaml")
    return sorted(files)


def _iter_log_files(run_dir: Path) -> List[Path]:
    files = list(run_dir.glob("worker-*/kge.log"))
    if (run_dir / "kge.log").is_file():
        files.append(run_dir / "kge.log")
    return sorted(files)


def _parse_trace_file(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = yaml.safe_load(line)
        except Exception:
            continue
        if isinstance(entry, dict):
            yield entry


def _get_nested(cfg: Any, dotted_key: str) -> Any:
    cur = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _pick_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.is_file():
        return {}
    try:
        cfg = yaml.safe_load(cfg_path.read_text())
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _find_reduce_logs(run_dir: Path) -> Dict[str, Any]:
    logs = _iter_log_files(run_dir)
    out = {
        "reduce_active": False,
        "reduce_inactive": False,
        "reduce_reason": None,
    }
    active_re = re.compile(r"ComplEx reduce_by_key active")
    inactive_re = re.compile(r"reduce_by_key configured but inactive")
    for log in logs:
        for line in log.read_text(errors="replace").splitlines():
            if active_re.search(line):
                out["reduce_active"] = True
            if inactive_re.search(line):
                out["reduce_inactive"] = True
                out["reduce_reason"] = line.strip()
    return out


def _collect_epoch_entries(
    run_dir: Path,
    event: str,
    type_filter: Optional[str] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    by_epoch: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for trace in _iter_trace_files(run_dir):
        for entry in _parse_trace_file(trace):
            if entry.get("event") != event:
                continue
            if type_filter is not None and entry.get("type") != type_filter:
                continue
            ep = entry.get("epoch")
            try:
                ep_i = int(ep)
            except Exception:
                continue
            by_epoch[ep_i].append(entry)
    return dict(by_epoch)


def _aggregate_epoch(entries: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    if not entries:
        return {}
    keys = set()
    for e in entries:
        keys.update(e.keys())
    out: Dict[str, Any] = {"workers": len(entries)}
    for key in sorted(keys):
        vals = [_safe_float(e.get(key)) for e in entries]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        if mode == "sum":
            out[key] = sum(vals)
        elif mode == "max":
            out[key] = max(vals)
        elif mode == "mean":
            out[key] = sum(vals) / max(1, len(vals))
        else:  # mixed
            if key.endswith("_bytes") or key in (
                "processed_batches",
                "expected_batches",
                "processed_examples",
                "ps_pull_calls",
                "ps_set_calls",
                "ps_push_calls",
                "ps_pull_keys",
                "ps_set_keys",
                "ps_push_keys",
                "chunk_count",
                "batches",
            ):
                out[key] = sum(vals)
            else:
                out[key] = max(vals)
    return out


def _summarize_gpu_monitor(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.is_file():
        return {}
    per_gpu_util: Dict[int, List[float]] = defaultdict(list)
    per_gpu_mem: Dict[int, List[float]] = defaultdict(list)
    for line in path.read_text(errors="replace").splitlines():
        parts = line.strip().split(";")
        if len(parts) < 4:
            continue
        try:
            gpu = int(parts[1])
            util = float(parts[2])
            mem = float(parts[3])
        except Exception:
            continue
        per_gpu_util[gpu].append(util)
        per_gpu_mem[gpu].append(mem)

    def summarize(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {}
        vals_sorted = sorted(vals)
        p90 = vals_sorted[int(0.9 * (len(vals_sorted) - 1))]
        return {
            "mean": sum(vals) / len(vals),
            "p90": p90,
            "max": max(vals),
            "n": float(len(vals)),
        }

    out: Dict[str, Dict[str, float]] = {}
    for gpu in sorted(set(per_gpu_util.keys()) | set(per_gpu_mem.keys())):
        util_s = summarize(per_gpu_util.get(gpu, []))
        mem_s = summarize(per_gpu_mem.get(gpu, []))
        if util_s:
            out[f"gpu{gpu}.util"] = util_s
        if mem_s:
            out[f"gpu{gpu}.mem"] = mem_s
    return out


def _summarize_hw_monitor(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.is_file():
        return {}
    cols: Dict[int, List[float]] = defaultdict(list)
    for line in path.read_text(errors="replace").splitlines():
        parts = line.strip().split(";")
        if len(parts) < 2:
            continue
        for idx in range(1, len(parts)):
            try:
                cols[idx].append(float(parts[idx]))
            except Exception:
                pass

    def summarize(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {}
        vals_sorted = sorted(vals)
        p90 = vals_sorted[int(0.9 * (len(vals_sorted) - 1))]
        return {
            "mean": sum(vals) / len(vals),
            "p90": p90,
            "max": max(vals),
            "n": float(len(vals)),
        }

    names = {
        1: "cpu_pct",
        2: "ram_pct",
        3: "mem_used_mb",
        4: "mem_total_mb",
    }
    out: Dict[str, Dict[str, float]] = {}
    for idx, vals in cols.items():
        name = names.get(idx, f"col{idx}")
        out[name] = summarize(vals)
    return out


def summarize_run(run_dir: Path, mode: str) -> Dict[str, Any]:
    cfg = _pick_config(run_dir)
    reduce_info = _find_reduce_logs(run_dir)

    train_epochs_raw = _collect_epoch_entries(
        run_dir, event="epoch_completed", type_filter="negative_sampling"
    )
    train_epochs = {
        ep: _aggregate_epoch(entries, mode=mode) for ep, entries in train_epochs_raw.items()
    }
    eval_epochs_raw = _collect_epoch_entries(
        run_dir, event="eval_completed", type_filter="entity_ranking"
    )
    eval_epochs = {
        ep: _aggregate_epoch(entries, mode=mode) for ep, entries in eval_epochs_raw.items()
    }

    best_eval = None
    best_epoch = None
    best_key = None
    for ep, e in eval_epochs.items():
        for key in [
            "mean_reciprocal_rank_filtered",
            "mean_reciprocal_rank_filtered_with_test",
            "mean_reciprocal_rank",
        ]:
            val = _safe_float(e.get(key))
            if val is None:
                continue
            if best_eval is None or val > best_eval:
                best_eval = val
                best_epoch = ep
                best_key = key

    gpu_mon = _summarize_gpu_monitor(run_dir / "gpu_monitor.log")
    hw_mon = _summarize_hw_monitor(run_dir / "hardware_monitor.log")

    return {
        "folder": str(run_dir),
        "config": cfg,
        "reduce": reduce_info,
        "train_epochs": train_epochs,
        "eval_epochs": eval_epochs,
        "best_eval": {"epoch": best_epoch, "key": best_key, "value": best_eval},
        "gpu_monitor": gpu_mon,
        "hardware_monitor": hw_mon,
    }


def _print_config_block(a_cfg: Dict[str, Any], b_cfg: Dict[str, Any]) -> None:
    keys = [
        "job.distributed.partition_type",
        "job.distributed.entity_pre_pull",
        "job.distributed.relation_pre_pull",
        "job.distributed.map_ids_on_gpu",
        "job.distributed.sample_on_gpu",
        "job.distributed.entity_sync_level",
        "job.distributed.relation_sync_level",
        "job.distributed.materialize_partition_batches",
        "job.distributed.stage_local_ids",
        "job.distributed.optimizer.max_pending_pushes",
        "negative_sampling.sampling_type",
        "negative_sampling.num_samples",
        "train.batch_size",
        "train.subbatch_size",
        "lookup_embedder.dim",
        "distributed_model.base_model.reduce_by_key",
        "distributed_model.base_model.entity_embedder.sparse",
        "distributed_model.base_model.relation_embedder.sparse",
    ]
    print("Config (selected)")
    for k in keys:
        av = _get_nested(a_cfg, k)
        bv = _get_nested(b_cfg, k)
        if av is None and bv is None:
            continue
        print(f"- {k}: A={av} B={bv}")
    print("")


def _print_reduce_block(a: Dict[str, Any], b: Dict[str, Any]) -> None:
    print("complex_reduce")
    print(f"- A active: {bool(a.get('reduce_active'))}")
    if a.get("reduce_reason"):
        print(f"  A note: {a['reduce_reason']}")
    print(f"- B active: {bool(b.get('reduce_active'))}")
    if b.get("reduce_reason"):
        print(f"  B note: {b['reduce_reason']}")
    print("")


def _print_epoch_summary(label: str, ep: int, d: Dict[str, Any]) -> None:
    if not d:
        print(f"{label} epoch {ep}: n/a")
        return
    w = d.get("workers", "?")
    print(f"{label} epoch {ep} (workers={w})")
    for key in ["epoch_time", "forward_time", "backward_time", "optimizer_time", "other_time"]:
        if key in d:
            print(f"  {key}: {_fmt_s(_safe_float(d.get(key)))}")
    for key in ["expected_batches", "processed_batches", "processed_examples"]:
        if key in d:
            print(f"  {key}: {_fmt_num(_safe_float(d.get(key)))}")
    for key in ["ps_pull_bytes", "ps_set_bytes", "ps_push_bytes"]:
        if key in d:
            print(f"  {key}: {_fmt_bytes(_safe_float(d.get(key)))}")


def _print_eval_summary(label: str, ep: int, d: Dict[str, Any]) -> None:
    if not d:
        return
    for key in [
        "mean_reciprocal_rank_filtered",
        "mean_reciprocal_rank_filtered_with_test",
        "mean_reciprocal_rank",
        "hits_at_10_filtered",
        "hits_at_10",
    ]:
        if key in d:
            val = _safe_float(d.get(key))
            print(f"  {label} {key}: {val:.6f}" if val is not None else f"  {label} {key}: n/a")


def _print_monitor(title: str, a: Dict[str, Any], b: Dict[str, Any]) -> None:
    if not a and not b:
        return
    print(title)
    keys = sorted(set(a.keys()) | set(b.keys()))
    for k in keys:
        av = a.get(k, {})
        bv = b.get(k, {})
        if not isinstance(av, dict) or not isinstance(bv, dict):
            print(f"- {k}: A={av} B={bv}")
            continue
        print(
            f"- {k}: "
            f"A(mean={_safe_float(av.get('mean')):.2f} p90={_safe_float(av.get('p90')):.2f} max={_safe_float(av.get('max')):.2f} n={int(_safe_float(av.get('n')) or 0)}) "
            f"B(mean={_safe_float(bv.get('mean')):.2f} p90={_safe_float(bv.get('p90')):.2f} max={_safe_float(bv.get('max')):.2f} n={int(_safe_float(bv.get('n')) or 0)})"
        )
    print("")


def compare_runs(run_a: Dict[str, Any], run_b: Dict[str, Any], epochs: int) -> None:
    print("Runs")
    print(f"- A: {run_a['folder']}")
    print(f"- B: {run_b['folder']}")
    print("")

    _print_config_block(run_a["config"], run_b["config"])
    _print_reduce_block(run_a["reduce"], run_b["reduce"])

    best_a = run_a.get("best_eval", {})
    best_b = run_b.get("best_eval", {})
    if best_a.get("value") is not None or best_b.get("value") is not None:
        print("Best eval (by MRR)")
        print(
            f"- A: epoch={best_a.get('epoch')} key={best_a.get('key')} value={best_a.get('value')}"
        )
        print(
            f"- B: epoch={best_b.get('epoch')} key={best_b.get('key')} value={best_b.get('value')}"
        )
        print("")

    a_epochs = run_a.get("train_epochs", {})
    b_epochs = run_b.get("train_epochs", {})
    common = sorted(set(a_epochs.keys()) & set(b_epochs.keys()))
    if not common:
        print("No overlapping training epoch_completed entries found in both runs.")
    else:
        common = common[: max(1, epochs)]
        print(f"Train epochs (first {len(common)})")
        for ep in common:
            _print_epoch_summary("A", ep, a_epochs.get(ep, {}))
            _print_epoch_summary("B", ep, b_epochs.get(ep, {}))
            a = a_epochs.get(ep, {})
            b = b_epochs.get(ep, {})
            print("  deltas (B-A)")
            for k in ["epoch_time", "forward_time", "backward_time", "optimizer_time", "other_time"]:
                print(f"    {k}: {_delta(_safe_float(a.get(k)), _safe_float(b.get(k)))}")
            for k in ["ps_pull_bytes", "ps_set_bytes", "ps_push_bytes"]:
                print(
                    f"    {k}: {_delta(_safe_float(a.get(k)), _safe_float(b.get(k)), _fmt_bytes)}"
                )
            for k in ["expected_batches", "processed_batches", "processed_examples"]:
                print(
                    f"    {k}: {_delta(_safe_float(a.get(k)), _safe_float(b.get(k)), _fmt_num)}"
                )
            print("")

    a_eval = run_a.get("eval_epochs", {})
    b_eval = run_b.get("eval_epochs", {})
    eval_common = sorted(set(a_eval.keys()) & set(b_eval.keys()))
    if eval_common:
        eval_common = eval_common[: max(1, epochs)]
        print(f"Eval epochs (first {len(eval_common)})")
        for ep in eval_common:
            print(f"epoch {ep}")
            _print_eval_summary("A", ep, a_eval.get(ep, {}))
            _print_eval_summary("B", ep, b_eval.get(ep, {}))
            print("")

    _print_monitor("GPU monitor", run_a.get("gpu_monitor", {}), run_b.get("gpu_monitor", {}))
    _print_monitor(
        "Hardware monitor",
        run_a.get("hardware_monitor", {}),
        run_b.get("hardware_monitor", {}),
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare two KGE run folders (config, train/eval epochs, monitors)."
    )
    ap.add_argument("run_a", type=Path)
    ap.add_argument("run_b", type=Path)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument(
        "--aggregate",
        choices=["mixed", "mean", "max", "sum"],
        default="mixed",
        help="How to aggregate per-worker metrics per epoch.",
    )
    args = ap.parse_args()
    run_a = summarize_run(args.run_a, mode=args.aggregate)
    run_b = summarize_run(args.run_b, mode=args.aggregate)
    compare_runs(run_a, run_b, epochs=args.epochs)


if __name__ == "__main__":
    main()

