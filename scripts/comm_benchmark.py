#!/usr/bin/env python3
"""Synthetic communication benchmark for row-wise partition broadcast + reduce.

Pattern per row i (owner rank i):
  1) broadcast partition Pi embeddings to all ranks
  2) each rank computes gradients for Pi (from Pi,Pj) and for its local Pj
  3) reduce Pi gradients back to owner rank i
  4) owner i applies Pi update locally; each rank applies Pj update locally

This approximates processing row i of an 8x8 partition matrix.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency for uniform sizes
    np = None


class _NvtxRange:
    def __init__(self, enabled: bool, name: str):
        self.enabled = enabled
        self.name = name

    def __enter__(self):
        if self.enabled and torch.cuda.is_available():
            torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.enabled and torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()
        return False


def _parse_rows(rows_arg: str, num_parts: int):
    if rows_arg == "all":
        return list(range(num_parts))
    rows = []
    for token in rows_arg.split(","):
        token = token.strip()
        if not token:
            continue
        rows.append(int(token))
    for r in rows:
        if r < 0 or r >= num_parts:
            raise ValueError(f"row index {r} out of range [0, {num_parts - 1}]")
    return rows


def _count_partitions_from_array(arr, num_parts: int):
    import numpy as np  # local import to avoid hard dependency at module level
    counts = np.zeros((num_parts,), dtype=np.int64)
    if arr.ndim == 1:
        vals = arr
    else:
        vals = arr.reshape(-1)
    vals = vals[(vals >= 0) & (vals < num_parts)]
    if vals.size == 0:
        return counts.tolist()
    uniq, cnt = np.unique(vals, return_counts=True)
    counts[uniq] += cnt
    return counts.tolist()


def _stream_count_partitions(del_path: Path, num_parts: int):
    counts = [0] * num_parts
    with del_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            for token in parts:
                if not token:
                    continue
                try:
                    pid = int(token)
                except ValueError:
                    continue
                if 0 <= pid < num_parts:
                    counts[pid] += 1
    return counts


def _load_sizes_from_partition_dir(part_dir: Path, num_parts: int):
    # Prefer pickled arrays if present
    pckl = part_dir / "entity_to_partitions.del.pckl"
    if pckl.exists():
        import pickle
        with pckl.open("rb") as f:
            arr = pickle.load(f)
        try:
            import numpy as np
            arr = np.asarray(arr)
            return _count_partitions_from_array(arr, num_parts)
        except Exception:
            pass
    # Fallback to raw .del file (streaming)
    del_file = part_dir / "entity_to_partitions.del"
    if del_file.exists():
        return _stream_count_partitions(del_file, num_parts)
    return None


def _load_partition_sizes(args, world_size: int):
    if not args.override_partitions:
        if args.partition_sizes or args.num_entities is not None:
            raise ValueError(
                "partition sizes are fixed by default; use --override-partitions to customize"
            )
        if args.scale != 1.0 or args.max_entities is not None:
            raise ValueError(
                "partition scaling is disabled by default; use --override-partitions to customize"
            )
    if args.partition_npz:
        path = Path(args.partition_npz)
        if path.exists():
            if np is None:
                raise RuntimeError("numpy is required to read --partition-npz")
            data = np.load(path)
            sizes = []
            for i in range(world_size):
                key = f"entity_part_{i}"
                if key not in data:
                    raise KeyError(f"{key} not found in {path}")
                sizes.append(int(data[key].shape[0]))
            return sizes
    # Fallback to partition directory mapping
    if args.partition_dir:
        part_dir = Path(args.partition_dir)
    else:
        part_dir = Path("data") / "wikidata5m" / "partitions" / "stratification" / f"num_{world_size}"
    sizes = _load_sizes_from_partition_dir(part_dir, world_size)
    if sizes is not None:
        return sizes
    elif args.partition_sizes:
        sizes = [int(x) for x in args.partition_sizes.split(",") if x.strip()]
        if len(sizes) != world_size:
            raise ValueError(
                f"--partition-sizes expects {world_size} values, got {len(sizes)}"
            )
    else:
        if args.num_entities is None:
            raise ValueError("provide --partition-npz, --partition-sizes, or --num-entities")
        base = int(args.num_entities // world_size)
        sizes = [base] * world_size
        sizes[-1] += int(args.num_entities - base * world_size)

    if args.scale != 1.0:
        sizes = [max(1, int(s * args.scale)) for s in sizes]
    if args.max_entities is not None:
        sizes = [min(s, args.max_entities) for s in sizes]

    return sizes


def _tensor_bytes(num_entities: int, dim: int, dtype: torch.dtype):
    return int(num_entities) * int(dim) * torch.tensor([], dtype=dtype).element_size()


def _human_bytes(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f}PB"


def _init_dist(backend: str):
    if dist.is_initialized():
        return
    dist.init_process_group(backend=backend, init_method="env://")


def _get_device(args):
    if args.device == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    if args.local_rank is None:
        env_rank = os.environ.get("LOCAL_RANK")
        local_rank = int(env_rank) if env_rank is not None else 0
    else:
        local_rank = args.local_rank

    if args.device_id is not None:
        device_id = args.device_id
    else:
        num_devices = torch.cuda.device_count()
        if num_devices == 0:
            raise RuntimeError("no CUDA devices found")
        device_id = local_rank % num_devices

    torch.cuda.set_device(device_id)
    return torch.device("cuda", device_id)


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _all_reduce_stats(value: float, device: torch.device):
    t_sum = torch.tensor([value], device=device, dtype=torch.float64)
    t_max = t_sum.clone()
    dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
    avg = (t_sum / dist.get_world_size()).item()
    mx = t_max.item()
    return avg, mx


def run_benchmark(args):
    device = _get_device(args)
    _init_dist(args.backend)

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if args.logical_workers is not None:
        num_parts = int(args.logical_workers)
    elif args.num_partitions is None:
        num_parts = world_size
    else:
        num_parts = args.num_partitions
    if not args.override_partitions:
        if num_parts != 8:
            raise ValueError(
                f"benchmark is fixed to 8 logical workers/partitions; got num_partitions={num_parts}. "
                "Use --override-partitions to customize."
            )
        if num_parts % world_size != 0:
            raise ValueError(
                f"logical workers must be divisible by world_size; got num_partitions={num_parts}, world_size={world_size}. "
                "Use --override-partitions to customize."
            )

    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if args.dtype not in dtype_map:
        raise ValueError(f"unsupported dtype {args.dtype}")
    dtype = dtype_map[args.dtype]

    sizes = _load_partition_sizes(args, num_parts)
    rows = _parse_rows(args.rows, num_parts)

    if rank == 0:
        print("comm_benchmark config:")
        print(f"  world_size: {world_size}")
        print(f"  logical_workers: {num_parts}")
        print(f"  backend: {args.backend}")
        print(f"  device: {device}")
        print(f"  embedding_dim: {args.embedding_dim}")
        print(f"  dtype: {args.dtype}")
        print(f"  rows: {rows}")
        print(f"  warmup: {args.warmup}")
        print(f"  iters: {args.iters}")
        print(f"  compute: {args.compute}")
        print(f"  row_lr: {args.row_lr}")
        print(f"  local_lr: {args.local_lr}")
        print(f"  row_unique_fraction: {args.row_unique_fraction}")
        print(f"  row_unique_ids: {args.row_unique_ids}")
        print(f"  include_id_bytes: {args.include_id_bytes}")
        print(f"  nvtx: {args.nvtx}")
        print(f"  sizes: {sizes}")

    # local partitions owned by this rank (logical workers)
    local_pids = [pid for pid in range(num_parts) if pid % world_size == rank]
    local_tensors = {}
    local_opts = {}
    for pid in local_pids:
        ent = sizes[pid]
        tensor = torch.empty((ent, args.embedding_dim), device=device, dtype=dtype)
        tensor.uniform_(-1, 1)
        local_tensors[pid] = tensor
        local_opts[pid] = torch.zeros((ent, args.embedding_dim), device=device, dtype=torch.float32)
    local_worker_count = max(1, len(local_pids))

    results = []

    for row in rows:
        owner_rank = row % world_size
        row_entities = sizes[row]
        if args.row_unique_ids is not None:
            reduce_entities = int(args.row_unique_ids)
        else:
            reduce_entities = int(row_entities * args.row_unique_fraction)
        if reduce_entities < 1:
            reduce_entities = 1
        if reduce_entities > row_entities:
            reduce_entities = row_entities

        bcast_bytes = _tensor_bytes(row_entities, args.embedding_dim, dtype)
        reduce_bytes = _tensor_bytes(reduce_entities, args.embedding_dim, dtype)
        if args.include_id_bytes:
            reduce_bytes += int(reduce_entities) * int(args.id_bytes)

        row_tensor = torch.empty((row_entities, args.embedding_dim), device=device, dtype=dtype)
        if rank == owner_rank:
            row_tensor.copy_(local_tensors[row])
        else:
            row_tensor.zero_()

        grad_row = torch.empty((reduce_entities, args.embedding_dim), device=device, dtype=dtype)
        local_grads = {
            pid: torch.empty_like(local_tensors[pid]) for pid in local_pids
        }

        # warmup
        for _ in range(args.warmup):
            with _NvtxRange(args.nvtx, f"row/{row}/broadcast"):
                dist.broadcast(row_tensor, src=owner_rank)
            if args.compute:
                with _NvtxRange(args.nvtx, f"row/{row}/compute"):
                    grad_row.copy_(row_tensor[:reduce_entities])
                    grad_row.mul_(0.001)
                    grad_row.mul_(float(local_worker_count))
                    for pid in local_pids:
                        local_grads[pid].copy_(local_tensors[pid])
                        local_grads[pid].mul_(0.001)
                    # update optimizer state locally (no communication)
                    for pid in local_pids:
                        local_opts[pid].add_(local_grads[pid].float().pow(2))
                    if rank == (row % world_size):
                        local_opts[row][:reduce_entities].add_(grad_row.float().pow(2))
                    if rank == owner_rank:
                        row_tensor[:reduce_entities].add_(grad_row, alpha=-args.row_lr)
                    for pid in local_pids:
                        local_tensors[pid].add_(local_grads[pid], alpha=-args.local_lr)
            else:
                grad_row.copy_(row_tensor[:reduce_entities])
            with _NvtxRange(args.nvtx, f"row/{row}/reduce"):
                dist.reduce(grad_row, dst=owner_rank, op=dist.ReduceOp.SUM)
            if rank == owner_rank:
                local_tensors[row][:reduce_entities].copy_(row_tensor[:reduce_entities])

        dist.barrier()

        bcast_total = 0.0
        reduce_total = 0.0

        for _ in range(args.iters):
            _sync(device)
            t0 = time.perf_counter()
            with _NvtxRange(args.nvtx, f"row/{row}/broadcast"):
                dist.broadcast(row_tensor, src=owner_rank)
            _sync(device)
            t1 = time.perf_counter()

            if args.compute:
                with _NvtxRange(args.nvtx, f"row/{row}/compute"):
                    grad_row.copy_(row_tensor[:reduce_entities])
                    grad_row.mul_(0.001)
                    grad_row.mul_(float(local_worker_count))
                    for pid in local_pids:
                        local_grads[pid].copy_(local_tensors[pid])
                        local_grads[pid].mul_(0.001)
                    # update optimizer state locally (no communication)
                    for pid in local_pids:
                        local_opts[pid].add_(local_grads[pid].float().pow(2))
                    if rank == (row % world_size):
                        local_opts[row][:reduce_entities].add_(grad_row.float().pow(2))
                    if rank == owner_rank:
                        row_tensor[:reduce_entities].add_(grad_row, alpha=-args.row_lr)
                    for pid in local_pids:
                        local_tensors[pid].add_(local_grads[pid], alpha=-args.local_lr)
            else:
                grad_row.copy_(row_tensor[:reduce_entities])

            _sync(device)
            t2 = time.perf_counter()
            with _NvtxRange(args.nvtx, f"row/{row}/reduce"):
                dist.reduce(grad_row, dst=owner_rank, op=dist.ReduceOp.SUM)
            _sync(device)
            t3 = time.perf_counter()
            if rank == owner_rank:
                local_tensors[row][:reduce_entities].copy_(row_tensor[:reduce_entities])

            bcast_total += t1 - t0
            reduce_total += t3 - t2

        bcast_avg = bcast_total / args.iters
        reduce_avg = reduce_total / args.iters

        bcast_avg_all, bcast_max_all = _all_reduce_stats(bcast_avg, device)
        reduce_avg_all, reduce_max_all = _all_reduce_stats(reduce_avg, device)

        if rank == 0:
            bcast_bw = (bcast_bytes / bcast_max_all) / 1e9
            reduce_bw = (reduce_bytes / reduce_max_all) / 1e9
            results.append(
                {
                    "row": row,
                    "row_entities": row_entities,
                    "row_reduce_entities": reduce_entities,
                    "bcast_bytes": bcast_bytes,
                    "reduce_bytes": reduce_bytes,
                    "broadcast_avg_s": bcast_avg_all,
                    "broadcast_max_s": bcast_max_all,
                    "broadcast_bw_gbps": bcast_bw,
                    "reduce_avg_s": reduce_avg_all,
                    "reduce_max_s": reduce_max_all,
                    "reduce_bw_gbps": reduce_bw,
                }
            )
            print(
                "row {row}: bcast_size={size} ({bytes}) reduce_size={rsize} ({rbytes}) | "
                "bcast avg={bavg:.6f}s max={bmax:.6f}s ({bbw:.2f} GB/s) | "
                "reduce avg={ravg:.6f}s max={rmax:.6f}s ({rbw:.2f} GB/s)".format(
                    row=row,
                    size=row_entities,
                    bytes=_human_bytes(bcast_bytes),
                    rsize=reduce_entities,
                    rbytes=_human_bytes(reduce_bytes),
                    bavg=bcast_avg_all,
                    bmax=bcast_max_all,
                    bbw=bcast_bw,
                    ravg=reduce_avg_all,
                    rmax=reduce_max_all,
                    rbw=reduce_bw,
                )
            )

    if rank == 0 and args.output:
        out = Path(args.output)
        summary = None
        if results:
            total_bcast_bytes = sum(r["bcast_bytes"] for r in results)
            total_reduce_bytes = sum(r["reduce_bytes"] for r in results)
            total_bcast_time = sum(r["broadcast_max_s"] for r in results)
            total_reduce_time = sum(r["reduce_max_s"] for r in results)
            summary = {
                "rows_measured": len(results),
                "rows_total": num_parts,
                "bcast_bytes_total": total_bcast_bytes,
                "reduce_bytes_total": total_reduce_bytes,
                "bcast_time_total_s": total_bcast_time,
                "reduce_time_total_s": total_reduce_time,
                "comm_time_total_s": total_bcast_time + total_reduce_time,
                "bcast_time_per_row_s": total_bcast_time / len(results),
                "reduce_time_per_row_s": total_reduce_time / len(results),
            }
            if len(results) != num_parts:
                scale = float(num_parts) / float(len(results))
                summary["estimated_bcast_time_total_s"] = total_bcast_time * scale
                summary["estimated_reduce_time_total_s"] = total_reduce_time * scale
                summary["estimated_comm_time_total_s"] = (
                    total_bcast_time + total_reduce_time
                ) * scale
                summary["estimated_bcast_bytes_total"] = int(total_bcast_bytes * scale)
                summary["estimated_reduce_bytes_total"] = int(total_reduce_bytes * scale)
        payload = {
            "world_size": world_size,
            "logical_workers": num_parts,
            "backend": args.backend,
            "device": str(device),
            "embedding_dim": args.embedding_dim,
            "dtype": args.dtype,
            "sizes": sizes,
            "rows": rows,
            "iters": args.iters,
            "warmup": args.warmup,
            "results": results,
            "summary": summary,
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}")
        if summary is not None:
            print(
                "summary: rows={}/{} bcast_total={} reduce_total={} "
                "bcast_time={:.3f}s reduce_time={:.3f}s comm_time={:.3f}s".format(
                    summary["rows_measured"],
                    summary["rows_total"],
                    _human_bytes(summary["bcast_bytes_total"]),
                    _human_bytes(summary["reduce_bytes_total"]),
                    summary["bcast_time_total_s"],
                    summary["reduce_time_total_s"],
                    summary["comm_time_total_s"],
                )
            )
            if len(results) != num_parts and "estimated_comm_time_total_s" in summary:
                print(
                    "estimated_full_rows: bcast_time={:.3f}s reduce_time={:.3f}s comm_time={:.3f}s".format(
                        summary["estimated_bcast_time_total_s"],
                        summary["estimated_reduce_time_total_s"],
                        summary["estimated_comm_time_total_s"],
                    )
                )

    if rank == 0 and args.output_csv:
        out_csv = Path(args.output_csv)
        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "row",
                    "row_entities",
                    "row_reduce_entities",
                    "bcast_bytes",
                    "reduce_bytes",
                    "broadcast_avg_s",
                    "broadcast_max_s",
                    "broadcast_bw_gbps",
                    "reduce_avg_s",
                    "reduce_max_s",
                    "reduce_bw_gbps",
                ]
            )
            for r in results:
                writer.writerow(
                    [
                        r["row"],
                        r["row_entities"],
                        r["row_reduce_entities"],
                        r["bcast_bytes"],
                        r["reduce_bytes"],
                        r["broadcast_avg_s"],
                        r["broadcast_max_s"],
                        r["broadcast_bw_gbps"],
                        r["reduce_avg_s"],
                        r["reduce_max_s"],
                        r["reduce_bw_gbps"],
                    ]
                )
            if results:
                total_bcast_bytes = sum(r["bcast_bytes"] for r in results)
                total_reduce_bytes = sum(r["reduce_bytes"] for r in results)
                total_bcast_time = sum(r["broadcast_max_s"] for r in results)
                total_reduce_time = sum(r["reduce_max_s"] for r in results)
                writer.writerow(
                    [
                        "summary",
                        "",
                        "",
                        total_bcast_bytes,
                        total_reduce_bytes,
                        "",
                        total_bcast_time,
                        "",
                        "",
                        total_reduce_time,
                        "",
                    ]
                )
        print(f"wrote {out_csv}")

    dist.barrier()


def build_parser():
    parser = argparse.ArgumentParser(description="Row-wise partition comm benchmark")
    parser.add_argument("--backend", default="nccl", choices=["nccl", "gloo"], help="torch.distributed backend")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="device type")
    parser.add_argument("--device-id", type=int, default=None, help="explicit CUDA device id per rank")
    parser.add_argument("--local-rank", type=int, default=None, help="LOCAL_RANK override")
    parser.add_argument("--local_rank", dest="local_rank", type=int, default=None, help="LOCAL_RANK override (torch.distributed.launch)")

    parser.add_argument("--num-partitions", type=int, default=None, help="number of partitions (default: world size)")
    parser.add_argument("--logical-workers", type=int, default=8, help="logical workers/partitions (default: 8)")
    parser.add_argument("--num-entities", type=int, default=None, help="total entities if not using partition files")
    parser.add_argument(
        "--partition-npz",
        type=str,
        default="data/wikidata5m/partitions/stratification/num_8/entity_partition_entities.npz",
        help="path to entity_partition_entities.npz (default: wikidata5m stratification num_8)",
    )
    parser.add_argument(
        "--partition-dir",
        type=str,
        default=None,
        help="partition directory with entity_to_partitions.del(.pckl) as fallback",
    )
    parser.add_argument("--partition-sizes", type=str, default=None, help="comma-separated sizes per partition")

    parser.add_argument("--embedding-dim", type=int, default=128, help="embedding dimension")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"], help="tensor dtype")
    parser.add_argument("--scale", type=float, default=1.0, help="scale partition sizes (e.g., 0.1)")
    parser.add_argument("--max-entities", type=int, default=None, help="cap per-partition entities")
    parser.add_argument(
        "--override-partitions",
        action="store_true",
        help="allow custom partition sizes/scale and non-8 workers",
    )

    parser.add_argument("--rows", type=str, default="all", help="rows to benchmark (e.g., '0' or '0,1,2' or 'all')")
    parser.add_argument("--warmup", type=int, default=3, help="warmup iterations per row")
    parser.add_argument("--iters", type=int, default=10, help="timed iterations per row")
    parser.add_argument("--compute", action="store_true", help="simulate grad compute + local updates")
    parser.add_argument("--row-lr", type=float, default=0.1, help="row-owner update scale")
    parser.add_argument("--local-lr", type=float, default=0.1, help="local partition update scale")
    parser.add_argument("--row-unique-fraction", type=float, default=1.0, help="fraction of row entities with gradients")
    parser.add_argument("--row-unique-ids", type=int, default=None, help="explicit unique gradient count per row")
    parser.add_argument("--include-id-bytes", action="store_true", help="account for ID payload bytes in reduce stats")
    parser.add_argument("--id-bytes", type=int, default=8, help="bytes per ID when include-id-bytes is set")
    parser.add_argument("--nvtx", action="store_true", help="emit NVTX ranges for Nsight Systems")
    parser.add_argument("--output", type=str, default=None, help="write JSON results to path")
    parser.add_argument("--output-csv", type=str, default=None, help="write CSV results to path")
    return parser


def main():
    args = build_parser().parse_args()

    if args.backend == "nccl" and args.device == "cpu":
        raise ValueError("nccl backend requires --device cuda")

    run_benchmark(args)


if __name__ == "__main__":
    main()
