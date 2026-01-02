#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def load_snapshot(snapshot_path: Path):
    with snapshot_path.open("r") as fp:
        data = json.load(fp)
    partitions = data.get("partitions") or {}
    stats = []
    for pid_str, values in partitions.items():
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        stats.append(
            (
                pid,
                float(values.get("avg", 0.0)),
                float(values.get("sum", 0.0)),
                int(values.get("count", 0)),
            )
        )
    if not stats:
        raise ValueError("Snapshot does not contain partition statistics.")
    stats.sort(key=lambda item: (-item[1], -item[3], item[0]))
    return stats


def split_partition(assignments: np.ndarray, partition_id: int, next_pid: int):
    idx = np.where(assignments == partition_id)[0]
    if idx.size <= 1:
        return next_pid
    half = idx.size // 2
    if half == 0:
        return next_pid
    assignments[idx[half:]] = next_pid
    return next_pid + 1


def merge_partitions(assignments: np.ndarray, keep_id: int, drop_id: int):
    assignments[assignments == drop_id] = keep_id


def renumber(assignments: np.ndarray):
    unique_ids = np.unique(assignments)
    mapping = {old: new for new, old in enumerate(unique_ids)}
    for old, new in mapping.items():
        if old == new:
            continue
        assignments[assignments == old] = new
    return len(unique_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Repartition triples based on gradient snapshots."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset folder (e.g., data/freebase)",
    )
    parser.add_argument(
        "--partition_type",
        type=str,
        required=True,
        help="Partition type folder name (random, relation, graph-cut, ...).",
    )
    parser.add_argument(
        "--num_partitions",
        type=int,
        required=True,
        help="Number of partitions to load (matches existing num_X folder).",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Path to the gradient_snapshot_*.json file to use.",
    )
    parser.add_argument(
        "--hot_splits",
        type=int,
        default=1,
        help="How many top partitions to split (each split increases partition count).",
    )
    parser.add_argument(
        "--cold_merges",
        type=int,
        default=1,
        help="How many cold partition pairs to merge (each merge reduces partition count).",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="glow",
        help="Suffix appended to num_<n> when writing the new partition folder.",
    )
    args = parser.parse_args()

    part_dir = (
        args.dataset
        / "partitions"
        / args.partition_type
        / f"num_{args.num_partitions}"
    )
    input_file = part_dir / "train_assign_partitions.del"
    if not input_file.is_file():
        raise SystemExit(f"Cannot find {input_file}")

    assignments = np.loadtxt(input_file, dtype=np.int64)
    stats = load_snapshot(args.snapshot)
    if not stats:
        raise SystemExit("Snapshot contained no partitions.")

    next_pid = assignments.max() + 1
    hot_partitions = [pid for pid, *_ in stats[: args.hot_splits]]
    cold_partitions = [pid for pid, *_ in stats[::-1][: args.cold_merges * 2]]

    for pid in hot_partitions:
        next_pid = split_partition(assignments, pid, next_pid)

    for i in range(0, len(cold_partitions), 2):
        if i + 1 >= len(cold_partitions):
            break
        keep_id = cold_partitions[i]
        drop_id = cold_partitions[i + 1]
        if keep_id == drop_id:
            continue
        merge_partitions(assignments, keep_id, drop_id)

    new_num = renumber(assignments)
    output_dir = (
        args.dataset
        / "partitions"
        / args.partition_type
        / f"num_{new_num}_{args.output_suffix}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "train_assign_partitions.del"
    np.savetxt(output_file, assignments.astype(np.int64), fmt="%d")
    print(
        f"Wrote repartitioned assignments to {output_file} "
        f"(original {args.num_partitions} → new {new_num} partitions)."
    )


if __name__ == "__main__":
    main()
