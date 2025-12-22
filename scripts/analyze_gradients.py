#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


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
    stats.sort(key=lambda item: (-item[1], -item[3], item[0]))
    return {
        "num_partitions": data.get("num_partitions"),
        "snapshot_index": data.get("snapshot_index"),
        "final": data.get("final", False),
        "ordered_partitions": [pid for pid, _, _, _ in stats],
        "stats": [
            {
                "partition": pid,
                "avg": avg,
                "sum": total,
                "count": count,
            }
            for pid, avg, total, count in stats
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze gradient snapshots and emit a partition ordering."
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="Path to the experiment folder containing gradient_snapshots/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the ordering JSON (default: experiment/gradient_order.json)",
    )
    args = parser.parse_args()

    snapshot_dir = args.experiment / "gradient_snapshots"
    if not snapshot_dir.is_dir():
        raise SystemExit(f"No gradient_snapshots directory in {args.experiment}")

    snapshots = sorted(snapshot_dir.glob("gradient_snapshot_*.json"))
    if not snapshots:
        final_path = snapshot_dir / "gradient_snapshot_final.json"
        if final_path.is_file():
            snapshots = [final_path]
    if not snapshots:
        raise SystemExit(f"No gradient snapshots found under {snapshot_dir}")

    latest = snapshots[-1]
    ordering = load_snapshot(latest)

    output_path = args.output or (args.experiment / "gradient_order.json")
    with output_path.open("w") as fp:
        json.dump(ordering, fp, indent=2)
    print(f"Wrote gradient ordering to {output_path}")


if __name__ == "__main__":
    main()
