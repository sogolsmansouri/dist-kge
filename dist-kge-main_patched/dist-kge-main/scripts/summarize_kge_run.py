#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import yaml


def _load_trace(trace_path: Path):
    entries = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for doc in yaml.safe_load_all(handle):
            if doc:
                entries.append(doc)
    return entries


def _best_metric(entries, key):
    values = [e.get(key) for e in entries if isinstance(e.get(key), (int, float))]
    return max(values) if values else None


def summarize(folder: Path):
    trace_path = folder / "trace.yaml"
    if not trace_path.exists():
        raise FileNotFoundError(f"trace.yaml not found in {folder}")

    entries = _load_trace(trace_path)
    train_epochs = [
        e
        for e in entries
        if e.get("event") == "epoch_completed"
        and e.get("job") == "train"
        and e.get("scope") == "epoch"
    ]
    epoch_times = [
        e.get("epoch_time")
        for e in train_epochs
        if isinstance(e.get("epoch_time"), (int, float))
    ]

    valid_entries = [
        e for e in entries if e.get("event") == "eval_completed" and e.get("split") == "valid"
    ]
    test_entries = [
        e for e in entries if e.get("event") == "eval_completed" and e.get("split") == "test"
    ]

    result = {
        "folder": str(folder),
        "epochs": len(epoch_times),
        "avg_epoch_time": (sum(epoch_times) / len(epoch_times)) if epoch_times else None,
        "min_epoch_time": min(epoch_times) if epoch_times else None,
        "max_epoch_time": max(epoch_times) if epoch_times else None,
        "best_mrr_filtered_valid": _best_metric(valid_entries, "mean_reciprocal_rank_filtered"),
        "best_mrr_filtered_with_test_valid": _best_metric(
            valid_entries, "mean_reciprocal_rank_filtered_with_test"
        ),
        "best_mrr_filtered_test": _best_metric(test_entries, "mean_reciprocal_rank_filtered"),
        "best_mrr_filtered_with_test_test": _best_metric(
            test_entries, "mean_reciprocal_rank_filtered_with_test"
        ),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Summarize a KGE run from trace.yaml")
    parser.add_argument("folder", help="Run folder that contains trace.yaml")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    result = summarize(Path(args.folder))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"folder: {result['folder']}")
        print(f"epochs: {result['epochs']}")
        print(f"avg_epoch_time_s: {result['avg_epoch_time']}")
        print(f"min_epoch_time_s: {result['min_epoch_time']}")
        print(f"max_epoch_time_s: {result['max_epoch_time']}")
        print(f"best_mrr_filtered_valid: {result['best_mrr_filtered_valid']}")
        print(f"best_mrr_filtered_with_test_valid: {result['best_mrr_filtered_with_test_valid']}")
        print(f"best_mrr_filtered_test: {result['best_mrr_filtered_test']}")
        print(
            "best_mrr_filtered_with_test_test:"
            f" {result['best_mrr_filtered_with_test_test']}"
        )


if __name__ == "__main__":
    main()
