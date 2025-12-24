#!/usr/bin/env python3
# Utility to derive Glow window proposals from a gradient snapshot and affinity threshold.
# snapshot JSON format: exported by work_scheduler gradient snapshot (partitions -> sum/count/avg)

import argparse
import json
from collections import defaultdict


def load_snapshot(path):
    with open(path, "r") as fp:
        data = json.load(fp)
    partitions = data.get("partitions", {})
    stats = {}
    for key, value in partitions.items():
        try:
            pid = int(key)
        except ValueError:
            continue
        stats[int(pid)] = {
            "sum": float(value.get("sum", 0.0)),
            "count": int(value.get("count", 0)),
            "avg": float(value.get("avg", 0.0)),
        }
    return stats


def compute_affinity(stats, alpha, threshold):
    scores = defaultdict(float)
    counts = defaultdict(int)
    pids = sorted(stats.keys())
    for idx, pid in enumerate(pids):
        avg_pi = abs(stats[pid]["avg"])
        for pj in pids[idx + 1 :]:
            avg_pj = abs(stats[pj]["avg"])
            denom = avg_pi + avg_pj
            if denom <= 0:
                affinity = 0.0
            else:
                affinity = 1.0 - abs(avg_pi - avg_pj) / denom
            pair = (pid, pj)
            prev = scores[pair]
            scores[pair] = (1.0 - alpha) * prev + alpha * affinity
            counts[pair] += 1
    filtered = {pair: score for pair, score in scores.items() if score >= threshold}
    return scores, counts, filtered


def dump_top_pairs(scores, top_k=20):
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    for pair, score in items[:top_k]:
        print(f"pair {pair}: affinity={score:.4f}")


def propose_windows(stats, affinity_scores, window_size=3):
    remaining = set(stats.keys())
    proposals = []
    while remaining:
        seed = remaining.pop()
        window = [seed]
        while len(window) < window_size and remaining:
            best = None
            best_score = float("-inf")
            for candidate in remaining:
                total = 0.0
                for member in window:
                    key = tuple(sorted((candidate, member)))
                    total += affinity_scores.get(key, 0.0)
                avg_score = total / len(window)
                if avg_score > best_score:
                    best_score = avg_score
                    best = candidate
            if best is None:
                break
            window.append(best)
            remaining.remove(best)
        proposals.append(window)
    return proposals


def main():
    parser = argparse.ArgumentParser(
        description="Compute Glow partition affinities from gradient snapshots."
    )
    parser.add_argument("--snapshot", required=True, help="path to snapshot JSON")
    parser.add_argument("--alpha", type=float, default=0.3, help="EMA weight")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="minimum affinity to report"
    )
    parser.add_argument("--window_size", type=int, default=3, help="proposed window size")
    parser.add_argument("--top_k", type=int, default=20, help="top pairs to print")
    args = parser.parse_args()

    stats = load_snapshot(args.snapshot)
    if not stats:
        raise SystemExit("No partitions found in snapshot")
    scores, counts, filtered = compute_affinity(stats, args.alpha, args.threshold)
    print(f"Computed {len(scores)} pair affinities (EMA alpha={args.alpha}).")
    print(f"{len(filtered)} pairs exceed threshold {args.threshold}.")
    dump_top_pairs(scores, top_k=args.top_k)

    proposals = propose_windows(stats, scores, window_size=args.window_size)
    print(f"Proposed {len(proposals)} windows (size≈{args.window_size}):")
    for win in proposals[: args.top_k]:
        print("  window:", win)


if __name__ == "__main__":
    main()
