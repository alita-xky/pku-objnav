"""Collate per-agent eval results into one comparison table.

Usage:
    python collate_results.py \\
        --runs outputs/eval_random_test outputs/eval_nearest_test \\
               outputs/eval_grid_full outputs/eval_grid_spatial \\
               outputs/eval_yolo_full outputs/eval_yolo_spatial \\
        --out outputs/comparison.csv

Produces:
  * outputs/comparison.csv         — one row per agent (summary metrics)
  * outputs/comparison_per_ep.csv  — one row per (agent, episode) for cross-tab
  * stdout: a readable per-target SR breakdown
"""

import argparse
import csv
import json
import os
from collections import defaultdict


def load_run(run_dir):
    summary_p = os.path.join(run_dir, "summary.json")
    per_ep_p = os.path.join(run_dir, "per_episode.json")
    if not os.path.exists(summary_p) or not os.path.exists(per_ep_p):
        return None, None
    summary = json.load(open(summary_p))
    per_ep = json.load(open(per_ep_p))
    return summary, per_ep


def split_episode_id(eid):
    parts = eid.split("__")
    scene = parts[0] if len(parts) > 0 else "?"
    target = parts[1] if len(parts) > 1 else "?"
    start = parts[2] if len(parts) > 2 else "?"
    return scene, target, start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/comparison.csv")
    args = ap.parse_args()

    rows = []
    per_ep_rows = []
    target_sr = defaultdict(lambda: defaultdict(list))

    for run_dir in args.runs:
        summary, per_ep = load_run(run_dir)
        if summary is None:
            print(f"[skip] no results in {run_dir}")
            continue
        agent = summary.get("agent", os.path.basename(run_dir))
        rows.append({
            "run": os.path.basename(run_dir),
            "agent": agent,
            "n": summary["n"],
            "SR": round(summary["SR"], 4),
            "SPL": round(summary["SPL"], 4),
            "SoftSPL": round(summary["SoftSPL"], 4),
            "mean_steps": round(summary["mean_steps"], 1),
            "mean_path_length": round(summary["mean_path_length"], 2),
            "n_errors": summary["n_errors"],
        })

        for e in per_ep:
            scene, target, start = split_episode_id(e["episode_id"])
            per_ep_rows.append({
                "agent": agent,
                "episode_id": e["episode_id"],
                "scene": scene,
                "target": target,
                "start": start,
                "success": int(e["success"]),
                "num_steps": e["num_steps"],
                "spl": round(e["spl"], 4),
                "final_dist": round(e["final_distance_to_target"], 3),
            })
            target_sr[agent][target].append(int(e["success"]))

    if not rows:
        raise SystemExit("no runs to collate")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"summary -> {args.out}")

    per_ep_out = args.out.replace(".csv", "_per_ep.csv")
    if per_ep_out == args.out:
        per_ep_out = args.out + ".per_ep"
    with open(per_ep_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_ep_rows[0].keys()))
        w.writeheader()
        w.writerows(per_ep_rows)
    print(f"per-ep  -> {per_ep_out}")

    print("\n=== overall ===")
    hdr = f"{'agent':<32}{'n':>4}{'SR':>8}{'SPL':>8}{'SoftSPL':>10}{'steps':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['agent']:<32}{r['n']:>4}{r['SR']*100:>7.1f}%"
              f"{r['SPL']:>8.3f}{r['SoftSPL']:>10.3f}"
              f"{r['mean_steps']:>8.0f}")

    print("\n=== SR by target (rows: agent, cols: target) ===")
    targets = sorted({t for agent in target_sr for t in target_sr[agent]})
    hdr = f"{'agent':<32}" + "".join(f"{t:>16}" for t in targets)
    print(hdr)
    print("-" * len(hdr))
    for agent in sorted(target_sr):
        row = f"{agent:<32}"
        for t in targets:
            vals = target_sr[agent].get(t, [])
            if vals:
                sr = sum(vals) / len(vals)
                row += f"{sr*100:>14.1f}% "
            else:
                row += f"{'—':>16}"
        print(row)


if __name__ == "__main__":
    main()
