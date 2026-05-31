"""Generate the headline ablation bar chart:

  Bar group 1 — Oracle detector:  Uniform sensor vs Spatial sensor
  Bar group 2 — YOLO detector:    Uniform sensor vs Spatial sensor

Shows the headline finding: spatial sensor helps YOLO (+16.7pp) but
slightly hurts oracle (-4.2pp), because spatial models the detector's
failure mode.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_sr(run_dir):
    p = os.path.join(run_dir, "summary.json")
    return json.load(open(p))["SR"] * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-uniform", required=True)
    ap.add_argument("--metadata-spatial", required=True)
    ap.add_argument("--yolo-uniform", required=True)
    ap.add_argument("--yolo-spatial", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    oracle_u = load_sr(args.metadata_uniform)
    oracle_s = load_sr(args.metadata_spatial)
    yolo_u = load_sr(args.yolo_uniform)
    yolo_s = load_sr(args.yolo_spatial)

    groups = ["Oracle detector", "YOLO-World detector"]
    uniform = [oracle_u, yolo_u]
    spatial = [oracle_s, yolo_s]

    x = np.arange(len(groups))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - w/2, uniform, w, label="uniform sensor",
                   color="#7fb3d5", edgecolor="black")
    bars2 = ax.bar(x + w/2, spatial, w, label="spatial sensor",
                   color="#e59866", edgecolor="black")
    for b, v in zip(bars1, uniform):
        ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.1f}%",
                ha="center", fontsize=11)
    for b, v in zip(bars2, spatial):
        ax.text(b.get_x() + b.get_width()/2, v + 0.6, f"{v:.1f}%",
                ha="center", fontsize=11)

    deltas = [oracle_s - oracle_u, yolo_s - yolo_u]
    for i, d in enumerate(deltas):
        sign = "+" if d >= 0 else ""
        color = "green" if d > 0 else "red"
        y = max(uniform[i], spatial[i]) + 5.5
        ax.annotate(f"Δ = {sign}{d:.1f}pp", xy=(x[i], y),
                    ha="center", color=color, fontsize=12, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_ylabel("Success Rate (%)", fontsize=12)
    ax.set_title("Spatial sensor model helps noisy detectors, "
                 "slightly hurts oracle", fontsize=13)
    ax.set_ylim(0, max(yolo_s, oracle_u, oracle_s, yolo_u) + 15)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
