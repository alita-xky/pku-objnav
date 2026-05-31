"""Three-bar ablation: uniform / synthetic-spatial / calibrated-spatial.

Compares the three sensor-model variants for both oracle and YOLO detectors.
For oracle there is no calibrated variant (oracle has nothing to calibrate),
so the third bar is omitted.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_sr(run_dir):
    return json.load(open(os.path.join(run_dir, "summary.json")))["SR"] * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-uniform", required=True)
    ap.add_argument("--metadata-spatial", required=True)
    ap.add_argument("--yolo-uniform", required=True)
    ap.add_argument("--yolo-spatial", required=True)
    ap.add_argument("--yolo-calibrated", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    oracle = [load_sr(args.metadata_uniform), load_sr(args.metadata_spatial), None]
    yolo = [load_sr(args.yolo_uniform), load_sr(args.yolo_spatial), load_sr(args.yolo_calibrated)]

    labels = ["uniform", "spatial (synthetic)", "spatial (calibrated)"]
    colors = ["#7fb3d5", "#e59866", "#a9dfbf"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, vals, title in zip(axes, [oracle, yolo], ["Oracle detector", "YOLO-World detector"]):
        x = np.arange(3)
        valid = [(i, v) for i, v in enumerate(vals) if v is not None]
        valid_x = [i for i, _ in valid]
        valid_v = [v for _, v in valid]
        bars = ax.bar(valid_x, valid_v, color=[colors[i] for i in valid_x],
                      edgecolor="black", width=0.6)
        for b, v in zip(bars, valid_v):
            ax.text(b.get_x() + b.get_width()/2, v + 0.7, f"{v:.1f}%",
                    ha="center", fontsize=12)
        if vals[2] is None:
            ax.text(2, 5, "(N/A:\noracle has\nno noise to\ncalibrate)",
                    ha="center", va="bottom", fontsize=10, color="grey", style="italic")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, 50)
    axes[0].set_ylabel("Success Rate (%)", fontsize=12)
    fig.suptitle("Spatial-sensor calibration does not help on this benchmark "
                 "(54 ep, 5 scenes × 4 targets × ~3 starts)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
