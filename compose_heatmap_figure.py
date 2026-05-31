"""Compose a 3-panel side-by-side belief-evolution figure for the paper.

Reads three timestamped heatmap PNGs from a heatmap dump dir and stacks
them horizontally with subtitles.

Usage:
    python compose_heatmap_figure.py \\
        --dir outputs/heatmap_sofa_demo \\
        --out outputs/figure_belief_evolution.png

If the directory has more than 3 PNGs, picks first, middle, last by name.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--captions", default="initial,mid,final")
    args = ap.parse_args()

    pngs = sorted(f for f in os.listdir(args.dir) if f.endswith(".png"))
    if not pngs:
        print(f"no PNGs in {args.dir}", file=sys.stderr)
        sys.exit(1)

    if len(pngs) >= 3:
        picks = [pngs[0], pngs[len(pngs) // 2], pngs[-1]]
    elif len(pngs) == 2:
        picks = [pngs[0], pngs[-1], pngs[-1]]
    else:
        picks = [pngs[0]] * 3

    captions = [c.strip() for c in args.captions.split(",")]
    while len(captions) < 3:
        captions.append("")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, p, cap in zip(axes, picks, captions):
        img = imread(os.path.join(args.dir, p))
        ax.imshow(img)
        # extract step number from filename if present
        base = os.path.splitext(p)[0]
        step_tag = base.split("_step")[-1] if "_step" in base else ""
        title = f"{cap} (step {int(step_tag)})" if step_tag.isdigit() else cap
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"saved {args.out} using {picks}")


if __name__ == "__main__":
    main()
