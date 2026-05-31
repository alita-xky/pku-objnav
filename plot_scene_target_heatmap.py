"""Plot a (scenes × targets) SR heatmap for a single agent run.

For the paper: shows where the agent succeeds / fails across the 5 scenes
and 4 target types. Highlights that small objects (book, RC) systematically
fail across all scenes (visibility-threshold issue), while big objects
(sofa, TV) succeed in most scenes.
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="path to a run dir with per_episode.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    eps = json.load(open(os.path.join(args.run, "per_episode.json")))
    if not eps:
        raise SystemExit("empty per_episode")

    # Group by (scene, target)
    by_st = defaultdict(list)
    for e in eps:
        parts = e["episode_id"].split("__")
        scene = parts[0] if len(parts) > 0 else "?"
        target = parts[1] if len(parts) > 1 else "?"
        by_st[(scene, target)].append(int(e["success"]))

    scenes = sorted({s for s, _ in by_st})
    targets = sorted({t for _, t in by_st})

    grid = np.full((len(scenes), len(targets)), np.nan)
    counts = np.zeros_like(grid)
    for i, sc in enumerate(scenes):
        for j, tg in enumerate(targets):
            vals = by_st.get((sc, tg), [])
            if vals:
                grid[i, j] = sum(vals) / len(vals) * 100
                counts[i, j] = len(vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = matplotlib.colormaps.get_cmap("RdYlGn").copy()
    cmap.set_bad("#dddddd")
    masked = np.ma.masked_invalid(grid)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, fontsize=11)
    ax.set_yticks(range(len(scenes)))
    ax.set_yticklabels(scenes, fontsize=11)
    for i in range(len(scenes)):
        for j in range(len(targets)):
            v = grid[i, j]
            if np.isnan(v):
                txt = "—"
            else:
                txt = f"{v:.0f}%\n(n={int(counts[i,j])})"
            color = "white" if (not np.isnan(v) and v < 35) else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=10, color=color)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Success Rate (%)", fontsize=11)
    title = args.title or os.path.basename(args.run.rstrip("/"))
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
