"""Generate belief heatmap snapshots for a single episode.

Runs the grid agent on one episode with `save_heatmaps_every=1`, so the
agent writes a PNG of the belief grid (with the agent and targets overlaid)
at every decision step.  Useful for producing figures in the final report.

Example:
    python dump_heatmap.py \
        --episodes episodes_test.json \
        --episode-id "FloorPlan201__Sofa__0" \
        --out-dir outputs/heatmaps/sofa_demo \
        --max-steps 80
"""

import argparse
import json
import os

from sim_env import AI2ThorObjNavEnv
from eval_objnav import Episode
from grid_agent import grid_bayes_agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--episode-id", required=True,
                    help="prefix match against episode_id")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--detector", default="metadata",
                    choices=["metadata", "yolo"])
    ap.add_argument("--spatial", default=None,
                    help="None | 'synthetic' | path/to/sensor_model.json")
    args = ap.parse_args()

    eps = json.load(open(args.episodes))
    matched = [Episode(**e) for e in eps if e["episode_id"].startswith(args.episode_id)]
    if not matched:
        raise SystemExit(
            f"no episode matched id prefix {args.episode_id!r}"
        )
    ep = matched[0]
    print(f"Running episode: {ep.episode_id}")
    print(f"  scene={ep.scene}  target={ep.target_type}  start={ep.start_position}")

    env = AI2ThorObjNavEnv(
        scene=ep.scene, width=320, height=240,
        grid_size=0.25, rotate_step_degrees=90,
        field_of_view=60, headless=True,
    )
    try:
        os.makedirs(args.out_dir, exist_ok=True)
        out = grid_bayes_agent(
            env, ep,
            max_steps=args.max_steps,
            save_dir=args.out_dir,
            detector=args.detector,
            spatial_sensor_model=args.spatial,
            save_heatmaps_every=1,
        )
        print(
            f"success={out['success']}  steps={out['num_steps']}  "
            f"final={out['final_position']}"
        )
        n = sum(1 for f in os.listdir(args.out_dir)
                if f.endswith(".png"))
        print(f"saved {n} heatmaps to {args.out_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
