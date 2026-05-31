"""ObjNav evaluation harness.

Computes the three standard ObjNav metrics on a fixed episode set:

    SR      Success rate              = frac(success)
    SPL     Success-weighted Path Len = mean_{episodes} S * d_geo / max(d_taken, d_geo)
    SoftSPL                           = mean_{episodes} (1 - clamp(d_final / d_geo, 0, 1)) * d_geo / max(d_taken, d_geo)

References:
    Anderson et al., "On Evaluation of Embodied Navigation Agents," arXiv 2018
    Habitat-Lab ObjectNav metrics.

Usage:
    # 1) build a fixed episode file (deterministic, reproducible)
    python eval_objnav.py make-episodes \
        --scenes FloorPlan201,FloorPlan202,FloorPlan203 \
        --targets "remote control,book,mug,sofa,television" \
        --per-target 4 --out episodes_small.json

    # 2) run an agent
    python eval_objnav.py run \
        --episodes episodes_small.json \
        --agent random \
        --out-dir outputs/eval_random

    # 3) compare runs
    python eval_objnav.py report --out-dirs outputs/eval_random outputs/eval_nearest
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Episode dataclass
# ============================================================

@dataclass
class Episode:
    episode_id: str
    scene: str
    target_type: str            # AI2-THOR object_type, e.g. "RemoteControl"
    target_prompt: str          # YOLO / NL form, e.g. "remote control"
    start_position: Dict[str, float]   # {"x", "y", "z"}
    start_rotation: float       # yaw degrees
    target_instances: List[Dict[str, float]]   # all target object positions
    geodesic_distance: float    # shortest reachable distance to nearest target
    success_distance: float = 1.0   # m, success threshold


@dataclass
class EpisodeResult:
    episode_id: str
    success: bool
    num_steps: int
    path_length: float
    geodesic_distance: float
    final_distance_to_target: float
    spl: float
    soft_spl: float
    wall_time_s: float
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Episode generation
# ============================================================

def position_distance_xz(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["z"] - b["z"]) ** 2)


def geodesic_distance_grid(
    start: Dict[str, float],
    targets: List[Dict[str, float]],
    reachable_positions: List[Dict[str, float]],
    grid_size: float = 0.25,
) -> float:
    """BFS on the grid graph of reachable positions.

    Returns shortest geodesic distance from start to any target position
    (via reachable cells). Uses 8-connected grid neighborhood at grid_size.
    """
    from collections import deque

    def quantize(p: Dict[str, float]) -> Tuple[int, int]:
        return (round(p["x"] / grid_size), round(p["z"] / grid_size))

    cells = {quantize(p) for p in reachable_positions}
    start_cell = quantize(start)
    target_cells = {quantize(t) for t in targets}

    # snap start to nearest reachable cell if not exactly on one
    if start_cell not in cells:
        start_cell = min(cells, key=lambda c: (c[0] - start_cell[0]) ** 2
                         + (c[1] - start_cell[1]) ** 2)

    if start_cell in target_cells:
        return 0.0

    visited = {start_cell: 0.0}
    q = deque([start_cell])
    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1),
              (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while q:
        c = q.popleft()
        cd = visited[c]
        for dx, dz in deltas:
            n = (c[0] + dx, c[1] + dz)
            if n in visited or n not in cells:
                continue
            step = grid_size * math.sqrt(dx * dx + dz * dz)
            visited[n] = cd + step
            if n in target_cells:
                return visited[n]
            q.append(n)

    # no path — use straight-line distance as fallback
    return min(position_distance_xz(start, t) for t in targets)


# ai2thor -> prompt mapping (mirrors run_yolo_bayes_approach.ai2thor_to_prompt)
_PROMPT_MAP = {
    "Television": "tv",
    "RemoteControl": "remote control",
    "TVStand": "tv stand",
    "DiningTable": "dining table",
    "CoffeeTable": "coffee table",
    "SideTable": "side table",
    "DeskLamp": "desk lamp",
    "FloorLamp": "floor lamp",
    "HousePlant": "house plant",
    "GarbageCan": "garbage can",
    "CreditCard": "credit card",
    "KeyChain": "key chain",
    "TissueBox": "tissue box",
    "LightSwitch": "light switch",
}


def _split_camel(name: str) -> str:
    import re
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()


def ai2thor_object_to_prompt(object_type: str) -> str:
    return _PROMPT_MAP.get(object_type, _split_camel(object_type))


def prompt_to_ai2thor_type(prompt: str) -> List[str]:
    """Reverse lookup. Returns possible AI2-THOR object_type names."""
    p = prompt.strip().lower()
    rev = {v: k for k, v in _PROMPT_MAP.items()}
    if p in rev:
        return [rev[p]]
    # match by re-camel
    cand = "".join(w.capitalize() for w in p.split())
    return [cand]


def generate_episodes(
    scenes: List[str],
    target_prompts: List[str],
    episodes_per_target: int,
    seed: int = 42,
    min_geodesic: float = 1.5,
    max_geodesic: float = 30.0,
    success_distance: float = 1.0,
    grid_size: float = 0.25,
) -> List[Episode]:
    """Build a deterministic episode set."""
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    rng = random.Random(seed)
    episodes: List[Episode] = []

    for scene in scenes:
        print(f"[gen] scene {scene}")
        ctrl = Controller(
            platform=CloudRendering,
            scene=scene,
            width=320, height=240,
            gridSize=grid_size,
            rotateStepDegrees=90,
            visibilityDistance=1.5,
            renderDepthImage=False,
        )
        try:
            ev = ctrl.step("GetReachablePositions")
            reachable = ev.metadata["actionReturn"] or []
            if not reachable:
                continue

            objs = ev.metadata["objects"]
            for tp in target_prompts:
                candidate_types = prompt_to_ai2thor_type(tp)
                # collect all instances of any candidate type with a position
                instances = [
                    {"x": o["position"]["x"], "y": o["position"]["y"],
                     "z": o["position"]["z"]}
                    for o in objs
                    if o.get("objectType") in candidate_types
                    and o.get("position") is not None
                ]
                if not instances:
                    continue

                # canonical ai2thor type for episode label
                canon_type = candidate_types[0]

                # sample start positions
                pool = list(reachable)
                rng.shuffle(pool)
                kept = 0
                for sp in pool:
                    if kept >= episodes_per_target:
                        break
                    geo = geodesic_distance_grid(
                        sp, instances, reachable, grid_size=grid_size,
                    )
                    if geo < min_geodesic or geo > max_geodesic:
                        continue
                    yaw = rng.choice([0, 90, 180, 270])
                    ep = Episode(
                        episode_id=(
                            f"{scene}__{canon_type}__{kept}__"
                            f"{int(sp['x']*100)}_{int(sp['z']*100)}_{yaw}"
                        ),
                        scene=scene,
                        target_type=canon_type,
                        target_prompt=tp,
                        start_position={
                            "x": sp["x"], "y": sp["y"], "z": sp["z"],
                        },
                        start_rotation=float(yaw),
                        target_instances=instances,
                        geodesic_distance=geo,
                        success_distance=success_distance,
                    )
                    episodes.append(ep)
                    kept += 1
        finally:
            ctrl.stop()

    return episodes


# ============================================================
# Metrics computation
# ============================================================

def trajectory_path_length(traj: List[Dict[str, float]]) -> float:
    if len(traj) < 2:
        return 0.0
    return sum(
        position_distance_xz(traj[i], traj[i + 1])
        for i in range(len(traj) - 1)
    )


def min_distance_to_targets(
    point: Dict[str, float],
    targets: List[Dict[str, float]],
) -> float:
    return min(position_distance_xz(point, t) for t in targets)


def compute_episode_metrics(
    episode: Episode,
    success: bool,
    trajectory: List[Dict[str, float]],
    num_steps: int,
    final_position: Dict[str, float],
    wall_time_s: float,
    extra: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> EpisodeResult:
    path_len = trajectory_path_length(trajectory) if trajectory else 0.0
    geo = max(episode.geodesic_distance, 1e-3)
    d_final = min_distance_to_targets(final_position, episode.target_instances)

    spl = (1.0 if success else 0.0) * geo / max(path_len, geo)

    progress = max(0.0, 1.0 - d_final / max(episode.geodesic_distance, 1e-3))
    progress = min(1.0, progress)
    soft_spl = progress * geo / max(path_len, geo)

    return EpisodeResult(
        episode_id=episode.episode_id,
        success=success,
        num_steps=num_steps,
        path_length=path_len,
        geodesic_distance=episode.geodesic_distance,
        final_distance_to_target=d_final,
        spl=spl,
        soft_spl=soft_spl,
        wall_time_s=wall_time_s,
        error=error,
        extra=extra or {},
    )


def aggregate_metrics(results: List[EpisodeResult]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"n": 0}
    sr = sum(1 for r in results if r.success) / n
    spl = sum(r.spl for r in results) / n
    soft_spl = sum(r.soft_spl for r in results) / n
    steps = sum(r.num_steps for r in results) / n
    path = sum(r.path_length for r in results) / n
    n_err = sum(1 for r in results if r.error)
    return {
        "n": n,
        "SR": round(sr, 4),
        "SPL": round(spl, 4),
        "SoftSPL": round(soft_spl, 4),
        "mean_steps": round(steps, 2),
        "mean_path_length": round(path, 3),
        "n_errors": n_err,
    }


# ============================================================
# Agent interface
# ============================================================

AgentFn = Callable[..., Dict[str, Any]]
# returns dict with keys:
#   success: bool
#   trajectory: List[{"x","y","z"}]
#   num_steps: int
#   final_position: {"x","y","z"}
#   extra: dict (optional)


def _agent_random(env, episode, max_steps=300, save_dir=None, **kw):
    """Random walk baseline."""
    actions = ["MoveAhead", "RotateLeft", "RotateRight"]
    rng = random.Random(int(time.time() * 1e6) % 2**32)
    obs = env.reset(scene=episode.scene)
    # teleport
    env.last_event = env.controller.step(
        "Teleport",
        position=episode.start_position,
        rotation={"x": 0, "y": episode.start_rotation, "z": 0},
        horizon=0,
    )
    obs = env._make_obs(env.last_event)
    traj = [dict(episode.start_position)]
    success = False
    for step in range(max_steps):
        a = rng.choice(actions)
        obs = env.step(a)
        p = obs["pose"]["position"]
        traj.append({"x": p["x"], "y": p["y"], "z": p["z"]})
        # success check: within threshold of any target AND target visible
        d_final = min_distance_to_targets(p, episode.target_instances)
        if d_final <= episode.success_distance:
            # check visibility via metadata
            visible = any(
                o.get("objectType") == episode.target_type and o.get("visible")
                for o in env.last_event.metadata["objects"]
            )
            if visible:
                success = True
                break
    final = {"x": traj[-1]["x"], "y": traj[-1]["y"], "z": traj[-1]["z"]}
    return {
        "success": success,
        "trajectory": traj,
        "num_steps": step + 1,
        "final_position": final,
    }


def _agent_nearest_metadata(env, episode, max_steps=300, save_dir=None, **kw):
    """Always head straight to the nearest known target via metadata."""
    from nav_utils import (
        plan_actions_to_position, position_to_node, normalize_yaw,
        yaw_to_face_position,
    )

    obs = env.reset(scene=episode.scene)
    env.last_event = env.controller.step(
        "Teleport",
        position=episode.start_position,
        rotation={"x": 0, "y": episode.start_rotation, "z": 0},
        horizon=0,
    )
    obs = env._make_obs(env.last_event)
    reachable = env.get_reachable_positions()

    # pick nearest target by straight-line distance
    cur = obs["pose"]["position"]
    target = min(
        episode.target_instances,
        key=lambda t: position_distance_xz(cur, t),
    )

    # pick a reachable point ~1m from target
    approach = min(
        reachable,
        key=lambda p: abs(position_distance_xz(p, target) - 0.85) + 0.01 *
        position_distance_xz(p, cur),
    )

    actions = plan_actions_to_position(
        reachable_positions=reachable,
        start_pose=obs["pose"],
        goal_position=approach,
        grid_size=env.grid_size,
    ) or []

    traj = [dict(cur)]
    success = False
    step = 0
    for a in actions[:max_steps]:
        obs = env.step(a)
        step += 1
        p = obs["pose"]["position"]
        traj.append({"x": p["x"], "y": p["y"], "z": p["z"]})

    # final orientation: rotate to face target
    if step < max_steps:
        cur = obs["pose"]["position"]
        desired_yaw = yaw_to_face_position(cur, target)
        cur_yaw = normalize_yaw(obs["pose"]["rotation"]["y"])
        from nav_utils import rotation_actions
        for a in rotation_actions(cur_yaw, desired_yaw):
            if step >= max_steps:
                break
            obs = env.step(a)
            step += 1
            p = obs["pose"]["position"]
            traj.append({"x": p["x"], "y": p["y"], "z": p["z"]})

    p = obs["pose"]["position"]
    d_final = min_distance_to_targets(p, episode.target_instances)
    visible = any(
        o.get("objectType") == episode.target_type and o.get("visible")
        for o in env.last_event.metadata["objects"]
    )
    success = d_final <= episode.success_distance and visible
    return {
        "success": success,
        "trajectory": traj,
        "num_steps": step,
        "final_position": {"x": p["x"], "y": p["y"], "z": p["z"]},
    }


AGENTS: Dict[str, AgentFn] = {
    "random": _agent_random,
    "nearest_metadata": _agent_nearest_metadata,
}


# ============================================================
# Runner
# ============================================================

class _EpisodeTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _EpisodeTimeout("episode wall-clock exceeded")


def run_evaluation(
    episodes_path: str,
    agent_name: str,
    out_dir: str,
    max_steps: int = 300,
    scene_width: int = 320,
    scene_height: int = 240,
    grid_size: float = 0.25,
    progress_every: int = 1,
    episode_timeout_s: int = 90,
) -> Dict[str, Any]:
    import signal

    from sim_env import AI2ThorObjNavEnv, ControllerDead

    os.makedirs(out_dir, exist_ok=True)

    signal.signal(signal.SIGALRM, _alarm_handler)

    with open(episodes_path) as f:
        ep_dicts = json.load(f)
    episodes = [Episode(**d) for d in ep_dicts]
    print(f"Loaded {len(episodes)} episodes from {episodes_path}")

    if agent_name not in AGENTS:
        raise SystemExit(
            f"unknown agent '{agent_name}'; pick one of {sorted(AGENTS)}"
        )
    agent = AGENTS[agent_name]

    results: List[EpisodeResult] = []

    # Group by scene to amortize controller startup cost
    by_scene: Dict[str, List[Episode]] = {}
    for ep in episodes:
        by_scene.setdefault(ep.scene, []).append(ep)

    for scene, scene_eps in by_scene.items():
        print(f"\n=== scene {scene} ({len(scene_eps)} episodes) ===")
        env = AI2ThorObjNavEnv(
            scene=scene,
            width=scene_width,
            height=scene_height,
            grid_size=grid_size,
            rotate_step_degrees=90,
            field_of_view=60,
            headless=True,
        )
        def _recreate(sleep_s: float = 2.0):
            """Tear down and rebuild env; tolerant of dead Unity subprocess."""
            nonlocal env
            try:
                env.close()
            except Exception:
                pass
            time.sleep(sleep_s)
            env = AI2ThorObjNavEnv(
                scene=scene, width=scene_width, height=scene_height,
                grid_size=grid_size, rotate_step_degrees=90,
                field_of_view=60, headless=True,
            )

        try:
            for i, ep in enumerate(scene_eps):
                # Pre-episode liveness check.  Catches the FP203 failure
                # mode where the Unity backend died during a previous
                # episode without raising, leaving every controller.step()
                # to return stale metadata in ~5 ms.  Retry once with a
                # longer cooldown if the first recreate also comes up dead.
                if not env.is_alive():
                    print(f"[health] controller dead before "
                          f"{ep.episode_id} — recreating (sleep 2s)")
                    _recreate(sleep_s=2.0)
                    if not env.is_alive():
                        print(f"[health] still dead after first recreate, "
                              f"retrying with sleep 5s")
                        _recreate(sleep_s=5.0)

                t0 = time.time()
                signal.alarm(int(episode_timeout_s))
                error = None
                try:
                    out = agent(env, ep, max_steps=max_steps,
                                save_dir=out_dir)
                except _EpisodeTimeout as e:
                    print(f"[timeout] episode {ep.episode_id} after "
                          f"{int(time.time() - t0)}s — recreating controller")
                    _recreate(sleep_s=2.0)
                    out = {
                        "success": False,
                        "trajectory": [dict(ep.start_position)],
                        "num_steps": 0,
                        "final_position": dict(ep.start_position),
                    }
                    error = "wall-clock timeout"
                except ControllerDead as e:
                    print(f"[health] {e} during {ep.episode_id} — "
                          f"recreating controller")
                    _recreate(sleep_s=2.0)
                    out = {
                        "success": False,
                        "trajectory": [dict(ep.start_position)],
                        "num_steps": 0,
                        "final_position": dict(ep.start_position),
                    }
                    error = "controller dead (in-step liveness check)"
                except TimeoutError as e:
                    # ai2thor.controller.Controller.step raises TimeoutError
                    # when the Unity backend stops responding to a single
                    # action — observed on FP203 (sequenceId=5 RotateLeft).
                    # Treat it as a dead controller and recreate so the
                    # next episode isn't poisoned.
                    print(f"[health] AI2-THOR TimeoutError during "
                          f"{ep.episode_id}: {e} — recreating controller")
                    _recreate(sleep_s=5.0)
                    out = {
                        "success": False,
                        "trajectory": [dict(ep.start_position)],
                        "num_steps": 0,
                        "final_position": dict(ep.start_position),
                    }
                    error = "ai2thor TimeoutError"
                except Exception as e:
                    print(f"[err] episode {ep.episode_id}: {e}")
                    traceback.print_exc()
                    out = {
                        "success": False,
                        "trajectory": [dict(ep.start_position)],
                        "num_steps": 0,
                        "final_position": dict(ep.start_position),
                    }
                    error = repr(e)
                finally:
                    signal.alarm(0)
                dt = time.time() - t0

                metrics = compute_episode_metrics(
                    episode=ep,
                    success=out["success"],
                    trajectory=out["trajectory"],
                    num_steps=out["num_steps"],
                    final_position=out["final_position"],
                    wall_time_s=dt,
                    extra=out.get("extra", {}),
                    error=error,
                )
                results.append(metrics)

                if (i + 1) % progress_every == 0:
                    print(
                        f"  [{i + 1}/{len(scene_eps)}] "
                        f"id={ep.episode_id[:60]} "
                        f"success={metrics.success} "
                        f"steps={metrics.num_steps} "
                        f"d_final={metrics.final_distance_to_target:.2f}m "
                        f"SPL={metrics.spl:.3f} "
                        f"({dt:.1f}s)"
                    )
        finally:
            env.close()

    # save
    rows = [asdict(r) for r in results]
    with open(os.path.join(out_dir, "per_episode.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(out_dir, "per_episode.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "episode_id", "success", "num_steps", "path_length",
            "geodesic_distance", "final_distance_to_target",
            "spl", "soft_spl", "wall_time_s", "error",
        ])
        w.writeheader()
        for r in rows:
            row = {k: r[k] for k in w.fieldnames}
            w.writerow(row)

    summary = aggregate_metrics(results)
    summary["agent"] = agent_name
    summary["episodes_path"] = episodes_path
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("===== SUMMARY =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    return summary


# ============================================================
# CLI
# ============================================================

def _make_episodes_cmd(args):
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    eps = generate_episodes(
        scenes=scenes,
        target_prompts=targets,
        episodes_per_target=args.per_target,
        seed=args.seed,
        min_geodesic=args.min_geo,
        max_geodesic=args.max_geo,
        success_distance=args.success_distance,
    )
    rows = [asdict(e) for e in eps]
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"saved {len(rows)} episodes to {args.out}")


def _run_cmd(args):
    summary = run_evaluation(
        episodes_path=args.episodes,
        agent_name=args.agent,
        out_dir=args.out_dir,
        max_steps=args.max_steps,
        episode_timeout_s=args.timeout,
    )
    print(f"\nsaved to {args.out_dir}/")


def _report_cmd(args):
    rows = []
    for d in args.out_dirs:
        p = os.path.join(d, "summary.json")
        if not os.path.exists(p):
            print(f"missing summary in {d}")
            continue
        with open(p) as f:
            rows.append(json.load(f))
    if not rows:
        return
    keys = ["agent", "n", "SR", "SPL", "SoftSPL",
            "mean_steps", "mean_path_length", "n_errors"]
    widths = [max(len(k), max(len(str(r.get(k, ""))) for r in rows))
              for k in keys]
    print(" | ".join(k.ljust(w) for k, w in zip(keys, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(r.get(k, "")).ljust(w)
                          for k, w in zip(keys, widths)))


def _register_optional_agents():
    """Pull in agent modules that live outside this file."""
    try:
        import grid_agent  # type: ignore
        AGENTS["grid_bayes_metadata"] = grid_agent.grid_bayes_metadata_agent
        AGENTS["grid_bayes_metadata_spatial"] = (
            grid_agent.grid_bayes_metadata_spatial_agent
        )
        AGENTS["grid_bayes_metadata_greedy"] = (
            grid_agent.grid_bayes_metadata_greedy_agent
        )
        AGENTS["grid_bayes_metadata_heuristic"] = (
            grid_agent.grid_bayes_metadata_heuristic_agent
        )
        AGENTS["grid_bayes_yolo"] = grid_agent.grid_bayes_yolo_agent
        AGENTS["grid_bayes_yolo_spatial"] = (
            grid_agent.grid_bayes_yolo_spatial_agent
        )
        AGENTS["grid_bayes_yolo_calibrated"] = (
            grid_agent.grid_bayes_yolo_calibrated_agent
        )
    except Exception as e:  # pragma: no cover
        print(f"[eval_objnav] grid_agent not registered: {e!r}")


def main():
    _register_optional_agents()

    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    p_make = sub.add_parser("make-episodes")
    p_make.add_argument("--scenes", required=True)
    p_make.add_argument("--targets", required=True)
    p_make.add_argument("--per-target", type=int, default=4)
    p_make.add_argument("--seed", type=int, default=42)
    p_make.add_argument("--min-geo", type=float, default=1.5)
    p_make.add_argument("--max-geo", type=float, default=30.0)
    p_make.add_argument("--success-distance", type=float, default=1.0)
    p_make.add_argument("--out", required=True)
    p_make.set_defaults(func=_make_episodes_cmd)

    p_run = sub.add_parser("run")
    p_run.add_argument("--episodes", required=True)
    p_run.add_argument("--agent", required=True,
                       choices=sorted(AGENTS))
    p_run.add_argument("--out-dir", required=True)
    p_run.add_argument("--max-steps", type=int, default=300)
    p_run.add_argument("--timeout", type=int, default=90,
                       help="per-episode wall-clock timeout (seconds)")
    p_run.set_defaults(func=_run_cmd)

    p_rep = sub.add_parser("report")
    p_rep.add_argument("--out-dirs", nargs="+", required=True)
    p_rep.set_defaults(func=_report_cmd)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
