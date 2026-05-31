"""ObjNav agent using BeliefGrid + information-gain-driven viewpoint selection.

This is the implementation of the "creative" pieces described in
项目方案与代码分析.md  (directions ① + ③):

  - Full 2D probability grid maintained over the scene floor.
  - Bayes update with explicit sensor model (per-cell TP / FP)
    (per-class spatial sensor model can be plugged in later via
    BeliefGrid.update_negative_observation's `spatial_sensor_model` arg).
  - Shannon entropy reduction (true info gain) used to pick the next
    viewpoint, traded off against travel cost.

Two detector modes are supported:

  - "metadata": uses AI2-THOR metadata (object_type + visible + position)
    as an oracle.  Useful for fast debugging — guarantees that any
    pipeline failure is not a perception failure.
  - "yolo": uses YoloWorldDetector (real visual model).  Slower but
    realistic; what the final report should use.

The agent function is registered into eval_objnav.AGENTS so it can be run
through the same evaluation harness:

    python eval_objnav.py run \\
        --episodes episodes_test.json --agent grid_bayes_metadata \\
        --out-dir outputs/eval_grid_metadata --max-steps 200
"""

from __future__ import annotations

import math
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from bayes_grid import BeliefGrid, BeliefGridConfig
from nav_utils import (
    plan_actions_to_position,
    position_to_node,
    normalize_yaw,
    yaw_to_face_position,
    rotation_actions,
)
from prior_backend import make_prior, norm_label, label_match


# ============================================================
# Detector adapters (metadata oracle / YOLO)
# ============================================================

def metadata_detect(env, target_type: str, context_types: List[str]):
    """Read visible objects from AI2-THOR metadata.

    Returns (target_detections, context_detections).  Each detection:
        {"label": str, "score": float, "position": {x,y,z}, "distance": float}
    """
    target_dets = []
    context_dets = []
    for obj in env.last_event.metadata["objects"]:
        if not obj.get("visible"):
            continue
        ot = obj.get("objectType")
        pos = obj.get("position")
        if pos is None:
            continue
        det = {
            "label": ot,
            "score": 1.0,
            "position": {"x": pos["x"], "y": pos["y"], "z": pos["z"]},
            "distance": obj.get("distance", -1.0),
        }
        if ot == target_type:
            target_dets.append(det)
        else:
            for ctype in context_types:
                if label_match(ot, ctype):
                    context_dets.append(det)
                    break
    return target_dets, context_dets


def _box_to_world_position(box, depth, pose, image_width=320,
                           image_height=240, field_of_view=60):
    """Project a YOLO bbox center into world coordinates via depth.

    Returns {"x", "y", "z", "depth"} or None if depth is invalid.
    """
    if depth is None:
        return None
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(image_width - 1, x1)))
    x2 = int(max(0, min(image_width - 1, x2)))
    y1 = int(max(0, min(image_height - 1, y1)))
    y2 = int(max(0, min(image_height - 1, y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    # take median depth over inner 60% of the box (robust to edge bleed)
    bw, bh = x2 - x1, y2 - y1
    px1 = int(x1 + 0.2 * bw)
    px2 = int(x2 - 0.2 * bw)
    py1 = int(y1 + 0.2 * bh)
    py2 = int(y2 - 0.2 * bh)
    if px2 <= px1 or py2 <= py1:
        return None
    patch = depth[py1:py2, px1:px2]
    valid = patch[np.isfinite(patch)]
    valid = valid[valid > 0.05]
    if len(valid) == 0:
        return None
    z_cam = float(np.median(valid))

    cx_box = (x1 + x2) / 2.0
    fov_rad = math.radians(field_of_view)
    fy = image_height / (2.0 * math.tan(fov_rad / 2.0))
    fx = fy * image_width / image_height
    cx = image_width / 2.0
    x_cam = (cx_box - cx) * z_cam / fx

    ap = pose["position"]
    yaw = math.radians(pose["rotation"]["y"])
    world_x = ap["x"] + math.cos(yaw) * x_cam + math.sin(yaw) * z_cam
    world_z = ap["z"] - math.sin(yaw) * x_cam + math.cos(yaw) * z_cam
    return {"x": world_x, "y": ap["y"], "z": world_z, "depth": z_cam}


def yolo_detect(env, obs, detector, target_prompt, context_types,
                image_width=320, image_height=240, field_of_view=60):
    """Run YOLO and back-project boxes to world positions via depth.

    Splits results into target_dets and context_dets based on label match.
    """
    detections = detector.detect(obs["rgb"])
    target_dets = []
    context_dets = []
    for d in detections:
        pos = _box_to_world_position(
            d["box"], obs.get("depth"), obs["pose"],
            image_width=image_width, image_height=image_height,
            field_of_view=field_of_view,
        )
        if pos is None:
            continue
        item = {"label": d["label"], "score": d["score"],
                "position": pos, "distance": pos["depth"]}
        if label_match(d["label"], target_prompt):
            target_dets.append(item)
            continue
        for ctype in context_types:
            if label_match(d["label"], ctype):
                context_dets.append(item)
                break
    return target_dets, context_dets


# ============================================================
# Candidate viewpoint generator
# ============================================================

def generate_view_candidates(
    reachable_positions: List[Dict[str, float]],
    yaws: Tuple[int, ...] = (0, 90, 180, 270),
    max_candidates: int = 100,
    seed: int = 0,
) -> List[Dict]:
    """Build a pool of (position, yaw) candidates.

    For small scenes we use every reachable position × every yaw.
    For larger scenes we down-sample uniformly to keep IG evaluation cheap.
    """
    n = len(reachable_positions)
    n_each = max(1, max_candidates // len(yaws))
    if n > n_each:
        rng = random.Random(seed)
        idxs = rng.sample(range(n), n_each)
        positions = [reachable_positions[i] for i in idxs]
    else:
        positions = reachable_positions

    candidates = []
    for p in positions:
        for yaw in yaws:
            candidates.append({
                "position": {"x": p["x"], "y": p["y"], "z": p["z"]},
                "rotation": {"y": float(yaw)},
            })
    return candidates


# ============================================================
# Agent
# ============================================================

def grid_bayes_agent(
    env,
    episode,
    max_steps: int = 300,
    save_dir: Optional[str] = None,
    detector: str = "metadata",
    prior_backend: str = "pmi",
    fov_deg: float = 60.0,
    visibility_distance: float = 1.5,
    tp_rate: float = 0.85,
    fp_rate: float = 0.05,
    ig_lambda: float = 0.05,    # trade-off: IG - lambda * travel_cost
    save_heatmaps_every: int = 0,  # 0 disables; otherwise every N decisions
    spatial_sensor_model: Optional[str] = None,   # None | "synthetic" | path/to/sensor_model.json
    ig_strategy: str = "shannon",     # "shannon" | "heuristic" | "greedy"
    episode_timeout_s: float = 60.0,  # abort episode if wall-clock exceeds this
    **_kw,
) -> Dict[str, Any]:
    """Run one episode with the BeliefGrid-based agent.

    Returns dict compatible with eval_objnav.AgentFn:
        {success, trajectory, num_steps, final_position, extra}
    """
    # ---- reset env and teleport to start --------------------------------
    obs = env.reset(scene=episode.scene)
    env.last_event = env.controller.step(
        "Teleport",
        position=episode.start_position,
        rotation={"x": 0, "y": episode.start_rotation, "z": 0},
        horizon=0,
    )
    obs = env._make_obs(env.last_event)
    reachable = env.get_reachable_positions()

    # ---- initialize grid & prior ----------------------------------------
    grid = BeliefGrid(reachable, BeliefGridConfig(resolution=env.grid_size))
    prior = make_prior(prior_backend)
    target_prompt = episode.target_prompt
    target_type = episode.target_type
    context_types = prior.get_context_prompts(target_prompt)

    # cache PMI weights for fast lookup during update
    context_weights = {
        ct: prior.get_context_weight(target_prompt, ct) for ct in context_types
    }

    # build candidate views
    candidates = generate_view_candidates(reachable)

    # optional spatial sensor model
    _sensor_callable = None
    if spatial_sensor_model is not None:
        from sensor_model import SensorModel
        if spatial_sensor_model == "synthetic":
            _sensor_callable = SensorModel.synthetic_spatial_model()
        else:
            sm = SensorModel.load(spatial_sensor_model)
            _sensor_callable = sm.as_spatial_callable(target_prompt)

    # ---- detector dispatch ----------------------------------------------
    if detector == "metadata":
        def _detect(_obs):
            return metadata_detect(env, target_type, context_types)
    elif detector == "yolo":
        # imported lazily so metadata mode does not pay torch startup cost
        from yolo_detector import YoloWorldDetector
        yolo_classes = sorted({target_prompt, *context_types})
        # absolute path so Ultralytics does NOT attempt to fetch from network
        import os as _os
        _model_path = _os.path.abspath(
            _kw.get("yolo_model_path", "yolov8s.pt")
        )
        _open_vocab = _kw.get("yolo_open_vocab", False)  # default closed-vocab
        det = YoloWorldDetector(
            classes=yolo_classes,
            model_name=_model_path,
            conf=0.12,
            device=_kw.get("yolo_device", "cuda:0"),
            open_vocab=_open_vocab,
        )

        def _detect(_obs):
            return yolo_detect(
                env, _obs, det, target_prompt, context_types,
                image_width=env.width, image_height=env.height,
                field_of_view=env.field_of_view,
            )
    else:
        raise ValueError(f"unknown detector: {detector!r}")

    # ---- main loop -------------------------------------------------------
    traj: List[Dict[str, float]] = [dict(episode.start_position)]
    step = 0
    success = False
    decisions = 0
    seen_contexts: Dict[Tuple[str, int, int], Dict] = {}
    # watchdog: if many decisions pass without any move, abort
    decisions_without_move = 0
    last_position_node = position_to_node(obs["pose"]["position"], env.grid_size)
    t_episode_start = time.time()
    # FP203-style watchdog: count consecutive failed approaches to the
    # *same* target instance.  Triggered when the target is metadata-visible
    # from many viewpoints but no reachable cell is within `success_distance`
    # — without this the agent re-runs approach forever and never enters
    # the IG branch.
    target_dets_failures = 0
    last_tpos_key: Optional[Tuple[float, float]] = None
    target_dets_giveup_threshold = 3
    # Algorithmic-deadlock watchdog for the IG branch: if plan_actions_to_position
    # returns empty for a goal we're not standing on, that goal is unreachable
    # from the current pose under the current path planner — without this
    # the IG picker keeps re-selecting the same unreachable cell every
    # iteration (FP203 Sofa__1: 113 decisions all pointing at (-4.0, 5.5)
    # while the agent walked 1.5m total).
    unreachable_goals: set = set()
    extra = {
        "entropies": [],
        "decisions": [],
        "timed_out": False,
        "target_dets_giveups": 0,
        "ig_blacklisted_goals": 0,
    }

    while step < max_steps:
        if time.time() - t_episode_start > episode_timeout_s:
            extra["timed_out"] = True
            break
        # 1. observe & update belief
        target_dets, context_dets = _detect(obs)
        for cd in context_dets:
            ctype = next(
                (c for c in context_types if label_match(cd["label"], c)),
                None,
            )
            if ctype is None:
                continue
            key = (ctype, int(round(cd["position"]["x"] / 0.5)),
                   int(round(cd["position"]["z"] / 0.5)))
            if key in seen_contexts:
                continue
            seen_contexts[key] = cd
            grid.update_context_detection(
                context_position=cd["position"],
                prior_weight=context_weights.get(ctype, 0.0),
                detector_confidence=cd["score"],
                spread_sigma=1.2,
                bump_radius=3.0,
            )

        grid.update_negative_observation(
            agent_pose=obs["pose"],
            fov_deg=fov_deg,
            max_distance=visibility_distance,
            tp_rate=tp_rate,
            fp_rate=fp_rate,
            spatial_sensor_model=_sensor_callable,
        )

        # 2. target seen?  approach directly via metadata position.
        # First decide whether to *honor* the detection this round: if the
        # same target instance has already produced N failed approaches in
        # a row (FP203 sofa case — visible but no reachable cell ≤1m),
        # skip the approach this iteration and fall through to IG so a
        # different viewpoint family gets tried.
        ignore_target_dets = False
        if target_dets:
            tpos_now = target_dets[0]["position"]
            tpos_key_now = (round(tpos_now["x"], 1), round(tpos_now["z"], 1))
            if tpos_key_now == last_tpos_key:
                if target_dets_failures >= target_dets_giveup_threshold:
                    ignore_target_dets = True
                    extra["target_dets_giveups"] += 1
            else:
                target_dets_failures = 0
                last_tpos_key = tpos_key_now

        if target_dets and not ignore_target_dets:
            grid.update_positive_observation(
                target_position=target_dets[0]["position"],
                tp_rate=tp_rate,
                fp_rate=fp_rate,
                spread_sigma=0.6,
                bump_radius=1.0,
            )
            tpos = target_dets[0]["position"]
            approach = _pick_approach_point(reachable, tpos)
            actions = plan_actions_to_position(
                reachable_positions=reachable,
                start_pose=obs["pose"],
                goal_position=approach,
                grid_size=env.grid_size,
            ) or []
            for a in actions:
                if step >= max_steps:
                    break
                obs = env.step(a)
                step += 1
                p = obs["pose"]["position"]
                traj.append({"x": p["x"], "y": p["y"], "z": p["z"]})
                if not obs["last_action_success"]:
                    break
            # face the target
            cur_pos = obs["pose"]["position"]
            cur_yaw = normalize_yaw(obs["pose"]["rotation"]["y"])
            for a in rotation_actions(cur_yaw, yaw_to_face_position(cur_pos, tpos)):
                if step >= max_steps:
                    break
                obs = env.step(a)
                step += 1
                p = obs["pose"]["position"]
                traj.append({"x": p["x"], "y": p["y"], "z": p["z"]})
            # success check
            visible = any(
                o.get("objectType") == target_type and o.get("visible")
                for o in env.last_event.metadata["objects"]
            )
            d_final = min(
                math.hypot(p["x"] - t["x"], p["z"] - t["z"])
                for t in episode.target_instances
            )
            if d_final <= episode.success_distance and visible:
                success = True
                break
            # approach finished without success — record failure for the
            # giveup watchdog above and try again.  After N failures the
            # next iteration will fall through to IG-driven exploration.
            target_dets_failures += 1
            continue

        # 3. pick next viewpoint by max IG - lambda*travel
        # IG-strategy ablation: three viewpoint scoring functions, all
        # discounted by travel cost (lambda * Euclidean distance).
        #   "shannon"   : H(b) - E[H(b'|D, v)]      (Shannon entropy reduction)
        #   "heuristic" : sum of belief mass in v's FOV (IPPON-style)
        #   "greedy"    : belief at v's centre cell (max-belief greedy)
        cur_pos = obs["pose"]["position"]
        best = None
        best_score = -1e9
        for cand in candidates:
            cand_key = (round(cand["position"]["x"], 1),
                        round(cand["position"]["z"], 1))
            if cand_key in unreachable_goals:
                continue
            cand_node = position_to_node(cand["position"], env.grid_size)
            if ig_strategy == "shannon":
                ig = grid.expected_information_gain(
                    cand, fov_deg=fov_deg, max_distance=visibility_distance,
                    tp_rate=tp_rate, fp_rate=fp_rate,
                )
            elif ig_strategy == "heuristic":
                ig = grid.belief_in_fov(
                    cand, fov_deg=fov_deg, max_distance=visibility_distance,
                )
            elif ig_strategy == "greedy":
                ig = grid.belief_at_position(cand["position"])
            else:
                raise ValueError(f"unknown ig_strategy: {ig_strategy!r}")
            travel = math.hypot(
                cand["position"]["x"] - cur_pos["x"],
                cand["position"]["z"] - cur_pos["z"],
            )
            score = ig - ig_lambda * travel
            if score > best_score:
                best_score = score
                best = cand

        decisions += 1
        extra["entropies"].append(grid.entropy())
        if best is not None:
            extra["decisions"].append({
                "step": step,
                "best_ig": best_score,
                "goal": best["position"],
            })

        if save_dir and save_heatmaps_every and decisions % save_heatmaps_every == 0:
            os.makedirs(save_dir, exist_ok=True)
            grid.save_heatmap(
                os.path.join(save_dir, f"{episode.episode_id}_step{step:04d}.png"),
                agent_pose=obs["pose"],
                targets=episode.target_instances,
            )

        if best is None:
            break

        # 4. navigate to best viewpoint
        actions = plan_actions_to_position(
            reachable_positions=reachable,
            start_pose=obs["pose"],
            goal_position=best["position"],
            grid_size=env.grid_size,
        ) or []
        if not actions:
            # Either we're already on top of the cell (legitimate — just
            # rotate to the wanted yaw), or the path planner can't get
            # there from here (unreachable — blacklist so the IG picker
            # tries a different viewpoint next iteration).
            cur_node = position_to_node(obs["pose"]["position"], env.grid_size)
            goal_node = position_to_node(best["position"], env.grid_size)
            if cur_node == goal_node:
                actions = rotation_actions(
                    normalize_yaw(obs["pose"]["rotation"]["y"]),
                    best["rotation"]["y"],
                )
            else:
                blk_key = (round(best["position"]["x"], 1),
                           round(best["position"]["z"], 1))
                unreachable_goals.add(blk_key)
                extra["ig_blacklisted_goals"] += 1
                # rotate once so the agent isn't completely frozen
                actions = ["RotateRight"]

        any_step_taken = False
        for a in actions:
            if step >= max_steps:
                break
            obs = env.step(a)
            step += 1
            any_step_taken = True
            p = obs["pose"]["position"]
            traj.append({"x": p["x"], "y": p["y"], "z": p["z"]})
            if not obs["last_action_success"]:
                break
        if not any_step_taken:
            # path planner returned empty; rotate to break stalemate
            obs = env.step("RotateRight")
            step += 1
            p = obs["pose"]["position"]
            traj.append({"x": p["x"], "y": p["y"], "z": p["z"]})

        # watchdog: detect stuck agent (no spatial movement across decisions)
        new_node = position_to_node(obs["pose"]["position"], env.grid_size)
        if new_node == last_position_node:
            decisions_without_move += 1
        else:
            decisions_without_move = 0
            last_position_node = new_node
        if decisions_without_move >= 5:
            # walk forward 3 random rotates to break possible AI2-THOR stall
            for _ in range(3):
                if step >= max_steps:
                    break
                obs = env.step("RotateRight")
                step += 1
                p = obs["pose"]["position"]
                traj.append({"x": p["x"], "y": p["y"], "z": p["z"]})
            decisions_without_move = 0

    # final position & success
    p = obs["pose"]["position"]
    final = {"x": p["x"], "y": p["y"], "z": p["z"]}
    if not success:
        # last-frame visibility check (we may have stumbled into success)
        visible = any(
            o.get("objectType") == target_type and o.get("visible")
            for o in env.last_event.metadata["objects"]
        )
        d_final = min(
            math.hypot(p["x"] - t["x"], p["z"] - t["z"])
            for t in episode.target_instances
        )
        if d_final <= episode.success_distance and visible:
            success = True

    return {
        "success": success,
        "trajectory": traj,
        "num_steps": step,
        "final_position": final,
        "extra": extra,
    }


def _pick_approach_point(
    reachable: List[Dict[str, float]],
    target_pos: Dict[str, float],
    ideal_dist: float = 0.85,
) -> Dict[str, float]:
    best = None
    best_score = float("inf")
    for p in reachable:
        d = math.hypot(p["x"] - target_pos["x"], p["z"] - target_pos["z"])
        score = abs(d - ideal_dist)
        if score < best_score:
            best_score = score
            best = p
    return best or reachable[0]


# ============================================================
# Register into eval harness
# ============================================================

def grid_bayes_metadata_agent(env, episode, **kw):
    return grid_bayes_agent(env, episode, detector="metadata", **kw)


def grid_bayes_metadata_greedy_agent(env, episode, **kw):
    """Metadata oracle detector + greedy max-belief viewpoint selection.

    Ablation baseline for §4: replaces the Shannon-entropy IG with a naive
    "head for the cell with the highest target probability" heuristic, still
    discounted by travel cost.  Intended to expose how much of the agent's
    SR comes from the IG-based viewpoint selector vs. the Bayesian belief
    itself.
    """
    kw.setdefault("ig_strategy", "greedy")
    return grid_bayes_agent(env, episode, detector="metadata", **kw)


def grid_bayes_metadata_heuristic_agent(env, episode, **kw):
    """Metadata oracle detector + IPPON-style sum-belief-in-FOV viewpoint
    selection.

    Ablation baseline for §4: scores each candidate viewpoint by the total
    target-belief mass currently inside its FOV cone (no entropy reduction).
    This approximates the "information gain" heuristic used by IPPON-style
    Bayesian ObjNav planners.
    """
    kw.setdefault("ig_strategy", "heuristic")
    return grid_bayes_agent(env, episode, detector="metadata", **kw)


def grid_bayes_metadata_spatial_agent(env, episode, **kw):
    """Same as grid_bayes_metadata but with the synthetic spatial sensor model
    (TP decays with distance and viewing angle).  Demonstrates the spatial
    sensor model ablation without requiring offline YOLO collection.
    """
    return grid_bayes_agent(
        env, episode, detector="metadata",
        spatial_sensor_model="synthetic", **kw,
    )


def grid_bayes_yolo_agent(env, episode, **kw):
    kw.setdefault("yolo_device", os.environ.get("YOLO_DEVICE", "cuda:0"))
    kw.setdefault("yolo_model_path", "yolov8s-world.pt")
    kw.setdefault("yolo_open_vocab", True)
    return grid_bayes_agent(env, episode, detector="yolo", **kw)


def grid_bayes_yolo_spatial_agent(env, episode, **kw):
    """YOLO + synthetic spatial sensor model.

    This is the headline experiment: with a real noisy detector, the spatial
    sensor model (TP decays with distance and viewing angle) should help
    more than the uniform sensor model used in grid_bayes_yolo.
    """
    kw.setdefault("yolo_device", os.environ.get("YOLO_DEVICE", "cuda:0"))
    kw.setdefault("yolo_model_path", "yolov8s-world.pt")
    kw.setdefault("yolo_open_vocab", True)
    return grid_bayes_agent(
        env, episode, detector="yolo",
        spatial_sensor_model="synthetic", **kw,
    )


def grid_bayes_yolo_calibrated_agent(env, episode, **kw):
    """YOLO + spatial sensor model calibrated from real YOLO measurements.

    Uses outputs/yolo_calibrated.json (collected by sensor_model.py collect).
    Per-class (TP, FP) per (distance, angle) bin from 500 random views over
    all 5 scenes; bins with too few positives fall back to default.
    """
    kw.setdefault("yolo_device", os.environ.get("YOLO_DEVICE", "cuda:0"))
    kw.setdefault("yolo_model_path", "yolov8s-world.pt")
    kw.setdefault("yolo_open_vocab", True)
    return grid_bayes_agent(
        env, episode, detector="yolo",
        spatial_sensor_model="outputs/yolo_calibrated.json", **kw,
    )


# autoregister
try:
    from eval_objnav import AGENTS
    AGENTS.setdefault("grid_bayes_metadata", grid_bayes_metadata_agent)
    AGENTS.setdefault("grid_bayes_metadata_greedy",
                      grid_bayes_metadata_greedy_agent)
    AGENTS.setdefault("grid_bayes_metadata_heuristic",
                      grid_bayes_metadata_heuristic_agent)
    AGENTS.setdefault("grid_bayes_yolo", grid_bayes_yolo_agent)
except Exception:
    pass
