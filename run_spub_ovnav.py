
import os
import csv
import json
import math
import argparse
import builtins
from typing import Dict

import numpy as np

from sim_env import AI2ThorObjNavEnv
from semantic_prior import SemanticPrior
from yolo_detector import YoloWorldDetector
from nav_utils import (
    plan_actions_to_position,
    position_to_node,
    rotation_actions,
    normalize_yaw,
    yaw_to_face_position,
)


REAL_PRINT = builtins.print


def setup_quiet_print(quiet: bool, log_path: str):
    """
    quiet=True 时，普通 print 写入 log 文件，不刷屏。
    最终 summary 用 REAL_PRINT 单独输出。
    """
    if not quiet:
        return None

    folder = os.path.dirname(log_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    log_file = open(log_path, "w")

    def log_print(*args, **kwargs):
        text = " ".join(str(a) for a in args)
        log_file.write(text + "\n")
        log_file.flush()

    builtins.print = log_print
    return log_file


def restore_print(log_file):
    builtins.print = REAL_PRINT
    if log_file is not None:
        log_file.close()


# ============================================================
# Basic utils
# ============================================================

def norm_label(label: str) -> str:
    return label.lower().replace(" ", "").replace("_", "").replace("-", "")


def label_match(a: str, b: str) -> bool:
    na = norm_label(a)
    nb = norm_label(b)

    if na == nb:
        return True

    alias_groups = [
        {"sofa", "couch"},
        {"tv", "television", "monitor", "screen"},
        {"remotecontrol", "remote", "remotecontroller", "tvremote"},
        {"coffeetable", "table"},
        {"diningtable", "table"},
        {"sidetable", "table"},
        {"houseplant", "plant"},
        {"armchair", "chair"},
        {"floorlamp", "lamp"},
        {"desklamp", "lamp"},
    ]

    for group in alias_groups:
        if na in group and nb in group:
            return True

    return False


def is_table_like_target(target_query, target_prompt):
    table_names = [
        "coffee table",
        "table",
        "dining table",
        "side table",
    ]

    for name in table_names:
        if label_match(target_query, name) or label_match(target_prompt, name):
            return True

    return False


def distance_xz(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["z"] - b["z"]) ** 2)


def get_current_position(obs) -> Dict[str, float]:
    pos = obs["pose"]["position"]

    return {
        "x": pos["x"],
        "y": pos["y"],
        "z": pos["z"],
    }


def median_position(positions):
    if len(positions) == 0:
        return None

    return {
        "x": float(np.median([p["x"] for p in positions])),
        "y": float(np.median([p["y"] for p in positions])),
        "z": float(np.median([p["z"] for p in positions])),
        "depth": float(np.median([p.get("depth", 0.0) for p in positions])),
    }


def save_trajectory_csv(trajectory, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "x", "y", "z"])

        for i, p in enumerate(trajectory):
            writer.writerow([i, p["x"], p["y"], p["z"]])


def clamp(x, low=0.0, high=1.0):
    return max(low, min(high, x))


# ============================================================
# Prior context matching / semantic anchors
# ============================================================

def match_context_from_prior(det_label, context_weights):
    for ctx in context_weights.keys():
        if label_match(det_label, ctx):
            return ctx
    return None


def get_prior_weight_from_context(context_name, context_weights):
    if context_name is None:
        return 0.0

    for ctx, score in context_weights.items():
        if label_match(context_name, ctx):
            return score

    return 0.0


def get_anchor_usefulness(target_query, context_name):
    """
    语义锚点有效性。
    不是所有 context 都适合拿来找目标。

    例如找 remote control 时：
    coffee table / sofa / side table / tv stand 是高价值锚点；
    painting / plant / box 是弱锚点。
    """
    tq = norm_label(target_query)
    ctx = norm_label(context_name)

    table_like = {"coffeetable", "sidetable", "diningtable", "table", "tvstand"}
    sofa_like = {"sofa", "couch", "armchair", "chair"}
    surface_like = {"desk", "shelf", "drawer", "cabinet", "countertop"}
    weak_context = {
        "painting",
        "window",
        "houseplant",
        "plant",
        "box",
        "floorlamp",
        "desklamp",
    }

    if tq in {"remotecontrol", "remote", "tvremote"}:
        if ctx in table_like:
            return 1.20
        if ctx in sofa_like:
            return 1.10
        if ctx in surface_like:
            return 0.90
        if ctx in {"tv", "television", "monitor"}:
            return 0.75
        if ctx in {"pillow", "laptop"}:
            return 0.65
        if ctx in weak_context:
            return 0.25
        return 0.50

    if tq in {"book"}:
        if ctx in {"shelf", "desk", "drawer", "cabinet"}:
            return 1.20
        if ctx in table_like:
            return 1.00
        if ctx in {"bed", "pillow", "laptop"}:
            return 0.75
        if ctx in weak_context:
            return 0.30
        return 0.60

    if tq in {"pencil", "pen"}:
        if ctx in {"desk", "drawer", "laptop", "book"}:
            return 1.20
        if ctx in table_like:
            return 1.00
        if ctx in {"bed", "pillow"}:
            return 0.40
        if ctx in weak_context:
            return 0.25
        return 0.60

    if tq in {"laptop"}:
        if ctx in {"desk", "chair"}:
            return 1.20
        if ctx in table_like:
            return 1.05
        if ctx in {"sofa", "couch"}:
            return 0.80
        if ctx in weak_context:
            return 0.30
        return 0.60

    if tq in {"plant", "houseplant"}:
        if ctx in {"window", "floorlamp", "sidetable", "table"}:
            return 1.00
        if ctx in weak_context:
            return 0.70
        return 0.50

    if tq in {"coffeetable", "table"}:
        if ctx in {"sofa", "couch", "tv", "television", "pillow"}:
            return 1.00
        if ctx in weak_context:
            return 0.50
        return 0.70

    return 0.70


def is_anchor_search_target(target_query):
    """
    哪些目标需要语义锚点局部搜索。
    第一版重点针对小物体和容易被遮挡的物体。
    """
    tq = norm_label(target_query)

    return tq in {
        "remotecontrol",
        "remote",
        "tvremote",
        "book",
        "pencil",
        "pen",
        "laptop",
    }


def is_high_value_anchor(target_query, context_name):
    """
    判断 context 是否值得作为局部搜索锚点。
    """
    usefulness = get_anchor_usefulness(target_query, context_name)
    return usefulness >= 0.90


def make_anchor_key(ctx):
    """
    避免同一个 anchor 被重复搜索。
    """
    p = ctx["position"]

    return (
        norm_label(ctx["matched_context"]),
        round(p["x"], 1),
        round(p["z"], 1),
    )


def select_anchor_view_goal(
    current_position,
    anchor_position,
    reachable_positions,
):
    """
    选择一个适合观察 anchor 的 reachable point。
    对桌子/沙发/桌面类锚点，不要太近。
    """
    min_dist = 0.85
    max_dist = 1.65
    desired_dist = 1.20

    best = None
    best_score = float("inf")

    for p in reachable_positions:
        d_anchor = distance_xz(p, anchor_position)

        if d_anchor < min_dist or d_anchor > max_dist:
            continue

        d_agent = distance_xz(current_position, p)

        score = abs(d_anchor - desired_dist) + 0.04 * d_agent

        if score < best_score:
            best_score = score
            best = p

    if best is None:
        best = min(
            reachable_positions,
            key=lambda p: distance_xz(p, anchor_position)
        )

    return best


def select_best_anchor_candidate(
    target_query,
    observed_contexts,
    searched_anchors,
    current_position,
    start_pose,
    reachable_positions,
    grid_size=0.25,
):
    """
    从已经检测到的 context objects 里选择最值得局部搜索的 anchor。

    anchor_score =
        effective_prior × detection_score × count_bonus
        - path_cost
    """
    if not is_anchor_search_target(target_query):
        return None, None

    best_ctx = None
    best_score = -1e9
    best_info = None

    for ctx in observed_contexts:
        matched_context = ctx["matched_context"]

        if not is_high_value_anchor(target_query, matched_context):
            continue

        key = make_anchor_key(ctx)

        if key in searched_anchors:
            continue

        anchor_pos = ctx["position"]

        view_goal = select_anchor_view_goal(
            current_position=current_position,
            anchor_position=anchor_pos,
            reachable_positions=reachable_positions,
        )

        actions = plan_actions_to_position(
            reachable_positions=reachable_positions,
            start_pose=start_pose,
            goal_position=view_goal,
            grid_size=grid_size,
        )

        if actions is None:
            continue

        path_cost = len(actions)
        count_bonus = min(1.0, 0.5 + 0.15 * ctx.get("count", 1))

        semantic_score = (
            ctx.get("prior_weight", 0.0)
            * ctx.get("score", 0.0)
            * count_bonus
        )

        score = 8.0 * semantic_score - 0.12 * path_cost

        if score > best_score:
            best_score = score
            best_ctx = ctx
            best_info = {
                "anchor": matched_context,
                "anchor_position": anchor_pos,
                "view_goal": view_goal,
                "semantic_score": semantic_score,
                "path_cost": path_cost,
                "score": score,
            }

    return best_ctx, best_info


# ============================================================
# Detection validity
# ============================================================

def box_size(det):
    x1, y1, x2, y2 = det["box"]

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h

    return w, h, area


def is_valid_detection(
    det,
    image_width=640,
    image_height=480,
    min_score=0.18,
    min_width=5,
    min_height=5,
    min_area_ratio=0.00008,
):
    if det["score"] < min_score:
        return False

    w, h, area = box_size(det)
    area_ratio = area / float(image_width * image_height)

    if w < min_width:
        return False

    if h < min_height:
        return False

    if area_ratio < min_area_ratio:
        return False

    return True


def find_target_detection(
    detections,
    target_query,
    target_prompt,
    image_width=640,
    image_height=480,
):
    candidates = []

    for d in detections:
        if label_match(d["label"], target_query) or label_match(d["label"], target_prompt):
            candidates.append(d)

    if not candidates:
        return False, None, "not_detected"

    best = max(candidates, key=lambda d: d["score"] * box_size(d)[2])

    if not is_valid_detection(best, image_width, image_height):
        return False, best, "too_small_or_low_conf"

    return True, best, "valid"


# ============================================================
# YOLO box + depth -> world position
# ============================================================

def yolo_box_to_world_position(
    det,
    depth,
    pose,
    image_width=640,
    image_height=480,
    field_of_view=60,
):
    if depth is None:
        return None

    x1, y1, x2, y2 = det["box"]

    x1 = int(max(0, min(image_width - 1, x1)))
    x2 = int(max(0, min(image_width - 1, x2)))
    y1 = int(max(0, min(image_height - 1, y1)))
    y2 = int(max(0, min(image_height - 1, y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1

    px1 = int(x1 + 0.2 * box_w)
    px2 = int(x2 - 0.2 * box_w)
    py1 = int(y1 + 0.2 * box_h)
    py2 = int(y2 - 0.2 * box_h)

    px1 = max(0, min(image_width - 1, px1))
    px2 = max(0, min(image_width, px2))
    py1 = max(0, min(image_height - 1, py1))
    py2 = max(0, min(image_height, py2))

    if px2 <= px1 or py2 <= py1:
        return None

    depth_patch = depth[py1:py2, px1:px2]
    valid_depth = depth_patch[np.isfinite(depth_patch)]
    valid_depth = valid_depth[valid_depth > 0.05]

    if len(valid_depth) == 0:
        return None

    z_cam = float(np.median(valid_depth))

    cx_box = int((x1 + x2) / 2)

    fov_rad = math.radians(field_of_view)
    fy = image_height / (2.0 * math.tan(fov_rad / 2.0))
    fx = fy * image_width / image_height

    cx = image_width / 2.0
    x_cam = (cx_box - cx) * z_cam / fx

    agent_pos = pose["position"]
    yaw = math.radians(pose["rotation"]["y"])

    world_x = agent_pos["x"] + math.cos(yaw) * x_cam + math.sin(yaw) * z_cam
    world_z = agent_pos["z"] - math.sin(yaw) * x_cam + math.cos(yaw) * z_cam

    return {
        "x": world_x,
        "y": agent_pos["y"],
        "z": world_z,
        "depth": z_cam,
    }


# ============================================================
# Context memory
# ============================================================

def add_or_update_context(
    observed_contexts,
    observed_keys,
    label,
    matched_context,
    prior_weight,
    score,
    position,
    merge_radius=0.6,
):
    for ctx in observed_contexts:
        if not label_match(ctx["matched_context"], matched_context):
            continue

        d = distance_xz(ctx["position"], position)

        if d <= merge_radius:
            old_w = ctx["count"]
            new_w = old_w + 1

            ctx["position"]["x"] = (ctx["position"]["x"] * old_w + position["x"]) / new_w
            ctx["position"]["y"] = (ctx["position"]["y"] * old_w + position["y"]) / new_w
            ctx["position"]["z"] = (ctx["position"]["z"] * old_w + position["z"]) / new_w
            ctx["score"] = max(ctx["score"], score)
            ctx["prior_weight"] = max(ctx.get("prior_weight", 0.0), prior_weight)
            ctx["count"] = new_w
            return

    key = (
        norm_label(label),
        round(position["x"], 1),
        round(position["z"], 1),
    )

    if key in observed_keys:
        return

    observed_keys.add(key)

    observed_contexts.append({
        "label": label,
        "matched_context": matched_context,
        "prior_weight": prior_weight,
        "score": score,
        "position": {
            "x": position["x"],
            "y": position["y"],
            "z": position["z"],
        },
        "count": 1,
    })


def update_observed_contexts_yolo_only(
    obs,
    detections,
    target_query,
    target_prompt,
    context_weights,
    observed_contexts,
    observed_keys,
    image_width=640,
    image_height=480,
    field_of_view=60,
):
    for det in detections:
        matched_context = match_context_from_prior(
            det_label=det["label"],
            context_weights=context_weights,
        )

        if matched_context is None:
            continue

        if not is_valid_detection(det, image_width, image_height):
            continue

        raw_prior_weight = get_prior_weight_from_context(
            matched_context,
            context_weights,
        )

        anchor_usefulness = get_anchor_usefulness(
            target_query=target_query,
            context_name=matched_context,
        )

        prior_weight = raw_prior_weight * anchor_usefulness

        pos = yolo_box_to_world_position(
            det=det,
            depth=obs["depth"],
            pose=obs["pose"],
            image_width=image_width,
            image_height=image_height,
            field_of_view=field_of_view,
        )

        if pos is None:
            continue

        print(
            f"[Context detected] label={det['label']}, "
            f"matched_context={matched_context}, "
            f"raw_prior={raw_prior_weight:.3f}, "
            f"anchor={anchor_usefulness:.2f}, "
            f"effective_prior={prior_weight:.3f}, "
            f"score={det['score']:.3f}"
        )

        add_or_update_context(
            observed_contexts=observed_contexts,
            observed_keys=observed_keys,
            label=det["label"],
            matched_context=matched_context,
            prior_weight=prior_weight,
            score=det["score"],
            position=pos,
        )


# ============================================================
# Belief map
# ============================================================

def init_belief_map(reachable_positions, grid_size=0.25):
    belief_map = {}

    for p in reachable_positions:
        node = position_to_node(p, grid_size)
        belief_map[node] = {
            "belief": 0.0,
            "context_score": 0.0,
            "uncertainty": 1.0,
            "seen_count": 0.0,
            "negative_count": 0.0,
            "visit_count": 0.0,
        }

    return belief_map


def mark_visit(belief_map, position, grid_size=0.25):
    node = position_to_node(position, grid_size)

    if node in belief_map:
        belief_map[node]["visit_count"] += 1.0


def mark_negative_observation(
    belief_map,
    current_position,
    reachable_positions,
    grid_size=0.25,
    radius=1.75,
    negative_increment=0.5,
):
    for p in reachable_positions:
        node = position_to_node(p, grid_size)

        if node not in belief_map:
            continue

        d = distance_xz(current_position, p)

        if d <= radius:
            belief_map[node]["seen_count"] += negative_increment
            belief_map[node]["negative_count"] += negative_increment


def recompute_belief_map_from_contexts(
    belief_map,
    reachable_positions,
    observed_contexts,
    grid_size=0.25,
    sigma=1.5,
):
    for state in belief_map.values():
        state["context_score"] = 0.0

    for p in reachable_positions:
        node = position_to_node(p, grid_size)

        if node not in belief_map:
            continue

        p_not = 1.0

        for ctx in observed_contexts:
            ctx_pos = ctx["position"]
            d = distance_xz(p, ctx_pos)

            prior_weight = ctx.get("prior_weight", 0.0)
            det_score = ctx.get("score", 0.0)
            count_bonus = min(1.0, 0.5 + 0.15 * ctx.get("count", 1))

            if prior_weight <= 0:
                continue

            evidence = (
                prior_weight
                * det_score
                * count_bonus
                * math.exp(-d / sigma)
            )

            evidence = clamp(evidence, 0.0, 0.95)
            p_not *= (1.0 - evidence)

        context_score = 1.0 - p_not

        state = belief_map[node]
        state["context_score"] = context_score

        negative_decay = 0.65 ** state["negative_count"]
        belief = context_score * negative_decay

        state["belief"] = clamp(belief, 0.0, 1.0)
        state["uncertainty"] = 1.0 / (1.0 + state["seen_count"])


def select_belief_goal(
    current_position,
    start_pose,
    reachable_positions,
    visited,
    belief_map,
    grid_size=0.25,
):
    """
    Speed-aware goal selection:
    用真实 path action cost，而不是欧氏距离。
    """
    best_pos = None
    best_score = -1e9
    best_info = None

    alpha = 10.0
    beta = 1.2
    gamma = 2.0
    lambda_path = 0.10
    eta = 1.0
    rho = 0.45
    neg_lambda = 1.2

    for p in reachable_positions:
        node = position_to_node(p, grid_size)

        if node in visited:
            continue

        if node not in belief_map:
            continue

        actions = plan_actions_to_position(
            reachable_positions=reachable_positions,
            start_pose=start_pose,
            goal_position=p,
            grid_size=grid_size,
        )

        if actions is None:
            continue

        path_cost = len(actions)
        state = belief_map[node]

        belief = state["belief"]
        uncertainty = state["uncertainty"]
        context_score = state["context_score"]
        visit_count = state["visit_count"]
        seen_count = state["seen_count"]
        negative_count = state["negative_count"]

        hard_negative_penalty = 0.0
        if negative_count >= 2.0:
            hard_negative_penalty = 3.0

        score = (
            alpha * belief
            + beta * uncertainty
            + gamma * context_score
            - lambda_path * path_cost
            - eta * visit_count
            - rho * seen_count
            - neg_lambda * negative_count
            - hard_negative_penalty
        )

        if score > best_score:
            best_score = score
            best_pos = p
            best_info = {
                "belief": belief,
                "uncertainty": uncertainty,
                "context_score": context_score,
                "path_cost": path_cost,
                "seen_count": seen_count,
                "negative_count": negative_count,
                "visit_count": visit_count,
                "score": score,
            }

    return best_pos, best_info


def print_top_belief_nodes(belief_map, reachable_positions, grid_size=0.25, top_k=10):
    scored = []

    for p in reachable_positions:
        node = position_to_node(p, grid_size)

        if node not in belief_map:
            continue

        state = belief_map[node]

        scored.append((
            state["belief"],
            state["context_score"],
            state["uncertainty"],
            state["negative_count"],
            p,
        ))

    scored.sort(key=lambda x: x[0], reverse=True)

    print("\nTop belief nodes:")
    for i, item in enumerate(scored[:top_k]):
        belief, context_score, uncertainty, negative_count, p = item
        print(
            f"  {i}: pos={p}, "
            f"belief={belief:.3f}, "
            f"context={context_score:.3f}, "
            f"uncertainty={uncertainty:.3f}, "
            f"negative={negative_count:.1f}"
        )


# ============================================================
# YOLO observation
# ============================================================

def observe_yolo_once(
    env,
    obs,
    detector,
    target_query,
    target_prompt,
    context_weights,
    observed_contexts,
    observed_keys,
    belief_map,
    reachable_positions,
    grid_size,
    save_path=None,
    debug_path=None,
    image_width=640,
    image_height=480,
    field_of_view=60,
):
    if save_path is not None:
        env.save_rgb(save_path)

    detections = detector.detect(obs["rgb"])

    if debug_path is not None:
        detector.save_debug_image(
            rgb_image=obs["rgb"],
            detections=detections,
            path=debug_path,
            target_prompt=target_prompt,
        )

    update_observed_contexts_yolo_only(
        obs=obs,
        detections=detections,
        target_query=target_query,
        target_prompt=target_prompt,
        context_weights=context_weights,
        observed_contexts=observed_contexts,
        observed_keys=observed_keys,
        image_width=image_width,
        image_height=image_height,
        field_of_view=field_of_view,
    )

    found, target_det, status = find_target_detection(
        detections=detections,
        target_query=target_query,
        target_prompt=target_prompt,
        image_width=image_width,
        image_height=image_height,
    )

    print(
        f"[Target check] target_query={target_query}, "
        f"target_prompt={target_prompt}, "
        f"status={status}, det={target_det}"
    )

    if not found:
        mark_negative_observation(
            belief_map=belief_map,
            current_position=get_current_position(obs),
            reachable_positions=reachable_positions,
            grid_size=grid_size,
            radius=1.75,
            negative_increment=0.5,
        )

    recompute_belief_map_from_contexts(
        belief_map=belief_map,
        reachable_positions=reachable_positions,
        observed_contexts=observed_contexts,
        grid_size=grid_size,
    )

    target_position = None

    if found:
        target_position = yolo_box_to_world_position(
            det=target_det,
            depth=obs["depth"],
            pose=obs["pose"],
            image_width=image_width,
            image_height=image_height,
            field_of_view=field_of_view,
        )

        if target_position is None:
            found = False
            status = "depth_invalid"

    return found, target_det, target_position, status, detections


def observe_by_rotating_yolo(
    env,
    obs,
    detector,
    target_query,
    target_prompt,
    context_weights,
    step_id,
    save_dir,
    observed_contexts,
    observed_keys,
    belief_map,
    reachable_positions,
    grid_size,
    image_width=640,
    image_height=480,
    field_of_view=60,
):
    rotate_actions = 0
    target_positions = []
    best_target_det = None

    for r in range(4):
        save_path = f"{save_dir}/observe_step_{step_id}_rot_{r}.png"
        debug_path = f"{save_dir}/observe_step_{step_id}_rot_{r}_yolo.png"

        found, target_det, target_position, status, detections = observe_yolo_once(
            env=env,
            obs=obs,
            detector=detector,
            target_query=target_query,
            target_prompt=target_prompt,
            context_weights=context_weights,
            observed_contexts=observed_contexts,
            observed_keys=observed_keys,
            belief_map=belief_map,
            reachable_positions=reachable_positions,
            grid_size=grid_size,
            save_path=save_path,
            debug_path=debug_path,
            image_width=image_width,
            image_height=image_height,
            field_of_view=field_of_view,
        )

        labels = [
            (
                d["label"],
                round(d["score"], 3),
                [round(x, 1) for x in d["box"]],
            )
            for d in detections
        ]

        print(f"Observe rotation {r}:")
        print("YOLO detections:", labels)
        print("Target status:", status)

        if found and target_position is not None:
            target_positions.append(target_position)
            best_target_det = target_det

        obs = env.step("RotateRight")
        rotate_actions += 1

    stable_target_position = median_position(target_positions)

    if stable_target_position is not None:
        print("Stable target position from multi-view YOLO-depth:", stable_target_position)
        print("Best target detection:", best_target_det)
        return True, obs, stable_target_position, rotate_actions

    return False, obs, None, rotate_actions


# ============================================================
# Approach / anchor local search
# ============================================================

def select_approach_goal(
    current_position,
    target_position,
    reachable_positions,
    target_query=None,
    target_prompt=None,
):
    if target_query is not None and target_prompt is not None and is_table_like_target(target_query, target_prompt):
        min_dist = 1.05
        max_dist = 1.75
        desired_dist = 1.35
    else:
        min_dist = 0.65
        max_dist = 1.15
        desired_dist = 0.85

    best = None
    best_score = float("inf")

    for p in reachable_positions:
        d_obj = distance_xz(p, target_position)

        if d_obj < min_dist or d_obj > max_dist:
            continue

        d_agent = distance_xz(current_position, p)

        score = abs(d_obj - desired_dist) + 0.05 * d_agent

        if score < best_score:
            best_score = score
            best = p

    if best is None:
        best = min(
            reachable_positions,
            key=lambda p: distance_xz(p, target_position)
        )

    return best


def rotate_to_face_target(env, obs, target_position, save_dir, trajectory):
    current_position = get_current_position(obs)
    desired_yaw = yaw_to_face_position(current_position, target_position)
    current_yaw = normalize_yaw(obs["pose"]["rotation"]["y"])

    actions = rotation_actions(current_yaw, desired_yaw)

    num_actions = 0

    for i, action in enumerate(actions):
        obs = env.step(action)
        trajectory.append(get_current_position(obs))
        num_actions += 1

        print(
            f"Face target action {i}: {action}, "
            f"success={obs['last_action_success']}"
        )

        env.save_rgb(f"{save_dir}/face_target_{i:03d}_{action}.png")

        if not obs["last_action_success"]:
            print("Rotate to face target failed:", obs["error_message"])
            break

    return obs, num_actions


def search_around_anchor(
    env,
    obs,
    detector,
    target_query,
    target_prompt,
    context_weights,
    observed_contexts,
    observed_keys,
    belief_map,
    reachable_positions,
    grid_size,
    anchor_ctx,
    save_dir,
    trajectory,
    image_width=640,
    image_height=480,
    field_of_view=60,
):
    """
    围绕语义锚点做局部主动搜索。

    例如 target=remote control, anchor=coffee table:
        1. 走到 coffee table 附近
        2. 面向 coffee table
        3. LookDown
        4. 原地旋转扫描
        5. 如果检测到 target，返回 target_position
    """
    anchor_name = anchor_ctx["matched_context"]
    anchor_position = anchor_ctx["position"]

    safe_anchor = norm_label(anchor_name)

    print("\n=== Semantic Anchor Local Search ===")
    print("Anchor:", anchor_name)
    print("Anchor position:", anchor_position)

    current_position = get_current_position(obs)

    view_goal = select_anchor_view_goal(
        current_position=current_position,
        anchor_position=anchor_position,
        reachable_positions=reachable_positions,
    )

    print("Selected anchor view goal:", view_goal)
    print("Distance to anchor:", round(distance_xz(view_goal, anchor_position), 3))

    actions = plan_actions_to_position(
        reachable_positions=reachable_positions,
        start_pose=obs["pose"],
        goal_position=view_goal,
        grid_size=grid_size,
    )

    if actions is None:
        print("No path to anchor view goal.")
        return obs, False, None, 0

    total_actions = 0
    target_positions = []

    print("Anchor approach actions:", actions)

    for i, action in enumerate(actions):
        obs = env.step(action)
        trajectory.append(get_current_position(obs))
        total_actions += 1

        mark_visit(
            belief_map=belief_map,
            position=get_current_position(obs),
            grid_size=grid_size,
        )

        print(
            f"Anchor approach action {i}: {action}, "
            f"success={obs['last_action_success']}"
        )

        env.save_rgb(
            f"{save_dir}/anchor_{safe_anchor}_approach_{i:03d}_{action}.png"
        )

        if not obs["last_action_success"]:
            print("Anchor approach action failed:", obs["error_message"])
            return obs, False, None, total_actions

        found, target_det, target_position, status, detections = observe_yolo_once(
            env=env,
            obs=obs,
            detector=detector,
            target_query=target_query,
            target_prompt=target_prompt,
            context_weights=context_weights,
            observed_contexts=observed_contexts,
            observed_keys=observed_keys,
            belief_map=belief_map,
            reachable_positions=reachable_positions,
            grid_size=grid_size,
            save_path=None,
            debug_path=f"{save_dir}/anchor_{safe_anchor}_move_{i:03d}_yolo.png",
            image_width=image_width,
            image_height=image_height,
            field_of_view=field_of_view,
        )

        print("Anchor move-time target status:", status)

        if found and target_position is not None:
            target_positions.append(target_position)

    obs, n_rot = rotate_to_face_target(
        env=env,
        obs=obs,
        target_position=anchor_position,
        save_dir=save_dir,
        trajectory=trajectory,
    )
    total_actions += n_rot

    local_actions = [
        None,
        "LookDown",
        "RotateRight",
        "RotateRight",
        "RotateRight",
        "LookUp",
        "RotateRight",
        "LookDown",
    ]

    for j, action in enumerate(local_actions):
        if action is not None:
            obs = env.step(action)
            trajectory.append(get_current_position(obs))
            total_actions += 1

            mark_visit(
                belief_map=belief_map,
                position=get_current_position(obs),
                grid_size=grid_size,
            )

        found, target_det, target_position, status, detections = observe_yolo_once(
            env=env,
            obs=obs,
            detector=detector,
            target_query=target_query,
            target_prompt=target_prompt,
            context_weights=context_weights,
            observed_contexts=observed_contexts,
            observed_keys=observed_keys,
            belief_map=belief_map,
            reachable_positions=reachable_positions,
            grid_size=grid_size,
            save_path=None,
            debug_path=f"{save_dir}/anchor_{safe_anchor}_local_{j:02d}_{action or 'None'}_yolo.png",
            image_width=image_width,
            image_height=image_height,
            field_of_view=field_of_view,
        )

        print(
            f"Anchor local search {j}, action={action}, "
            f"target status={status}"
        )

        if found and target_position is not None:
            target_positions.append(target_position)

            stable_target_position = median_position(target_positions)

            print("Anchor search found target.")
            print("Stable target position:", stable_target_position)

            return obs, True, stable_target_position, total_actions

    print("Anchor search failed.")
    return obs, False, None, total_actions


def final_verify_active(
    env,
    obs,
    detector,
    target_query,
    target_prompt,
    save_dir,
    trajectory,
    image_width=640,
    image_height=480,
):
    best_det = None
    best_status = "not_detected"
    total_actions = 0

    view_actions = [
        None,
        "LookDown",
        "RotateRight",
        "RotateRight",
        "RotateRight",
        "LookUp",
        "RotateRight",
        "LookDown",
    ]

    for i, action in enumerate(view_actions):
        if action is not None:
            obs = env.step(action)
            trajectory.append(get_current_position(obs))
            total_actions += 1

        detections = detector.detect(obs["rgb"])

        detector.save_debug_image(
            rgb_image=obs["rgb"],
            detections=detections,
            path=f"{save_dir}/final_verify_{i:02d}_{action or 'None'}_yolo.png",
            target_prompt=target_prompt,
        )

        found, det, status = find_target_detection(
            detections,
            target_query=target_query,
            target_prompt=target_prompt,
            image_width=image_width,
            image_height=image_height,
        )

        print(f"Final verify {i}, action={action}, status={status}, det={det}")

        if found:
            return obs, True, det, status, total_actions

        if det is not None:
            best_det = det
            best_status = status

    return obs, False, best_det, best_status, total_actions


def approach_target(
    env,
    obs,
    detector,
    target_query,
    target_prompt,
    target_position,
    reachable_positions,
    grid_size,
    save_dir,
    trajectory,
    image_width=640,
    image_height=480,
    field_of_view=60,
):
    current_position = get_current_position(obs)

    approach_goal = select_approach_goal(
        current_position=current_position,
        target_position=target_position,
        reachable_positions=reachable_positions,
        target_query=target_query,
        target_prompt=target_prompt,
    )

    print("\n=== Approach Target ===")
    print("Stable estimated target position:", target_position)
    print("Selected approach goal:", approach_goal)
    print("Goal distance to estimated target:", round(distance_xz(approach_goal, target_position), 3))

    actions = plan_actions_to_position(
        reachable_positions=reachable_positions,
        start_pose=obs["pose"],
        goal_position=approach_goal,
        grid_size=grid_size,
    )

    if actions is None:
        print("No path found for approach.")
        return obs, False, 0

    print("Approach actions:", actions)
    print("Number of approach actions:", len(actions))

    num_actions = 0

    for i, action in enumerate(actions):
        obs = env.step(action)
        trajectory.append(get_current_position(obs))
        num_actions += 1

        print(
            f"Approach action {i}: {action}, "
            f"success={obs['last_action_success']}"
        )

        env.save_rgb(f"{save_dir}/approach_action_{i:03d}_{action}.png")

        if not obs["last_action_success"]:
            print("Approach action failed:", obs["error_message"])
            return obs, False, num_actions

    obs, n_rot = rotate_to_face_target(
        env=env,
        obs=obs,
        target_position=target_position,
        save_dir=save_dir,
        trajectory=trajectory,
    )
    num_actions += n_rot

    env.save_rgb(f"{save_dir}/after_approach_face_target.png")

    final_position = get_current_position(obs)
    final_distance = distance_xz(final_position, target_position)

    obs, final_found, final_det, final_status, verify_actions = final_verify_active(
        env=env,
        obs=obs,
        detector=detector,
        target_query=target_query,
        target_prompt=target_prompt,
        save_dir=save_dir,
        trajectory=trajectory,
        image_width=image_width,
        image_height=image_height,
    )

    num_actions += verify_actions

    print("\n=== Approach Finished ===")
    print("Final agent position:", final_position)
    print("Final distance to estimated target:", round(final_distance, 3))
    print("Final YOLO status:", final_status)
    print("Final YOLO target detection:", final_det)

    if is_table_like_target(target_query, target_prompt):
        success = final_distance <= 1.6 and final_found
    else:
        success = final_distance <= 1.15 and final_found

    return obs, success, num_actions


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene", type=str, default="FloorPlan201")
    parser.add_argument("--target", type=str, default="coffee table")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_goals", type=int, default=50)
    parser.add_argument("--debug_force_search", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--result_json", type=str, default=None)

    args = parser.parse_args()

    scene = args.scene
    target_query = args.target
    debug_force_search = args.debug_force_search

    safe_target = target_query.replace(" ", "_")

    if args.save_dir is None:
        save_dir = f"outputs/spub_ovnav/{scene}_{safe_target}"
    else:
        save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    log_path = os.path.join(save_dir, "run.log")
    log_file = setup_quiet_print(args.quiet, log_path)

    width = 640
    height = 480
    field_of_view = 60
    grid_size = 0.25

    prior_db = SemanticPrior("outputs/prior_build/semantic_prior.json")

    matched = prior_db.resolve_query(target_query)

    if len(matched) > 0:
        target_prompt = matched[0][0]
    else:
        target_prompt = target_query

    context_items = prior_db.get_navigation_contexts_for_query(
        query=target_query,
        top_k_contexts=8,
        min_score=0.05,
    )

    context_prompts = [ctx for ctx, score in context_items]
    context_weights = {ctx: score for ctx, score in context_items}

    extra_classes = [
        "chair",
        "arm chair",
        "sofa",
        "couch",
        "table",
        "coffee table",
        "dining table",
        "side table",
        "desk",
        "tv",
        "television",
        "monitor",
        "tv stand",
        "lamp",
        "floor lamp",
        "desk lamp",
        "plant",
        "house plant",
        "box",
        "book",
        "newspaper",
        "remote control",
        "pen",
        "pencil",
        "laptop",
        "pillow",
    ]

    yolo_classes = sorted(set([target_query, target_prompt] + context_prompts + extra_classes))

    print("\n=== Scene ===")
    print(scene)

    print("\n=== Target Query ===")
    print(target_query)

    print("\n=== Resolved Target Prompt ===")
    print(target_prompt)

    print("\n=== Debug Force Search ===")
    print(debug_force_search)

    print("\n=== Contexts From Semantic Prior ===")
    for ctx, score in context_items:
        print(f"{ctx:20s} score={score:.3f}")

    print("\n=== YOLO Classes ===")
    print(yolo_classes)

    env = AI2ThorObjNavEnv(
        scene=scene,
        width=width,
        height=height,
        grid_size=grid_size,
        rotate_step_degrees=90,
        field_of_view=field_of_view,
        headless=True,
    )

    obs = env.reset()

    print("\n=== Initial Pose ===")
    print(obs["pose"])

    reachable_positions = env.get_reachable_positions()
    print("\nNumber of reachable positions:", len(reachable_positions))

    belief_map = init_belief_map(
        reachable_positions=reachable_positions,
        grid_size=grid_size,
    )

    mark_visit(
        belief_map=belief_map,
        position=get_current_position(obs),
        grid_size=grid_size,
    )

    detector = YoloWorldDetector(
        classes=yolo_classes,
        model_name="yolov8s-world.pt",
        conf=0.12,
        device=args.device,
    )

    visited = set()
    observed_contexts = []
    observed_keys = set()
    searched_anchors = set()
    anchor_searches = 0

    trajectory = [get_current_position(obs)]

    max_goals = args.max_goals
    total_actions = 0
    success = False

    env.save_rgb(f"{save_dir}/start.png")

    current_node = position_to_node(get_current_position(obs), grid_size)
    visited.add(current_node)

    print("\n========== Initial YOLO Observation ==========")

    found, obs, target_position, rotate_actions = observe_by_rotating_yolo(
        env=env,
        obs=obs,
        detector=detector,
        target_query=target_query,
        target_prompt=target_prompt,
        context_weights=context_weights,
        step_id=0,
        save_dir=save_dir,
        observed_contexts=observed_contexts,
        observed_keys=observed_keys,
        belief_map=belief_map,
        reachable_positions=reachable_positions,
        grid_size=grid_size,
        image_width=width,
        image_height=height,
        field_of_view=field_of_view,
    )

    for _ in range(rotate_actions):
        trajectory.append(get_current_position(obs))

    total_actions += rotate_actions

    print("Initial observed context objects:")
    for ctx in observed_contexts:
        print(ctx)

    recompute_belief_map_from_contexts(
        belief_map=belief_map,
        reachable_positions=reachable_positions,
        observed_contexts=observed_contexts,
        grid_size=grid_size,
    )

    print_top_belief_nodes(
        belief_map=belief_map,
        reachable_positions=reachable_positions,
        grid_size=grid_size,
        top_k=10,
    )

    if found and target_position is not None and not debug_force_search:
        print("\nYOLO found stable target at initial area. Now approaching target...")

        obs, approach_success, n = approach_target(
            env=env,
            obs=obs,
            detector=detector,
            target_query=target_query,
            target_prompt=target_prompt,
            target_position=target_position,
            reachable_positions=reachable_positions,
            grid_size=grid_size,
            save_dir=save_dir,
            trajectory=trajectory,
            image_width=width,
            image_height=height,
            field_of_view=field_of_view,
        )

        total_actions += n
        success = approach_success

    goal_id = 0

    while not success and goal_id < max_goals:
        current_position = get_current_position(obs)

        recompute_belief_map_from_contexts(
            belief_map=belief_map,
            reachable_positions=reachable_positions,
            observed_contexts=observed_contexts,
            grid_size=grid_size,
        )

        print_top_belief_nodes(
            belief_map=belief_map,
            reachable_positions=reachable_positions,
            grid_size=grid_size,
            top_k=5,
        )

        # ====================================================
        # Semantic Anchor Local Search
        # ====================================================

        anchor_ctx, anchor_info = select_best_anchor_candidate(
            target_query=target_query,
            observed_contexts=observed_contexts,
            searched_anchors=searched_anchors,
            current_position=current_position,
            start_pose=obs["pose"],
            reachable_positions=reachable_positions,
            grid_size=grid_size,
        )

        if anchor_ctx is not None:
            anchor_searches += 1
            anchor_key = make_anchor_key(anchor_ctx)
            searched_anchors.add(anchor_key)

            print("\n========== Anchor Search ==========")
            print("Anchor info:", anchor_info)

            obs, anchor_found, anchor_target_position, n_anchor = search_around_anchor(
                env=env,
                obs=obs,
                detector=detector,
                target_query=target_query,
                target_prompt=target_prompt,
                context_weights=context_weights,
                observed_contexts=observed_contexts,
                observed_keys=observed_keys,
                belief_map=belief_map,
                reachable_positions=reachable_positions,
                grid_size=grid_size,
                anchor_ctx=anchor_ctx,
                save_dir=save_dir,
                trajectory=trajectory,
                image_width=width,
                image_height=height,
                field_of_view=field_of_view,
            )

            total_actions += n_anchor

            if anchor_found and anchor_target_position is not None:
                print("\nAnchor search found target. Now approaching target...")

                obs, approach_success, n = approach_target(
                    env=env,
                    obs=obs,
                    detector=detector,
                    target_query=target_query,
                    target_prompt=target_prompt,
                    target_position=anchor_target_position,
                    reachable_positions=reachable_positions,
                    grid_size=grid_size,
                    save_dir=save_dir,
                    trajectory=trajectory,
                    image_width=width,
                    image_height=height,
                    field_of_view=field_of_view,
                )

                total_actions += n
                success = approach_success

                if success:
                    break

            goal_id += 1
            continue

        # ====================================================
        # Global belief search
        # ====================================================

        goal_position, goal_info = select_belief_goal(
            current_position=current_position,
            start_pose=obs["pose"],
            reachable_positions=reachable_positions,
            visited=visited,
            belief_map=belief_map,
            grid_size=grid_size,
        )

        if goal_position is None:
            print("\nNo more unvisited reachable positions.")
            break

        goal_node = position_to_node(goal_position, grid_size)
        visited.add(goal_node)

        print(f"\n========== Belief Goal {goal_id} ==========")
        print("Current position:", current_position)
        print("Selected goal:", goal_position)
        print("Goal info:", goal_info)
        print("Observed context objects:")
        for ctx in observed_contexts:
            print(ctx)

        actions = plan_actions_to_position(
            reachable_positions=reachable_positions,
            start_pose=obs["pose"],
            goal_position=goal_position,
            grid_size=grid_size,
        )

        if actions is None:
            print("No path found to this goal. Skip.")
            goal_id += 1
            continue

        print("Planned actions:", actions)
        print("Number of actions:", len(actions))

        action_failed = False
        moving_target_positions = []

        for i, action in enumerate(actions):
            obs = env.step(action)
            trajectory.append(get_current_position(obs))
            total_actions += 1

            mark_visit(
                belief_map=belief_map,
                position=get_current_position(obs),
                grid_size=grid_size,
            )

            print(
                f"Goal {goal_id}, action {i}: "
                f"{action}, success={obs['last_action_success']}"
            )

            env.save_rgb(
                f"{save_dir}/goal_{goal_id:03d}_action_{i:03d}_{action}.png"
            )

            if not obs["last_action_success"]:
                print("Action failed:", obs["error_message"])
                action_failed = True
                break

            found, target_det, target_position, status, detections = observe_yolo_once(
                env=env,
                obs=obs,
                detector=detector,
                target_query=target_query,
                target_prompt=target_prompt,
                context_weights=context_weights,
                observed_contexts=observed_contexts,
                observed_keys=observed_keys,
                belief_map=belief_map,
                reachable_positions=reachable_positions,
                grid_size=grid_size,
                save_path=None,
                debug_path=f"{save_dir}/goal_{goal_id:03d}_action_{i:03d}_yolo.png",
                image_width=width,
                image_height=height,
                field_of_view=field_of_view,
            )

            print("Move-time YOLO target status:", status)

            if found and target_position is not None:
                moving_target_positions.append(target_position)

                if len(moving_target_positions) >= 2:
                    stable_target_position = median_position(moving_target_positions)

                    print(f"\nYOLO found stable target while moving: {target_query}")
                    print("Stable target position:", stable_target_position)

                    obs, approach_success, n = approach_target(
                        env=env,
                        obs=obs,
                        detector=detector,
                        target_query=target_query,
                        target_prompt=target_prompt,
                        target_position=stable_target_position,
                        reachable_positions=reachable_positions,
                        grid_size=grid_size,
                        save_dir=save_dir,
                        trajectory=trajectory,
                        image_width=width,
                        image_height=height,
                        field_of_view=field_of_view,
                    )

                    total_actions += n
                    success = approach_success
                    break

        if success:
            break

        if action_failed:
            goal_id += 1
            continue

        current_node = position_to_node(get_current_position(obs), grid_size)
        visited.add(current_node)

        found, obs, target_position, rotate_actions = observe_by_rotating_yolo(
            env=env,
            obs=obs,
            detector=detector,
            target_query=target_query,
            target_prompt=target_prompt,
            context_weights=context_weights,
            step_id=goal_id + 1,
            save_dir=save_dir,
            observed_contexts=observed_contexts,
            observed_keys=observed_keys,
            belief_map=belief_map,
            reachable_positions=reachable_positions,
            grid_size=grid_size,
            image_width=width,
            image_height=height,
            field_of_view=field_of_view,
        )

        for _ in range(rotate_actions):
            trajectory.append(get_current_position(obs))

        total_actions += rotate_actions

        if found and target_position is not None:
            print(f"\nYOLO found stable target after observing: {target_query}")
            print("Stable target position:", target_position)

            obs, approach_success, n = approach_target(
                env=env,
                obs=obs,
                detector=detector,
                target_query=target_query,
                target_prompt=target_prompt,
                target_position=target_position,
                reachable_positions=reachable_positions,
                grid_size=grid_size,
                save_dir=save_dir,
                trajectory=trajectory,
                image_width=width,
                image_height=height,
                field_of_view=field_of_view,
            )

            total_actions += n
            success = approach_success
            break

        goal_id += 1

    result = {
        "scene": scene,
        "target": target_query,
        "resolved_target": target_prompt,
        "success": bool(success),
        "visited_belief_goals": int(goal_id),
        "visited_reachable_points": int(len(visited)),
        "total_actions": int(total_actions),
        "num_context_objects": int(len(observed_contexts)),
        "anchor_searches": int(anchor_searches),
        "searched_anchors": int(len(searched_anchors)),
        "save_dir": save_dir,
        "log_path": log_path,
    }

    print("\n========== Summary ==========")
    for k, v in result.items():
        print(f"{k}: {v}")

    env.save_rgb(f"{save_dir}/final.png")
    save_trajectory_csv(trajectory, f"{save_dir}/trajectory.csv")

    if args.result_json is not None:
        result_dir = os.path.dirname(args.result_json)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)

        with open(args.result_json, "w") as f:
            json.dump(result, f, indent=2)

    env.close()

    restore_print(log_file)

    REAL_PRINT(
        f"[DONE] scene={scene}, target={target_query}, "
        f"success={success}, actions={total_actions}, "
        f"goals={goal_id}, anchors={anchor_searches}, "
        f"contexts={len(observed_contexts)}"
    )
    REAL_PRINT(f"       save_dir={save_dir}")
    REAL_PRINT(f"       log={log_path}")


if __name__ == "__main__":
    main()
