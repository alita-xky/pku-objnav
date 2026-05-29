import os
import re
import csv
import math
from typing import List, Dict, Tuple, Set

import numpy as np

from sim_env import AI2ThorObjNavEnv
from nav_utils import (
    plan_actions_to_position,
    position_to_node,
    rotation_actions,
    normalize_yaw,
    yaw_to_face_position,
)
from yolo_detector import YoloWorldDetector


# ============================================================
# Basic utils
# ============================================================

def split_camel_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()


def norm_label(label: str) -> str:
    return label.lower().replace(" ", "").replace("_", "").replace("-", "")


def ai2thor_to_prompt(object_type: str) -> str:
    mapping = {
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

    if object_type in mapping:
        return mapping[object_type]

    return split_camel_case(object_type)


def label_match(a: str, b: str) -> bool:
    na = norm_label(a)
    nb = norm_label(b)

    if na == nb:
        return True

    alias_groups = [
        {"sofa", "couch"},
        {"tv", "television", "monitor", "screen"},
        {"remotecontrol", "remote", "remotecontroller", "remotecontrol"},
        {"coffeetable", "table"},
        {"diningtable", "table"},
        {"houseplant", "plant"},
        {"garbagecan", "trashcan", "bin"},
    ]

    for group in alias_groups:
        if na in group and nb in group:
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


# ============================================================
# Context priors
# ============================================================

CONTEXT_PRIOR = {
    "pencil": {
        "pen": 1.0,
        "dining table": 0.9,
        "desk": 0.9,
        "book": 0.5,
        "newspaper": 0.5,
        "laptop": 0.4,
    },
    "pen": {
        "pencil": 1.0,
        "dining table": 0.9,
        "desk": 0.9,
        "book": 0.5,
        "newspaper": 0.5,
    },
    "remotecontrol": {
        "sofa": 1.0,
        "couch": 1.0,
        "tv": 0.9,
        "television": 0.9,
        "coffee table": 0.8,
        "tv stand": 0.7,
    },
    "book": {
        "shelf": 1.0,
        "desk": 0.8,
        "dining table": 0.6,
        "coffee table": 0.6,
        "sofa": 0.4,
        "couch": 0.4,
    },
    "laptop": {
        "desk": 1.0,
        "dining table": 0.8,
        "chair": 0.5,
    },
    "mug": {
        "dining table": 1.0,
        "coffee table": 0.9,
        "side table": 0.7,
        "desk": 0.7,
    },
    "sofa": {
        "tv": 0.9,
        "television": 0.9,
        "coffee table": 0.8,
        "remote control": 0.7,
        "pillow": 0.6,
    },
    "tv": {
        "sofa": 0.9,
        "couch": 0.9,
        "coffee table": 0.7,
        "tv stand": 1.0,
    },
}


def get_context_prompts(target_prompt: str) -> List[str]:
    return list(CONTEXT_PRIOR.get(norm_label(target_prompt), {}).keys())


def get_context_weight(target_prompt: str, object_prompt: str) -> float:
    table = CONTEXT_PRIOR.get(norm_label(target_prompt), {})

    for k, v in table.items():
        if label_match(k, object_prompt):
            return v

    return 0.0


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
    """
    太小、太低分的框不算有效。
    小物体可适当放宽。
    """
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


def find_target_detection(detections, target_prompt, image_width=640, image_height=480):
    candidates = [
        d for d in detections
        if label_match(d["label"], target_prompt)
    ]

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
    """
    只用 YOLO box + depth + agent pose 估计物体世界坐标。
    不使用 AI2-THOR object metadata。

    稳定性改进：
    不取 box 中心一个点，而取 bbox 中心 60% 区域 median depth。
    """
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

    # AI2-THOR: yaw=0 朝 +z，yaw=90 朝 +x
    world_x = agent_pos["x"] + math.cos(yaw) * x_cam + math.sin(yaw) * z_cam
    world_z = agent_pos["z"] - math.sin(yaw) * x_cam + math.cos(yaw) * z_cam

    return {
        "x": world_x,
        "y": agent_pos["y"],
        "z": world_z,
        "depth": z_cam,
    }


# ============================================================
# Bayesian memory
# ============================================================

def add_or_update_context(
    observed_contexts,
    observed_keys,
    label,
    matched_context,
    score,
    position,
    merge_radius=0.6,
):
    """
    减少 YOLO-depth 抖动：
    如果同类 context 已经在附近出现过，就融合位置，不新增点。
    """
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
    target_prompt,
    observed_contexts,
    observed_keys,
    image_width=640,
    image_height=480,
    field_of_view=60,
):
    context_prompts = get_context_prompts(target_prompt)

    for det in detections:
        matched_context = None

        for ctx_prompt in context_prompts:
            if label_match(det["label"], ctx_prompt):
                matched_context = ctx_prompt
                break

        if matched_context is None:
            continue

        if not is_valid_detection(det, image_width, image_height):
            continue

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

        add_or_update_context(
            observed_contexts=observed_contexts,
            observed_keys=observed_keys,
            label=det["label"],
            matched_context=matched_context,
            score=det["score"],
            position=pos,
        )


def bayesian_context_probability(goal_position, target_prompt, observed_contexts):
    """
    p = 1 - Π(1 - evidence_i)
    """
    p_not = 1.0
    sigma = 1.5

    for ctx in observed_contexts:
        ctx_pos = ctx["position"]
        d = distance_xz(goal_position, ctx_pos)

        w = get_context_weight(target_prompt, ctx["matched_context"])

        if w <= 0:
            continue

        count_bonus = min(1.0, 0.5 + 0.15 * ctx.get("count", 1))
        evidence = w * ctx["score"] * count_bonus * math.exp(-d / sigma)
        evidence = max(0.0, min(0.95, evidence))

        p_not *= (1.0 - evidence)

    return 1.0 - p_not


def select_bayes_goal(
    current_position: Dict[str, float],
    reachable_positions: List[Dict[str, float]],
    visited: Set[Tuple[int, int]],
    observed_contexts,
    target_prompt,
    grid_size: float = 0.25,
):
    best_pos = None
    best_score = -1e9
    best_info = None

    for p in reachable_positions:
        node = position_to_node(p, grid_size)

        if node in visited:
            continue

        dist = distance_xz(current_position, p)

        belief = bayesian_context_probability(
            goal_position=p,
            target_prompt=target_prompt,
            observed_contexts=observed_contexts,
        )

        # 没有上下文时 belief=0，退化成 nearest search。
        score = 10.0 * belief - 0.25 * dist

        if score > best_score:
            best_score = score
            best_pos = p
            best_info = {
                "belief": belief,
                "distance": dist,
                "score": score,
            }

    return best_pos, best_info


# ============================================================
# YOLO observation
# ============================================================

def observe_yolo_once(
    env,
    obs,
    detector,
    target_prompt,
    observed_contexts,
    observed_keys,
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
        target_prompt=target_prompt,
        observed_contexts=observed_contexts,
        observed_keys=observed_keys,
        image_width=image_width,
        image_height=image_height,
        field_of_view=field_of_view,
    )

    found, target_det, status = find_target_detection(
        detections=detections,
        target_prompt=target_prompt,
        image_width=image_width,
        image_height=image_height,
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
    target_prompt,
    step_id,
    save_dir,
    observed_contexts,
    observed_keys,
    image_width=640,
    image_height=480,
    field_of_view=60,
):
    """
    原地转一圈。
    关键修改：
    不再单帧发现就立刻 approach。
    而是收集多个 target_position，最后取 median。
    """
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
            target_prompt=target_prompt,
            observed_contexts=observed_contexts,
            observed_keys=observed_keys,
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
# Approach target
# ============================================================

def select_approach_goal(
    current_position,
    target_position,
    reachable_positions,
    min_dist=0.65,
    max_dist=1.15,
):
    best = None
    best_score = float("inf")

    for p in reachable_positions:
        d_obj = distance_xz(p, target_position)

        if d_obj < min_dist or d_obj > max_dist:
            continue

        d_agent = distance_xz(current_position, p)

        score = abs(d_obj - 0.85) + 0.05 * d_agent

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


def approach_target(
    env,
    obs,
    detector,
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

    final_detections = detector.detect(obs["rgb"])
    detector.save_debug_image(
        rgb_image=obs["rgb"],
        detections=final_detections,
        path=f"{save_dir}/after_approach_yolo.png",
        target_prompt=target_prompt,
    )

    final_found, final_det, final_status = find_target_detection(
        final_detections,
        target_prompt,
        image_width=image_width,
        image_height=image_height,
    )

    print("\n=== Approach Finished ===")
    print("Final agent position:", final_position)
    print("Final distance to estimated target:", round(final_distance, 3))
    print("Final YOLO status:", final_status)
    print("Final YOLO target detection:", final_det)

    success = final_distance <= 1.15 and final_found

    return obs, success, num_actions


# ============================================================
# Main
# ============================================================

def main():
    save_dir = "outputs/yolo_only_bayes_approach_stable"
    os.makedirs(save_dir, exist_ok=True)

    scene = "FloorPlan201"

    # 调试阶段建议先用 Sofa / Television。
    # 小物体 Pencil / RemoteControl 后面再调。

    width = 640
    height = 480
    field_of_view = 60
    grid_size = 0.25

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
    from semantic_prior import SemanticPrior

    target_query = "remote"

    prior = SemanticPrior("outputs/prior_build/semantic_prior.json")

    context_items = prior.get_contexts_for_query(
        target_query,
        top_k_contexts=8,
    )

    context_prompts = [x[0] for x in context_items]

    yolo_classes = sorted(set([target_query] + context_prompts + extra_classes))


    print("\n=== Scene ===")
    print(scene)

    print("\n=== Target ===")
    print(target)
    print("Target prompt:", target_prompt)

    print("\n=== YOLO Classes ===")
    print(yolo_classes)

    print("\n=== Initial Pose ===")
    print(obs["pose"])

    reachable_positions = env.get_reachable_positions()
    print("\nNumber of reachable positions:", len(reachable_positions))

    detector = YoloWorldDetector(
        classes=yolo_classes,
        model_name="yolov8s-world.pt",
        conf=0.12,
        device="cpu",      # 有 GPU 可以改成 "cuda:0"
    )

    visited = set()
    observed_contexts = []
    observed_keys = set()

    trajectory = [get_current_position(obs)]

    max_goals = 50
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
        target_prompt=target_prompt,
        step_id=0,
        save_dir=save_dir,
        observed_contexts=observed_contexts,
        observed_keys=observed_keys,
        image_width=width,
        image_height=height,
        field_of_view=field_of_view,
    )

    for _ in range(rotate_actions):
        trajectory.append(get_current_position(obs))

    total_actions += rotate_actions

    print("Initial observed context objects:", observed_contexts)

    if found and target_position is not None:
        print("\nYOLO found stable target at initial area. Now approaching target...")

        obs, approach_success, n = approach_target(
            env=env,
            obs=obs,
            detector=detector,
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

        goal_position, goal_info = select_bayes_goal(
            current_position=current_position,
            reachable_positions=reachable_positions,
            visited=visited,
            observed_contexts=observed_contexts,
            target_prompt=target_prompt,
            grid_size=grid_size,
        )

        if goal_position is None:
            print("\nNo more unvisited reachable positions.")
            break

        goal_node = position_to_node(goal_position, grid_size)
        visited.add(goal_node)

        print(f"\n========== Bayesian Goal {goal_id} ==========")
        print("Current position:", current_position)
        print("Selected goal:", goal_position)
        print("Goal info:", goal_info)
        print("Observed context objects:", observed_contexts)

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
                target_prompt=target_prompt,
                observed_contexts=observed_contexts,
                observed_keys=observed_keys,
                save_path=None,
                debug_path=f"{save_dir}/goal_{goal_id:03d}_action_{i:03d}_yolo.png",
                image_width=width,
                image_height=height,
                field_of_view=field_of_view,
            )

            print("Move-time YOLO target status:", status)

            if found and target_position is not None:
                moving_target_positions.append(target_position)

                # 移动过程中需要至少 2 次稳定检测，减少误检导致乱 approach。
                if len(moving_target_positions) >= 2:
                    stable_target_position = median_position(moving_target_positions)

                    print(f"\nYOLO found stable target while moving: {target}")
                    print("Stable target position:", stable_target_position)

                    obs, approach_success, n = approach_target(
                        env=env,
                        obs=obs,
                        detector=detector,
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
            target_prompt=target_prompt,
            step_id=goal_id + 1,
            save_dir=save_dir,
            observed_contexts=observed_contexts,
            observed_keys=observed_keys,
            image_width=width,
            image_height=height,
            field_of_view=field_of_view,
        )

        for _ in range(rotate_actions):
            trajectory.append(get_current_position(obs))

        total_actions += rotate_actions

        if found and target_position is not None:
            print(f"\nYOLO found stable target after observing: {target}")
            print("Stable target position:", target_position)

            obs, approach_success, n = approach_target(
                env=env,
                obs=obs,
                detector=detector,
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

    print("\n========== Summary ==========")
    print("Success:", success)
    print("Target:", target)
    print("Visited Bayesian goals:", goal_id)
    print("Visited reachable points:", len(visited))
    print("Observed context objects:", observed_contexts)
    print("Total actions:", total_actions)

    env.save_rgb(f"{save_dir}/final.png")
    save_trajectory_csv(trajectory, f"{save_dir}/trajectory.csv")

    print(f"\nSaved trajectory to: {save_dir}/trajectory.csv")
    print(f"Saved images to: {save_dir}")

    env.close()


if __name__ == "__main__":
    main()
