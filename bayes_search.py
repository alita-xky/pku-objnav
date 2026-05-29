import math
import numpy as np


CONTEXT_PRIOR = {
    "RemoteControl": {
        "Sofa": 1.0,
        "Television": 0.9,
        "TVStand": 0.9,
        "CoffeeTable": 0.8,
        "SideTable": 0.5,
        "ArmChair": 0.5,
    },
    "Book": {
        "Shelf": 1.0,
        "Desk": 0.9,
        "CoffeeTable": 0.6,
        "Sofa": 0.4,
        "SideTable": 0.4,
    },
    "Laptop": {
        "Desk": 1.0,
        "DiningTable": 0.7,
        "CoffeeTable": 0.6,
        "SideTable": 0.4,
    },
    "Mug": {
        "CoffeeTable": 1.0,
        "DiningTable": 0.9,
        "Desk": 0.7,
        "SideTable": 0.5,
    },
    "Sofa": {
        "CoffeeTable": 0.8,
        "Television": 0.6,
        "TVStand": 0.6,
        "Pillow": 0.7,
    }
}


def dist_xz(a, b):
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["z"] - b["z"]) ** 2)


def normalize_belief(belief):
    s = float(np.sum(belief))

    if s <= 1e-8:
        return np.ones_like(belief) / len(belief)

    return belief / s


def init_belief(num_positions):
    return np.ones(num_positions, dtype=np.float32) / num_positions


def angle_wrap(deg):
    while deg > 180:
        deg -= 360
    while deg < -180:
        deg += 360
    return deg


def positions_in_view(
    reachable_positions,
    agent_pose,
    max_distance=3.0,
    fov_degrees=90,
):
    """
    根据当前 agent 位姿，粗略判断哪些 reachable positions 已经被观察过。
    AI2-THOR 中 rotation.y = 0 时通常朝 +z 方向。
    """
    agent_pos = agent_pose["position"]
    yaw = agent_pose["rotation"]["y"]

    observed_ids = set()

    for i, pos in enumerate(reachable_positions):
        dx = pos["x"] - agent_pos["x"]
        dz = pos["z"] - agent_pos["z"]
        d = math.sqrt(dx * dx + dz * dz)

        if d > max_distance:
            continue

        # yaw=0 朝 +z，所以用 atan2(dx, dz)
        angle_to_pos = math.degrees(math.atan2(dx, dz))
        rel_angle = angle_wrap(angle_to_pos - yaw)

        if abs(rel_angle) <= fov_degrees / 2:
            observed_ids.add(i)

    return observed_ids


def context_score_for_position(pos, known_context_objects, target_type):
    """
    计算某个候选点附近是否存在和目标相关的上下文物体。
    """
    priors = CONTEXT_PRIOR.get(target_type, {})
    score = 0.0

    for obj in known_context_objects.values():
        obj_type = obj["object_type"]

        if obj_type not in priors:
            continue

        obj_pos = obj["position"]
        d = math.sqrt(
            (pos["x"] - obj_pos["x"]) ** 2
            + (pos["z"] - obj_pos["z"]) ** 2
        )

        # 越近越重要
        score += priors[obj_type] * math.exp(-d)

    return score


def update_known_context_objects(known_context_objects, visible_objects, target_type):
    """
    保存已经看见过的上下文物体。
    不保存目标本身，只保存 sofa/table/tvstand 这种线索物体。
    """
    for obj in visible_objects:
        obj_type = obj["object_type"]

        if obj_type == target_type:
            continue

        if obj_type not in CONTEXT_PRIOR.get(target_type, {}):
            continue

        object_id = obj.get("object_id", obj_type + str(obj["position"]))

        known_context_objects[object_id] = {
            "object_type": obj_type,
            "position": obj["position"],
        }

    return known_context_objects


def update_belief(
    belief,
    reachable_positions,
    newly_observed_ids,
    known_context_objects,
    target_type,
    miss_likelihood=0.25,
    context_strength=2.0,
):
    """
    贝叶斯更新：
    1. 如果某个区域刚刚看过但没看到目标，则该区域概率降低。
    2. 如果某个区域靠近目标相关物体，则概率升高。
    """
    new_belief = belief.copy()

    for i, pos in enumerate(reachable_positions):
        likelihood = 1.0

        # 负观测：看过但没看到目标
        if i in newly_observed_ids:
            likelihood *= miss_likelihood

        # 语义上下文：靠近 Sofa / TVStand / CoffeeTable 等
        ctx = context_score_for_position(
            pos=pos,
            known_context_objects=known_context_objects,
            target_type=target_type,
        )

        likelihood *= 1.0 + context_strength * ctx

        new_belief[i] *= likelihood

    return normalize_belief(new_belief)


def select_next_goal(
    belief,
    reachable_positions,
    current_position,
    visited_ids,
    known_context_objects,
    target_type,
):
    """
    选择下一个搜索点。
    """
    best_id = None
    best_score = -1e9

    alpha = 8.0     # belief 权重
    beta = 1.0      # 未访问奖励
    gamma = 3.0     # 上下文奖励
    lamb = 0.6      # 距离惩罚

    for i, pos in enumerate(reachable_positions):
        if i in visited_ids:
            continue

        p_target = belief[i]
        ctx = context_score_for_position(
            pos=pos,
            known_context_objects=known_context_objects,
            target_type=target_type,
        )

        info_gain = 1.0
        path_cost = dist_xz(current_position, pos)

        score = (
            alpha * p_target
            + beta * info_gain
            + gamma * ctx
            - lamb * path_cost
        )

        if score > best_score:
            best_score = score
            best_id = i

    return best_id, best_score
