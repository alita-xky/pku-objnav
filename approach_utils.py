import os
import math
from typing import Dict, List, Tuple, Any

from nav_utils import plan_actions_to_position


def distance_xz(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["z"] - b["z"]) ** 2)


def select_reachable_near_object(
    reachable_positions: List[Dict[str, float]],
    object_position: Dict[str, float],
    preferred_distance: float = 0.75,
    max_distance: float = 1.5,
):
    """
    从所有 reachable positions 里选一个适合接近目标物的位置。

    preferred_distance:
        希望机器人离目标大约多远，比如 0.75 米。

    max_distance:
        优先考虑目标 1.5 米以内的可达点。

    如果没有合适点，就选离目标最近的可达点。
    """

    candidates = []

    for p in reachable_positions:
        d = distance_xz(p, object_position)

        if d <= max_distance:
            candidates.append((p, d))

    if len(candidates) > 0:
        # 选择距离最接近 preferred_distance 的点
        best_p, best_d = min(
            candidates,
            key=lambda item: abs(item[1] - preferred_distance)
        )
        return best_p, best_d

    # 如果目标附近 1.5m 内没有可达点，就退化成最近点
    best_p = None
    best_d = float("inf")

    for p in reachable_positions:
        d = distance_xz(p, object_position)

        if d < best_d:
            best_d = d
            best_p = p

    return best_p, best_d


def approach_detected_object(
    env,
    obs,
    target_det: Dict[str, Any],
    reachable_positions: List[Dict[str, float]],
    save_dir: str,
    tag: str,
    grid_size: float = 0.25,
    max_actions: int = 100,
):
    """
    目标已经被检测到后，规划并走到目标附近。

    返回：
        obs: 最后的 observation
        success: 是否成功执行路径
        info: 过程信息
    """

    os.makedirs(save_dir, exist_ok=True)

    object_position = target_det.get("position")

    if object_position is None:
        return obs, False, {
            "reason": "target_det has no position",
            "target_det": target_det,
        }

    goal_position, goal_dist_to_obj = select_reachable_near_object(
        reachable_positions=reachable_positions,
        object_position=object_position,
        preferred_distance=0.75,
        max_distance=1.5,
    )

    if goal_position is None:
        return obs, False, {
            "reason": "no reachable position near target",
            "object_position": object_position,
        }

    print("\n=== Approach Target ===")
    print("Target label:", target_det["label"])
    print("Target position:", object_position)
    print("Selected reachable goal:", goal_position)
    print("Goal distance to object:", round(goal_dist_to_obj, 3))

    actions = plan_actions_to_position(
        reachable_positions=reachable_positions,
        start_pose=obs["pose"],
        goal_position=goal_position,
        grid_size=grid_size,
    )

    if actions is None:
        return obs, False, {
            "reason": "no path to approach goal",
            "goal_position": goal_position,
        }

    print("Approach actions:", actions)
    print("Number of approach actions:", len(actions))

    if len(actions) > max_actions:
        return obs, False, {
            "reason": "approach path too long",
            "num_actions": len(actions),
        }

    env.save_rgb(f"{save_dir}/{tag}_approach_start.png")

    for i, action in enumerate(actions):
        obs = env.step(action)

        print(
            f"Approach action {i}: "
            f"{action}, success={obs['last_action_success']}"
        )

        env.save_rgb(f"{save_dir}/{tag}_approach_{i:03d}_{action}.png")

        if not obs["last_action_success"]:
            return obs, False, {
                "reason": "action failed",
                "failed_action": action,
                "error_message": obs["error_message"],
            }

    final_position = obs["pose"]["position"]
    final_dist = distance_xz(final_position, object_position)

    env.save_rgb(f"{save_dir}/{tag}_approach_final.png")

    print("\n=== Approach Finished ===")
    print("Final agent position:", final_position)
    print("Final distance to object:", round(final_dist, 3))

    return obs, True, {
        "object_position": object_position,
        "approach_goal": goal_position,
        "final_distance": final_dist,
        "num_actions": len(actions),
    }
