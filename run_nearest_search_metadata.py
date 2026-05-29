import os
import math
from typing import List, Dict, Tuple, Set

from sim_env import AI2ThorObjNavEnv
from nav_utils import (
    plan_actions_to_position,
    position_to_node,
)
from metadata_detector import AI2ThorMetadataDetector
from detection_utils import target_detected


def distance_xz(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["z"] - b["z"]) ** 2)


def get_current_position(obs) -> Dict[str, float]:
    pos = obs["pose"]["position"]
    return {
        "x": pos["x"],
        "y": pos["y"],
        "z": pos["z"],
    }


def select_nearest_unvisited(
    current_position: Dict[str, float],
    reachable_positions: List[Dict[str, float]],
    visited: Set[Tuple[float, float]],
    grid_size: float = 0.25,
):
    best_pos = None
    best_dist = float("inf")

    for p in reachable_positions:
        node = position_to_node(p, grid_size)

        if node in visited:
            continue

        d = distance_xz(current_position, p)

        if d < best_dist:
            best_dist = d
            best_pos = p

    return best_pos, best_dist


def check_target_with_detector(detector, obs, target):
    detections = detector.detect(obs)

    found, target_det = target_detected(
        detections=detections,
        target_type=target,
        threshold=0.5,
    )

    return found, target_det, detections


def observe_by_rotating(env, detector, obs, target, step_id, save_dir):
    """
    到达一个点之后，原地转一圈观察。
    这里不再用 env.target_visible()，而是用 detector.detect(obs) 判断。
    """

    for r in range(4):
        found, target_det, detections = check_target_with_detector(
            detector=detector,
            obs=obs,
            target=target,
        )

        print(f"Observe rotation {r}:")
        print("Visible labels:", sorted(set(d["label"] for d in detections)))

        if found:
            print("Target detection:", target_det)
            env.save_rgb(f"{save_dir}/target_found_step_{step_id}_rot_{r}.png")
            return True, obs

        obs = env.step("RotateRight")
        env.save_rgb(f"{save_dir}/observe_step_{step_id}_rot_{r}.png")

    return False, obs


def main():
    save_dir = "outputs/nearest_search_metadata"
    os.makedirs(save_dir, exist_ok=True)

    scene = "FloorPlan201"
    target = "Sofa"
    grid_size = 0.25

    env = AI2ThorObjNavEnv(
        scene=scene,
        grid_size=grid_size,
        rotate_step_degrees=90,
    )

    detector = AI2ThorMetadataDetector()

    obs = env.reset()

    print("\n=== Scene ===")
    print(scene)

    print("\n=== Target ===")
    print(target)

    print("\n=== Initial Pose ===")
    print(obs["pose"])

    all_types = sorted(set(obj["object_type"] for obj in obs["all_objects"]))
    print("\n=== All Object Types in This Scene ===")
    print(all_types)

    reachable_positions = env.get_reachable_positions()
    print("\nNumber of reachable positions:", len(reachable_positions))

    visited = set()

    max_goals = 50
    total_actions = 0
    success = False

    env.save_rgb(f"{save_dir}/start.png")

    current_node = position_to_node(get_current_position(obs), grid_size)
    visited.add(current_node)

    # 初始位置先转一圈观察
    found, obs = observe_by_rotating(
        env=env,
        detector=detector,
        obs=obs,
        target=target,
        step_id=0,
        save_dir=save_dir,
    )

    if found:
        print("\nSuccess! Target visible at initial area.")
        success = True

    goal_id = 0

    while not success and goal_id < max_goals:
        current_position = get_current_position(obs)

        goal_position, dist = select_nearest_unvisited(
            current_position=current_position,
            reachable_positions=reachable_positions,
            visited=visited,
            grid_size=grid_size,
        )

        if goal_position is None:
            print("\nNo more unvisited reachable positions.")
            break

        goal_node = position_to_node(goal_position, grid_size)
        visited.add(goal_node)

        print(f"\n========== Goal {goal_id} ==========")
        print("Current position:", current_position)
        print("Selected goal:", goal_position)
        print("Distance:", round(dist, 3))

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

        for i, action in enumerate(actions):
            obs = env.step(action)
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

            found, target_det, detections = check_target_with_detector(
                detector=detector,
                obs=obs,
                target=target,
            )

            if found:
                print(f"\nSuccess! Target found while moving: {target}")
                print("Target detection:", target_det)
                env.save_rgb(f"{save_dir}/target_found_moving.png")
                success = True
                break

        if success:
            break

        if action_failed:
            goal_id += 1
            continue

        current_node = position_to_node(get_current_position(obs), grid_size)
        visited.add(current_node)

        found, obs = observe_by_rotating(
            env=env,
            detector=detector,
            obs=obs,
            target=target,
            step_id=goal_id + 1,
            save_dir=save_dir,
        )

        total_actions += 4

        if found:
            print(f"\nSuccess! Target found after observing: {target}")
            success = True
            break

        goal_id += 1

    print("\n========== Summary ==========")
    print("Success:", success)
    print("Target:", target)
    print("Visited reachable points:", len(visited))
    print("Total actions:", total_actions)

    env.save_rgb(f"{save_dir}/final.png")
    env.close()


if __name__ == "__main__":
    main()
