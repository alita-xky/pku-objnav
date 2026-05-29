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
from approach_utils import approach_detected_object

from bayes_search import (
    init_belief,
    positions_in_view,
    update_known_context_objects,
    update_belief,
    select_next_goal,
)


def distance_xz(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["z"] - b["z"]) ** 2)


def get_current_position(obs) -> Dict[str, float]:
    pos = obs["pose"]["position"]
    return {
        "x": pos["x"],
        "y": pos["y"],
        "z": pos["z"],
    }


def get_visible_objects_from_obs(obs):
    """
    从 sim_env 返回的 obs 中整理当前可见物体。
    bayes_search.py 需要 object_type / position / object_id 这些字段。
    """
    visible_objects = []

    for obj in obs.get("visible_objects", []):
        visible_objects.append({
            "object_id": obj.get("object_id"),
            "object_type": obj.get("object_type"),
            "position": obj.get("position"),
            "distance": obj.get("distance", None),
        })

    return visible_objects


def check_target_with_detector(detector, obs, target):
    detections = detector.detect(obs)

    found, target_det = target_detected(
        detections=detections,
        target_type=target,
        threshold=0.5,
    )

    return found, target_det, detections


def update_bayes_state(
    belief,
    reachable_positions,
    observed_ids,
    known_context_objects,
    seen_visible_objects,
    seen_reachable_ids,
    target,
):
    """
    根据一次观察过程更新：
    1. known_context_objects：已经看到过的上下文物体
    2. observed_ids：已经看过的 reachable positions
    3. belief：目标在各 reachable position 附近的概率
    """
    before_context_keys = set(known_context_objects.keys())

    known_context_objects = update_known_context_objects(
        known_context_objects=known_context_objects,
        visible_objects=seen_visible_objects,
        target_type=target,
    )

    after_context_keys = set(known_context_objects.keys())
    context_changed = before_context_keys != after_context_keys

    newly_observed_ids = seen_reachable_ids - observed_ids
    observed_ids.update(newly_observed_ids)

    if len(newly_observed_ids) > 0 or context_changed:
        belief = update_belief(
            belief=belief,
            reachable_positions=reachable_positions,
            newly_observed_ids=newly_observed_ids,
            known_context_objects=known_context_objects,
            target_type=target,
        )

    return belief, observed_ids, known_context_objects, newly_observed_ids


def observe_by_rotating(
    env,
    detector,
    obs,
    target,
    step_id,
    save_dir,
    reachable_positions,
):
    """
    原地转一圈观察。

    返回：
    found:
        是否看到目标
    obs:
        最后的 observation
    target_det:
        目标检测结果
    rotate_actions:
        这次观察中实际执行了多少次 RotateRight
    seen_visible_objects:
        观察过程中看到过的所有 visible objects
    seen_reachable_ids:
        观察过程中视野覆盖到的 reachable position ids
    """
    rotate_actions = 0
    seen_visible_objects = []
    seen_reachable_ids = set()

    for r in range(4):
        # 记录当前朝向下看到的物体
        current_visible_objects = get_visible_objects_from_obs(obs)
        seen_visible_objects.extend(current_visible_objects)

        # 记录当前朝向下看到过的 reachable positions
        current_view_ids = positions_in_view(
            reachable_positions=reachable_positions,
            agent_pose=obs["pose"],
            max_distance=3.0,
            fov_degrees=90,
        )
        seen_reachable_ids.update(current_view_ids)

        # 用 detector 判断目标是否可见
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
            return (
                True,
                obs,
                target_det,
                rotate_actions,
                seen_visible_objects,
                seen_reachable_ids,
            )

        # 没找到则转向
        obs = env.step("RotateRight")
        rotate_actions += 1
        env.save_rgb(f"{save_dir}/observe_step_{step_id}_rot_{r}.png")

    return (
        False,
        obs,
        None,
        rotate_actions,
        seen_visible_objects,
        seen_reachable_ids,
    )


def main():
    save_dir = "outputs/bayes_search_metadata"
    os.makedirs(save_dir, exist_ok=True)

    scene = "FloorPlan201"

    # 建议先用 RemoteControl 测试贝叶斯搜索。
    # Sofa 太大，通常初始视角就能看到，体现不出搜索策略。
    target = "Television"
    # target = "Sofa"

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

    # ========== Bayesian search state ==========
    belief = init_belief(len(reachable_positions))
    visited_ids = set()
    observed_ids = set()
    known_context_objects = {}

    # 这个 visited_nodes 只是用于统计，不再用于高层选点
    visited_nodes = set()

    max_goals = 50
    total_actions = 0
    success = False

    env.save_rgb(f"{save_dir}/start.png")

    current_node = position_to_node(get_current_position(obs), grid_size)
    visited_nodes.add(current_node)

    # ========== 初始位置先转一圈观察 ==========
    print("\n========== Initial Observation ==========")

    (
        found,
        obs,
        target_det,
        rotate_actions,
        seen_visible_objects,
        seen_reachable_ids,
    ) = observe_by_rotating(
        env=env,
        detector=detector,
        obs=obs,
        target=target,
        step_id=0,
        save_dir=save_dir,
        reachable_positions=reachable_positions,
    )

    total_actions += rotate_actions

    belief, observed_ids, known_context_objects, newly_observed_ids = update_bayes_state(
        belief=belief,
        reachable_positions=reachable_positions,
        observed_ids=observed_ids,
        known_context_objects=known_context_objects,
        seen_visible_objects=seen_visible_objects,
        seen_reachable_ids=seen_reachable_ids,
        target=target,
    )

    print("Initial rotate actions:", rotate_actions)
    print("Initial newly observed reachable points:", len(newly_observed_ids))
    print("Initial known context objects:", [
        obj["object_type"] for obj in known_context_objects.values()
    ])

    if found:
        print("\nTarget found at initial area. Now approaching target...")

        obs, approach_success, approach_info = approach_detected_object(
            env=env,
            obs=obs,
            target_det=target_det,
            reachable_positions=reachable_positions,
            save_dir=save_dir,
            tag="initial",
            grid_size=grid_size,
        )

        total_actions += approach_info.get("num_actions", 0)

        print("Approach success:", approach_success)
        print("Approach info:", approach_info)

        success = approach_success

    # ========== Bayesian goal search loop ==========
    goal_count = 0

    while not success and goal_count < max_goals:
        current_position = get_current_position(obs)

        goal_id, goal_score = select_next_goal(
            belief=belief,
            reachable_positions=reachable_positions,
            current_position=current_position,
            visited_ids=visited_ids,
            known_context_objects=known_context_objects,
            target_type=target,
        )

        if goal_id is None:
            print("\nNo more candidate goals.")
            break

        goal_position = reachable_positions[goal_id]
        visited_ids.add(goal_id)

        goal_node = position_to_node(goal_position, grid_size)
        visited_nodes.add(goal_node)

        print(f"\n========== Goal {goal_count} ==========")
        print("Current position:", current_position)
        print("Selected goal id:", goal_id)
        print("Selected goal:", goal_position)
        print("Goal score:", round(float(goal_score), 6))
        print("Goal belief:", round(float(belief[goal_id]), 8))
        print("Known context objects:", [
            obj["object_type"] for obj in known_context_objects.values()
        ])

        actions = plan_actions_to_position(
            reachable_positions=reachable_positions,
            start_pose=obs["pose"],
            goal_position=goal_position,
            grid_size=grid_size,
        )

        if actions is None:
            print("No path found to this goal. Skip.")
            goal_count += 1
            continue

        print("Planned actions:", actions)
        print("Number of actions:", len(actions))

        action_failed = False

        for i, action in enumerate(actions):
            obs = env.step(action)
            total_actions += 1

            print(
                f"Goal {goal_count}, action {i}: "
                f"{action}, success={obs['last_action_success']}"
            )

            env.save_rgb(
                f"{save_dir}/goal_{goal_count:03d}_action_{i:03d}_{action}.png"
            )

            if not obs["last_action_success"]:
                print("Action failed:", obs["error_message"])
                action_failed = True
                break

            # 移动过程中也检查目标
            found, target_det, detections = check_target_with_detector(
                detector=detector,
                obs=obs,
                target=target,
            )

            # 移动过程中，也顺便更新 belief
            current_visible_objects = get_visible_objects_from_obs(obs)
            current_view_ids = positions_in_view(
                reachable_positions=reachable_positions,
                agent_pose=obs["pose"],
                max_distance=3.0,
                fov_degrees=90,
            )

            belief, observed_ids, known_context_objects, newly_observed_ids = update_bayes_state(
                belief=belief,
                reachable_positions=reachable_positions,
                observed_ids=observed_ids,
                known_context_objects=known_context_objects,
                seen_visible_objects=current_visible_objects,
                seen_reachable_ids=current_view_ids,
                target=target,
            )

            if found:
                print(f"\nTarget found while moving: {target}")
                print("Target detection:", target_det)
                env.save_rgb(f"{save_dir}/target_found_moving.png")

                obs, approach_success, approach_info = approach_detected_object(
                    env=env,
                    obs=obs,
                    target_det=target_det,
                    reachable_positions=reachable_positions,
                    save_dir=save_dir,
                    tag=f"moving_goal_{goal_count}",
                    grid_size=grid_size,
                )

                total_actions += approach_info.get("num_actions", 0)

                print("Approach success:", approach_success)
                print("Approach info:", approach_info)

                success = approach_success
                break

        if success:
            break

        if action_failed:
            goal_count += 1
            continue

        current_node = position_to_node(get_current_position(obs), grid_size)
        visited_nodes.add(current_node)

        # 到达当前 goal 后，原地转一圈观察
        print(f"\n========== Observe After Goal {goal_count} ==========")

        (
            found,
            obs,
            target_det,
            rotate_actions,
            seen_visible_objects,
            seen_reachable_ids,
        ) = observe_by_rotating(
            env=env,
            detector=detector,
            obs=obs,
            target=target,
            step_id=goal_count + 1,
            save_dir=save_dir,
            reachable_positions=reachable_positions,
        )

        total_actions += rotate_actions

        belief, observed_ids, known_context_objects, newly_observed_ids = update_bayes_state(
            belief=belief,
            reachable_positions=reachable_positions,
            observed_ids=observed_ids,
            known_context_objects=known_context_objects,
            seen_visible_objects=seen_visible_objects,
            seen_reachable_ids=seen_reachable_ids,
            target=target,
        )

        print("Rotate actions:", rotate_actions)
        print("Newly observed reachable points:", len(newly_observed_ids))
        print("Total observed reachable points:", len(observed_ids))
        print("Known context objects:", [
            obj["object_type"] for obj in known_context_objects.values()
        ])

        if found:
            print(f"\nTarget found after observing: {target}")

            obs, approach_success, approach_info = approach_detected_object(
                env=env,
                obs=obs,
                target_det=target_det,
                reachable_positions=reachable_positions,
                save_dir=save_dir,
                tag=f"observe_goal_{goal_count}",
                grid_size=grid_size,
            )

            total_actions += approach_info.get("num_actions", 0)

            print("Approach success:", approach_success)
            print("Approach info:", approach_info)

            success = approach_success
            break

        goal_count += 1

    print("\n========== Summary ==========")
    print("Success:", success)
    print("Target:", target)
    print("Visited Bayesian goals:", len(visited_ids))
    print("Visited reachable nodes:", len(visited_nodes))
    print("Observed reachable points:", len(observed_ids))
    print("Known context objects:", [
        obj["object_type"] for obj in known_context_objects.values()
    ])
    print("Total actions:", total_actions)

    env.save_rgb(f"{save_dir}/final.png")
    env.close()


if __name__ == "__main__":
    main()
