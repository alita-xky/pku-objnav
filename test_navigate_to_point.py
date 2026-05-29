import os
import random

from sim_env import AI2ThorObjNavEnv
from nav_utils import plan_actions_to_position


def main():
    os.makedirs("outputs/nav_test", exist_ok=True)

    env = AI2ThorObjNavEnv(scene="FloorPlan201")
    obs = env.reset()

    reachable_positions = env.get_reachable_positions()

    print("Number of reachable positions:", len(reachable_positions))

    # 随机选一个可达位置作为目标点
    goal_position = random.choice(reachable_positions)

    print("\nStart pose:")
    print(obs["pose"])

    print("\nGoal position:")
    print(goal_position)

    actions = plan_actions_to_position(
        reachable_positions=reachable_positions,
        start_pose=obs["pose"],
        goal_position=goal_position,
        grid_size=0.25,
    )

    if actions is None:
        print("No path found.")
        env.close()
        return

    print("\nPlanned actions:")
    print(actions)
    print("Number of actions:", len(actions))

    env.save_rgb("outputs/nav_test/start.png")

    for i, action in enumerate(actions):
        obs = env.step(action)

        print(f"Step {i}: {action}, success={obs['last_action_success']}")

        env.save_rgb(f"outputs/nav_test/step_{i:03d}.png")

        if not obs["last_action_success"]:
            print("Action failed. Stop execution.")
            print("Error:", obs["error_message"])
            break

    env.save_rgb("outputs/nav_test/final.png")

    print("\nFinal pose:")
    print(obs["pose"])

    print("\nSaved images to outputs/nav_test/")

    env.close()


if __name__ == "__main__":
    main()
