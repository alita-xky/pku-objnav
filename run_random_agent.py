import os
import random

from sim_env import AI2ThorObjNavEnv


def print_visible_objects(obs):
    visible = sorted(
        set(obj["object_type"] for obj in obs["visible_objects"])
    )
    print("Visible objects:", visible)


def main():
    os.makedirs("outputs/random_agent", exist_ok=True)

    env = AI2ThorObjNavEnv(scene="FloorPlan201")

    obs = env.reset()

    print("\n=== Initial Pose ===")
    print(obs["pose"])

    print("\n=== All Object Types in This Scene ===")
    all_types = sorted(set(obj["object_type"] for obj in obs["all_objects"]))
    print(all_types)

    target = "Television"

    actions = [
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
    ]

    max_steps = 30

    for step in range(max_steps):
        print(f"\n========== Step {step} ==========")

        print_visible_objects(obs)

        if env.target_visible(target):
            print(f"\nSuccess! Target found: {target}")
            env.save_rgb("outputs/random_agent/target_found_rgb.png")
            env.save_depth("outputs/random_agent/target_found_depth.png")
            break

        action = random.choice(actions)
        print("Action:", action)

        obs = env.step(action)

        print("Action success:", obs["last_action_success"])

        env.save_rgb(f"outputs/random_agent/frame_{step:03d}.png")

    env.close()


if __name__ == "__main__":
    main()
