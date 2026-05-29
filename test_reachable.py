from sim_env import AI2ThorObjNavEnv


def main():
    env = AI2ThorObjNavEnv(scene="FloorPlan201")
    obs = env.reset()

    positions = env.get_reachable_positions()

    print("Number of reachable positions:", len(positions))
    print("First 10 reachable positions:")

    for p in positions[:10]:
        print(p)

    env.close()


if __name__ == "__main__":
    main()
