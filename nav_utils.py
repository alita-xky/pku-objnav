import heapq
from typing import Dict, List, Tuple, Optional


def position_to_node(position: Dict[str, float], grid_size: float = 0.25) -> Tuple[int, int]:
    return (
        int(round(position["x"] / grid_size)),
        int(round(position["z"] / grid_size)),
    )


def normalize_yaw(yaw: float) -> int:
    yaw = int(round(yaw / 90.0)) * 90
    return yaw % 360


def pose_to_state(pose, grid_size: float = 0.25):
    position = pose["position"]
    rotation = pose["rotation"]
    node = position_to_node(position, grid_size)
    yaw = normalize_yaw(rotation["y"])
    return node, yaw


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def build_graph(reachable_positions, grid_size: float = 0.25):
    return set(position_to_node(p, grid_size) for p in reachable_positions)


def astar(nodes, start, goal):
    if start not in nodes:
        return None

    if goal not in nodes:
        return None

    frontier = []
    heapq.heappush(frontier, (0, start))

    came_from = {start: None}
    cost_so_far = {start: 0}

    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            break

        for dx, dz in directions:
            nxt = (current[0] + dx, current[1] + dz)

            if nxt not in nodes:
                continue

            new_cost = cost_so_far[current] + 1

            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + heuristic(nxt, goal)
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    if goal not in came_from:
        return None

    path = []
    cur = goal

    while cur is not None:
        path.append(cur)
        cur = came_from[cur]

    path.reverse()
    return path


def desired_yaw_from_delta(dx: int, dz: int) -> int:
    if dx == 1 and dz == 0:
        return 90
    if dx == -1 and dz == 0:
        return 270
    if dx == 0 and dz == 1:
        return 0
    if dx == 0 and dz == -1:
        return 180

    raise ValueError(f"Invalid move delta: {(dx, dz)}")


def rotation_actions(current_yaw: int, desired_yaw: int) -> List[str]:
    current_yaw = normalize_yaw(current_yaw)
    desired_yaw = normalize_yaw(desired_yaw)

    diff = (desired_yaw - current_yaw) % 360

    if diff == 0:
        return []
    if diff == 90:
        return ["RotateRight"]
    if diff == 180:
        return ["RotateRight", "RotateRight"]
    if diff == 270:
        return ["RotateLeft"]

    return []


def plan_actions_to_position(
    reachable_positions,
    start_pose,
    goal_position,
    grid_size: float = 0.25,
) -> Optional[List[str]]:
    nodes = build_graph(reachable_positions, grid_size)

    start_node, current_yaw = pose_to_state(start_pose, grid_size)
    goal_node = position_to_node(goal_position, grid_size)

    path = astar(nodes, start_node, goal_node)

    if path is None:
        return None

    actions = []

    for i in range(len(path) - 1):
        cur = path[i]
        nxt = path[i + 1]

        dx = nxt[0] - cur[0]
        dz = nxt[1] - cur[1]

        desired_yaw = desired_yaw_from_delta(dx, dz)
        rots = rotation_actions(current_yaw, desired_yaw)

        actions.extend(rots)
        actions.append("MoveAhead")

        current_yaw = desired_yaw

    return actions


def yaw_to_face_position(current_position, target_position) -> int:
    dx = target_position["x"] - current_position["x"]
    dz = target_position["z"] - current_position["z"]

    if abs(dx) > abs(dz):
        return 90 if dx > 0 else 270

    return 0 if dz > 0 else 180
