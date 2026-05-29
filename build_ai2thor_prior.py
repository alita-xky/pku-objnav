import os
import re
import json
import math
from collections import defaultdict
from typing import Dict, List, Tuple

from ai2thor.controller import Controller

try:
    from ai2thor.platform import CloudRendering
    HAS_CLOUD_RENDERING = True
except Exception:
    CloudRendering = None
    HAS_CLOUD_RENDERING = False


# ============================================================
# Name utils
# ============================================================

def split_camel_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()


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


def distance_xz(a: Dict[str, float], b: Dict[str, float]) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["z"] - b["z"]) ** 2)


# ============================================================
# Scene list
# ============================================================

def get_ai2thor_scenes() -> List[str]:
    """
    AI2-THOR 常用 120 个场景：
    FloorPlan1-30     kitchen
    FloorPlan201-230  living room
    FloorPlan301-330  bedroom
    FloorPlan401-430  bathroom
    """
    scenes = []

    scenes += [f"FloorPlan{i}" for i in range(1, 31)]
    scenes += [f"FloorPlan{i}" for i in range(201, 231)]
    scenes += [f"FloorPlan{i}" for i in range(301, 331)]
    scenes += [f"FloorPlan{i}" for i in range(401, 431)]

    return scenes


# ============================================================
# Controller
# ============================================================

def create_controller(scene: str):
    kwargs = dict(
        scene=scene,
        width=640,
        height=480,
        gridSize=0.25,
        rotateStepDegrees=90,
        visibilityDistance=1.5,
        fieldOfView=60,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
    )

    if HAS_CLOUD_RENDERING:
        try:
            return Controller(platform=CloudRendering, **kwargs)
        except Exception as e:
            print("CloudRendering failed, trying normal rendering.")
            print("Reason:", repr(e))

    return Controller(**kwargs)


# ============================================================
# Object extraction
# ============================================================

def get_scene_objects(event):
    """
    离线统计 prior 时使用 metadata。
    只取有 position 的物体。
    """
    objects = []

    for obj in event.metadata["objects"]:
        object_type = obj.get("objectType")
        position = obj.get("position")

        if object_type is None or position is None:
            continue

        prompt = ai2thor_to_prompt(object_type)

        objects.append({
            "object_type": object_type,
            "prompt": prompt,
            "position": position,
        })

    return objects


# ============================================================
# Prior statistics
# ============================================================

def build_prior_statistics(
    scenes: List[str],
    near_threshold: float = 2.0,
    output_dir: str = "outputs/prior_build",
):
    os.makedirs(output_dir, exist_ok=True)

    object_scene_count = defaultdict(int)
    pair_scene_count = defaultdict(int)
    pair_near_count = defaultdict(int)

    total_scenes = 0
    all_object_names = set()

    for scene_id, scene in enumerate(scenes):
        print(f"\n[{scene_id + 1}/{len(scenes)}] Processing {scene}")

        controller = None

        try:
            controller = create_controller(scene)
            event = controller.last_event

            objects = get_scene_objects(event)

            if len(objects) == 0:
                print("No objects found. Skip.")
                continue

            total_scenes += 1

            # 当前场景中出现过的类别
            scene_object_set = sorted(set(o["prompt"] for o in objects))

            for name in scene_object_set:
                object_scene_count[name] += 1
                all_object_names.add(name)

            # 同场景共现统计
            for i in range(len(scene_object_set)):
                for j in range(len(scene_object_set)):
                    if i == j:
                        continue

                    g = scene_object_set[i]
                    c = scene_object_set[j]
                    pair_scene_count[(g, c)] += 1

            # 空间邻近统计
            # 同一类可能有多个实例，所以逐实例判断
            near_pairs_in_scene = set()

            for i in range(len(objects)):
                for j in range(len(objects)):
                    if i == j:
                        continue

                    g = objects[i]["prompt"]
                    c = objects[j]["prompt"]

                    if g == c:
                        continue

                    d = distance_xz(objects[i]["position"], objects[j]["position"])

                    if d <= near_threshold:
                        near_pairs_in_scene.add((g, c))

            for pair in near_pairs_in_scene:
                pair_near_count[pair] += 1

            print("Number of object types:", len(scene_object_set))
            print("Object types:", scene_object_set[:20], "..." if len(scene_object_set) > 20 else "")

        except Exception as e:
            print(f"Failed scene {scene}: {repr(e)}")

        finally:
            if controller is not None:
                controller.stop()

    print("\nTotal valid scenes:", total_scenes)

    # 计算 prior
    prior = {}

    all_object_names = sorted(all_object_names)

    eps = 1e-9

    for g in all_object_names:
        contexts = []

        count_g = object_scene_count[g]

        if count_g == 0:
            continue

        p_g = count_g / max(total_scenes, 1)

        for c in all_object_names:
            if c == g:
                continue

            count_c = object_scene_count[c]
            count_gc = pair_scene_count[(g, c)]
            count_near = pair_near_count[(g, c)]

            if count_gc == 0:
                continue

            p_c = count_c / max(total_scenes, 1)
            p_gc = count_gc / max(total_scenes, 1)

            # P(c | g)
            cond_prob = count_gc / count_g

            # PMI
            pmi = math.log((p_gc + eps) / (p_g * p_c + eps))

            # near frequency: g 出现场景中，c 有多少比例在空间上接近过 g
            near_freq = count_near / count_g

            # score 先保留 raw，后面再归一化
            contexts.append({
                "context": c,
                "count_g": count_g,
                "count_c": count_c,
                "count_gc": count_gc,
                "count_near": count_near,
                "p_c_given_g": cond_prob,
                "pmi": pmi,
                "near_freq": near_freq,
            })

        if len(contexts) == 0:
            continue

        # 对当前 target 的 PMI 归一化到 0-1
        pmi_values = [x["pmi"] for x in contexts]
        min_pmi = min(pmi_values)
        max_pmi = max(pmi_values)

        for x in contexts:
            if max_pmi > min_pmi:
                pmi_norm = (x["pmi"] - min_pmi) / (max_pmi - min_pmi)
            else:
                pmi_norm = 0.0

            # 最终 prior score:
            # cond_prob 表示共现概率
            # pmi_norm 减少 floor/table 这种过于常见类别的偏置
            # near_freq 表示空间邻近性
            score = (
                0.35 * x["p_c_given_g"]
                + 0.35 * pmi_norm
                + 0.30 * x["near_freq"]
            )

            x["score"] = score

        contexts = sorted(contexts, key=lambda x: x["score"], reverse=True)

        prior[g] = contexts

    output_path = os.path.join(output_dir, "semantic_prior.json")

    with open(output_path, "w") as f:
        json.dump(
            {
                "total_scenes": total_scenes,
                "near_threshold": near_threshold,
                "objects": all_object_names,
                "prior": prior,
            },
            f,
            indent=2,
        )

    print("\nSaved semantic prior to:", output_path)

    # 打印几个例子
    examples = [
        "remote control",
        "pencil",
        "book",
        "mug",
        "laptop",
        "sofa",
        "tv",
    ]

    print("\n========== Example Priors ==========")

    for g in examples:
        if g not in prior:
            print(f"\nTarget: {g} not found in prior.")
            continue

        print(f"\nTarget: {g}")

        for item in prior[g][:10]:
            print(
                f"  {item['context']:20s} "
                f"score={item['score']:.3f} "
                f"P(c|g)={item['p_c_given_g']:.3f} "
                f"PMI={item['pmi']:.3f} "
                f"near={item['near_freq']:.3f}"
            )

    return output_path


def main():
    scenes = get_ai2thor_scenes()

    output_path = build_prior_statistics(
        scenes=scenes,
        near_threshold=2.0,
        output_dir="outputs/prior_build",
    )

    print("\nDone.")
    print("Prior file:", output_path)


if __name__ == "__main__":
    main()
