import os

from sim_env import AI2ThorObjNavEnv
from semantic_prior import SemanticPrior
from yolo_detector import YoloWorldDetector


def main():
    save_dir = "outputs/test_yolo_with_prior"
    os.makedirs(save_dir, exist_ok=True)

    scene = "FloorPlan201"

    target_query = "remote control"

    prior = SemanticPrior("outputs/prior_build/semantic_prior.json")

    matched = prior.resolve_query(target_query)
    if len(matched) > 0:
        target_prompt = matched[0][0]
    else:
        target_prompt = target_query

    context_items = prior.get_navigation_contexts_for_query(
        query=target_query,
        top_k_contexts=8,
        min_score=0.05,
    )

    context_prompts = [ctx for ctx, score in context_items]

    extra_classes = [
        "sofa",
        "couch",
        "chair",
        "arm chair",
        "table",
        "coffee table",
        "dining table",
        "side table",
        "desk",
        "tv",
        "television",
        "tv stand",
        "laptop",
        "book",
        "pillow",
        "lamp",
        "floor lamp",
        "desk lamp",
        "plant",
        "house plant",
        "box",
        "remote control",
        "pen",
        "pencil",
    ]

    yolo_classes = sorted(set([target_query, target_prompt] + context_prompts + extra_classes))

    print("\n=== Target Query ===")
    print(target_query)

    print("\n=== Resolved Target Prompt ===")
    print(target_prompt)

    print("\n=== Contexts From Semantic Prior ===")
    for ctx, score in context_items:
        print(f"{ctx:20s} score={score:.3f}")

    print("\n=== YOLO Classes ===")
    print(yolo_classes)

    env = AI2ThorObjNavEnv(
        scene=scene,
        width=640,
        height=480,
        grid_size=0.25,
        rotate_step_degrees=90,
        field_of_view=60,
        headless=True,
    )

    obs = env.reset()

    detector = YoloWorldDetector(
        classes=yolo_classes,
        model_name="yolov8s-world.pt",
        conf=0.12,
        device="cpu",
    )

    for r in range(4):
        detections = detector.detect(obs["rgb"])

        print(f"\n========== Rotation {r} ==========")
        print("YOLO detections:")

        for d in detections:
            print(
                f"  {d['label']:20s} "
                f"score={d['score']:.3f} "
                f"box={[round(x, 1) for x in d['box']]}"
            )

        raw_path = f"{save_dir}/rot_{r}_rgb.png"
        yolo_path = f"{save_dir}/rot_{r}_yolo.png"

        env.save_rgb(raw_path)
        detector.save_debug_image(
            rgb_image=obs["rgb"],
            detections=detections,
            path=yolo_path,
            target_prompt=target_prompt,
        )

        obs = env.step("RotateRight")

    print("\nSaved images to:", save_dir)
    env.close()


if __name__ == "__main__":
    main()
