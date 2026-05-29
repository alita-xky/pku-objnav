import os
from PIL import Image, ImageDraw

from sim_env import AI2ThorObjNavEnv
from metadata_detector import AI2ThorMetadataDetector
from detection_utils import target_detected


def draw_detections(rgb, detections, save_path, target=None):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    for det in detections:
        box = det.get("box")

        if box is None:
            continue

        x1, y1, x2, y2 = box
        label = det["label"]
        score = det["score"]

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text(
            (x1, max(0, y1 - 18)),
            f"{label} {score:.2f}",
            fill="red",
        )

    image.save(save_path)


def main():
    os.makedirs("outputs/metadata_detector", exist_ok=True)

    env = AI2ThorObjNavEnv(scene="FloorPlan201")
    detector = AI2ThorMetadataDetector()

    obs = env.reset()

    target = "Sofa"

    for view_id in range(4):
        print(f"\n========== View {view_id} ==========")

        detections = detector.detect(obs)

        print("Metadata detections:")
        for det in detections:
            print(
                det["label"],
                "distance=",
                round(det["distance"], 3) if det["distance"] is not None else None,
                "box=",
                det["box"],
            )

        found, target_det = target_detected(
            detections=detections,
            target_type=target,
            threshold=0.5,
        )

        if found:
            print(f"\nTarget found: {target}")
            print(target_det)
        else:
            print(f"\nTarget not found: {target}")

        draw_detections(
            rgb=obs["rgb"],
            detections=detections,
            save_path=f"outputs/metadata_detector/view_{view_id}.png",
        )

        obs = env.step("RotateRight")

    env.close()


if __name__ == "__main__":
    main()
