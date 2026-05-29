import os
from PIL import Image, ImageDraw

from sim_env import AI2ThorObjNavEnv
from detector import YoloDetector


def draw_detections(rgb, detections, save_path):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
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
    os.makedirs("outputs/yolo_test", exist_ok=True)

    env = AI2ThorObjNavEnv(scene="FloorPlan201")
    obs = env.reset()

    # 原地转一下，换个视角
    obs = env.step("RotateRight")

    rgb = obs["rgb"]
    print("rgb type:", type(rgb))
    print("rgb shape:", getattr(rgb, "shape", None))
    print("rgb dtype:", getattr(rgb, "dtype", None))


    detector = YoloDetector(
        model_name="yolo11n.pt",
        conf=0.25,
    )

    print("\nRunning YOLO on current AI2-THOR frame...")

    detections = detector.detect(rgb)

    print("\n=== YOLO Detections ===")
    if len(detections) == 0:
        print("No objects detected.")
    else:
        for det in detections:
            print(
                det["label"],
                round(det["score"], 3),
                det["box"],
            )

    env.save_rgb("outputs/yolo_test/raw_rgb.png")

    draw_detections(
        rgb=rgb,
        detections=detections,
        save_path="outputs/yolo_test/yolo_result.png",
    )

    print("\nSaved:")
    print("outputs/yolo_test/raw_rgb.png")
    print("outputs/yolo_test/yolo_result.png")

    env.close()


if __name__ == "__main__":
    main()
