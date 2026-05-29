import os
import numpy as np
from PIL import Image
from ai2thor.controller import Controller


def save_rgb(rgb, path):
    Image.fromarray(rgb).save(path)


def save_depth(depth, path):
    if depth is None:
        print("Depth is None, skip saving depth image.")
        return

    depth = np.asarray(depth)
    depth_norm = depth - depth.min()

    if depth_norm.max() > 0:
        depth_norm = depth_norm / depth_norm.max()

    depth_img = (depth_norm * 255).astype(np.uint8)
    Image.fromarray(depth_img).save(path)


def main():
    os.makedirs("outputs", exist_ok=True)

    print("Starting AI2-THOR controller...")

    controller = Controller(
        scene="FloorPlan201",
        width=640,
        height=480,
        gridSize=0.25,
        rotateStepDegrees=90,
        visibilityDistance=1.5,
        renderDepthImage=True,
    )

    print("Controller started.")

    event = controller.step(action="RotateRight")

    print("\n=== Basic Info ===")
    print("Action success:", event.metadata["lastActionSuccess"])
    print("Scene:", event.metadata["sceneName"])

    print("\n=== Agent Pose ===")
    print(event.metadata["agent"])

    rgb = event.frame
    depth = event.depth_frame

    print("\n=== Observation Shape ===")
    print("RGB shape:", rgb.shape)
    print("Depth shape:", None if depth is None else depth.shape)

    visible_objects = sorted(
        set(
            obj["objectType"]
            for obj in event.metadata["objects"]
            if obj.get("visible", False)
        )
    )

    print("\n=== Visible Objects ===")
    print(visible_objects)

    save_rgb(rgb, "outputs/rgb_test.png")
    save_depth(depth, "outputs/depth_test.png")

    print("\nSaved:")
    print("outputs/rgb_test.png")
    print("outputs/depth_test.png")

    controller.stop()


if __name__ == "__main__":
    main()
