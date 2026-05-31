import os
import numpy as np
from PIL import Image
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

print("[1/5] Starting AI2-THOR with CloudRendering...", flush=True)
controller = Controller(
    platform=CloudRendering,
    scene="FloorPlan201",
    width=640,
    height=480,
    gridSize=0.25,
    rotateStepDegrees=90,
    visibilityDistance=1.5,
    renderDepthImage=True,
)
print("[2/5] Controller started OK", flush=True)

event = controller.step(action="RotateRight")
print("[3/5] Step OK. Scene:", event.metadata["sceneName"], flush=True)
print("      lastActionSuccess:", event.metadata["lastActionSuccess"], flush=True)
print("      RGB shape:", event.frame.shape, flush=True)
print("      Depth shape:", event.depth_frame.shape if event.depth_frame is not None else None, flush=True)

os.makedirs("outputs", exist_ok=True)
Image.fromarray(event.frame).save("outputs/cloud_rgb.png")
d = event.depth_frame
d_norm = (d - d.min()) / (d.max() - d.min() + 1e-9)
Image.fromarray((d_norm*255).astype(np.uint8)).save("outputs/cloud_depth.png")
print("[4/5] Saved RGB + depth", flush=True)

visible = sorted(set(o["objectType"] for o in event.metadata["objects"] if o.get("visible", False)))
print("[5/5] Visible objects:", visible, flush=True)
controller.stop()
print("Done. SUCCESS.", flush=True)
