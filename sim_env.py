import os
import cv2
from typing import Dict, Any, List

from ai2thor.controller import Controller

try:
    from ai2thor.platform import CloudRendering
    HAS_CLOUD_RENDERING = True
except Exception:
    CloudRendering = None
    HAS_CLOUD_RENDERING = False


class AI2ThorObjNavEnv:
    def __init__(
        self,
        scene: str = "FloorPlan201",
        width: int = 640,
        height: int = 480,
        grid_size: float = 0.25,
        rotate_step_degrees: int = 90,
        visibility_distance: float = 1.5,
        field_of_view: int = 60,
        headless: bool = True,
    ):
        self.scene = scene
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.rotate_step_degrees = rotate_step_degrees
        self.visibility_distance = visibility_distance
        self.field_of_view = field_of_view
        self.headless = headless

        kwargs = dict(
            scene=scene,
            width=width,
            height=height,
            gridSize=grid_size,
            rotateStepDegrees=rotate_step_degrees,
            visibilityDistance=visibility_distance,
            fieldOfView=field_of_view,
            renderDepthImage=True,
            renderInstanceSegmentation=True,
        )

        if headless and HAS_CLOUD_RENDERING:
            try:
                self.controller = Controller(platform=CloudRendering, **kwargs)
            except Exception as e:
                print("CloudRendering failed, trying normal rendering.")
                print("Reason:", repr(e))
                self.controller = Controller(**kwargs)
        else:
            self.controller = Controller(**kwargs)

        self.last_event = self.controller.last_event

    def reset(self, scene: str = None) -> Dict[str, Any]:
        if scene is not None:
            self.scene = scene
            self.last_event = self.controller.reset(scene=scene)
        else:
            self.last_event = self.controller.reset(scene=self.scene)

        return self._make_obs(self.last_event)

    def step(self, action: str) -> Dict[str, Any]:
        self.last_event = self.controller.step(action=action)
        return self._make_obs(self.last_event)

    def _make_obs(self, event) -> Dict[str, Any]:
        agent = event.metadata["agent"]

        return {
            "rgb": event.frame,
            "depth": event.depth_frame,
            "pose": {
                "position": agent["position"],
                "rotation": agent["rotation"],
                "camera_horizon": agent["cameraHorizon"],
            },
            "last_action_success": event.metadata["lastActionSuccess"],
            "error_message": event.metadata.get("errorMessage", ""),
        }

    def get_reachable_positions(self) -> List[Dict[str, float]]:
        event = self.controller.step(action="GetReachablePositions")
        return event.metadata["actionReturn"]

    def save_rgb(self, path: str):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        rgb = self.last_event.frame
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        self.controller.stop()
