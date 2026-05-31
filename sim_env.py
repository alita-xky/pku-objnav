import os
import time
import collections
import cv2
from typing import Dict, Any, List

from ai2thor.controller import Controller

try:
    from ai2thor.platform import CloudRendering
    HAS_CLOUD_RENDERING = True
except Exception:
    CloudRendering = None
    HAS_CLOUD_RENDERING = False


class ControllerDead(RuntimeError):
    """Raised when the underlying AI2-THOR / Unity process appears dead
    (silent failures: every step returns lastActionSuccess=False instantly).

    Observed on FloorPlan203 during the 05-31 batch: after one episode hangs
    for ~60 s the Unity subprocess goes into a zombie state where each
    controller.step() returns in <5 ms with empty metadata.  Without this
    detector the agent loop "finishes" 200 steps in 8 s and the outer
    wall-clock timeout (180 s) never fires, so no recreate happens.
    """


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

        # ---- liveness telemetry ----
        # Tracks recent controller.step() wall-times and a streak counter of
        # consecutive lastActionSuccess=False results.  step() raises
        # ControllerDead when both signals indicate the Unity backend has
        # gone silent.
        self._step_times: collections.deque = collections.deque(maxlen=20)
        self._failed_streak: int = 0
        self._dead: bool = False

        # Thresholds (chosen against FP203 evidence: ~40 ms / step on a dead
        # controller vs 30-100 ms healthy; 15 consecutive lastActionSuccess
        # =False is well above any plausible "bumped into wall" run).
        self._dead_failed_streak_threshold: int = 15
        self._dead_step_time_threshold_s: float = 0.005   # 5 ms

    def reset(self, scene: str = None) -> Dict[str, Any]:
        if scene is not None:
            self.scene = scene
            self.last_event = self.controller.reset(scene=scene)
        else:
            self.last_event = self.controller.reset(scene=self.scene)

        # A successful reset means a fresh scene state — clear failure
        # telemetry so a healthy episode is not contaminated by the previous
        # one's stuck counter.
        self._step_times.clear()
        self._failed_streak = 0

        return self._make_obs(self.last_event)

    def step(self, action: str) -> Dict[str, Any]:
        if self._dead:
            raise ControllerDead(
                "controller previously marked dead; will not issue more steps"
            )

        t0 = time.time()
        self.last_event = self.controller.step(action=action)
        dt = time.time() - t0
        self._step_times.append(dt)

        success = self.last_event.metadata.get("lastActionSuccess", False)
        if success:
            self._failed_streak = 0
        else:
            self._failed_streak += 1

        # Liveness heuristic: when the Unity backend has crashed silently
        # every call returns "fail" in ~5 ms.  In normal operation steps
        # take 30-100 ms (CloudRendering) and bumps into walls are rare.
        if (
            self._failed_streak >= self._dead_failed_streak_threshold
            and len(self._step_times) >= 10
        ):
            avg_dt = sum(self._step_times) / len(self._step_times)
            if avg_dt < self._dead_step_time_threshold_s:
                self._dead = True
                raise ControllerDead(
                    f"controller dead: {self._failed_streak} consecutive "
                    f"failed steps at avg {avg_dt*1000:.1f} ms/step"
                )

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

    def is_alive(self) -> bool:
        """Cheap pre-episode health check.

        Returns False if the Unity backend has been previously marked dead by
        the in-step detector, or if a `GetReachablePositions` query fails /
        returns implausibly empty data.  Idempotent — safe to call between
        episodes without touching scene state.
        """
        if self._dead:
            return False
        try:
            ev = self.controller.step(action="GetReachablePositions")
        except Exception:
            self._dead = True
            return False
        if not ev.metadata.get("lastActionSuccess", False):
            self._dead = True
            return False
        ret = ev.metadata.get("actionReturn") or []
        # Any normal FloorPlan has hundreds of reachable cells; 5 is a wide
        # safety margin against false negatives.
        if len(ret) < 5:
            self._dead = True
            return False
        objs = ev.metadata.get("objects") or []
        if len(objs) < 5:
            self._dead = True
            return False
        return True

    def save_rgb(self, path: str):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        rgb = self.last_event.frame
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def close(self):
        try:
            self.controller.stop()
        except Exception:
            pass
        self._dead = True
