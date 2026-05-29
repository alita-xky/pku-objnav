from typing import List, Dict, Any


class AI2ThorMetadataDetector:
    """
    使用 AI2-THOR 的 metadata 作为 oracle detector。

    它不是一个真实视觉模型，而是直接读取仿真器提供的可见物体信息。
    优点：不会把画框误识别成 TV。
    用途：先把 ObjNav 搜索算法跑通。
    """

    def __init__(self, min_distance: float = None):
        self.min_distance = min_distance

    def detect(self, obs) -> List[Dict[str, Any]]:
        detections = []

        boxes = obs.get("instance_detections2D", {})

        for obj in obs["visible_objects"]:
            object_id = obj["object_id"]
            object_type = obj["object_type"]
            position = obj.get("position")
            distance = obj.get("distance")

            if self.min_distance is not None and distance is not None:
                if distance > self.min_distance:
                    continue

            box = boxes.get(object_id, None)

            if box is not None:
                try:
                    box = box.tolist()
                except AttributeError:
                    box = list(box)

            detections.append(
                {
                    "label": object_type,
                    "score": 1.0,
                    "box": box,
                    "object_id": object_id,
                    "position": position,
                    "distance": distance,
                }
            )

        return detections
