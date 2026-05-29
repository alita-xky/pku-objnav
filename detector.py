import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from typing import List, Dict, Any

import numpy as np
import torch
from ultralytics import YOLO


def get_default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class YoloDetector:
    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        conf: float = 0.25,
        device: str = None,
    ):
        self.model = YOLO(model_name)
        self.conf = conf
        self.device = device if device is not None else get_default_device()

        print(f"YOLO model: {model_name}")
        print(f"YOLO device: {self.device}")

    def _prepare_image(self, image):
        """
        AI2-THOR 的 rgb 有时类型不完全是标准 np.ndarray。
        这里统一转成 YOLO / OpenCV 能处理的格式。
        """

        image = np.asarray(image)

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 image, got shape {image.shape}")

        image = np.ascontiguousarray(image)

        return image

    def detect(self, image) -> List[Dict[str, Any]]:
        image = self._prepare_image(image)

        results = self.model.predict(
            source=image,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )

        detections = []

        for result in results:
            names = result.names

            if result.boxes is None:
                continue

            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                score = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(
                    {
                        "label": label,
                        "score": score,
                        "box": [x1, y1, x2, y2],
                    }
                )

        return detections
