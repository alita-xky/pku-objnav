import os
import cv2
import numpy as np
from ultralytics import YOLO


class YoloWorldDetector:
    def __init__(
        self,
        classes,
        model_name="yolov8s-world.pt",
        conf=0.12,
        device="cpu",
    ):
        self.classes = classes
        self.model = YOLO(model_name)
        self.model.set_classes(classes)
        self.conf = conf
        self.device = device

    def detect(self, rgb_image):
        if isinstance(rgb_image, np.ndarray) and rgb_image.ndim == 3:
            bgr_image = rgb_image[:, :, ::-1]
        else:
            bgr_image = rgb_image

        results = self.model.predict(
            bgr_image,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )

        detections = []

        for r in results:
            names = r.names

            if r.boxes is None:
                continue

            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = names[cls_id]
                score = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "label": label,
                    "score": score,
                    "box": [x1, y1, x2, y2],
                })

        return detections

    def save_debug_image(self, rgb_image, detections, path, target_prompt=None):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            label = det["label"]
            score = det["score"]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img,
                f"{label} {score:.2f}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        cv2.imwrite(path, img)
