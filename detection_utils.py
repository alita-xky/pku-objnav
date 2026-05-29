TARGET_ALIASES = {
    "Television": ["Television", "tv"],
    "Sofa": ["Sofa", "couch"],
    "Mug": ["Mug", "cup"],
    "RemoteControl": ["RemoteControl", "remote"],
    "CellPhone": ["CellPhone", "cell phone"],
    "Laptop": ["Laptop"],
    "Book": ["Book"],
    "Chair": ["Chair", "ArmChair", "DiningChair"],
    "CoffeeTable": ["CoffeeTable", "table"],
}


def target_detected(detections, target_type, threshold=0.5):
    """
    判断 detector 输出里是否包含目标物体。

    这个函数可以同时兼容：
    1. MetadataDetector 输出的 AI2-THOR 类别名，比如 Television
    2. YOLO 输出的类别名，比如 tv
    """

    valid_labels = TARGET_ALIASES.get(target_type, [target_type])
    valid_labels = [x.lower() for x in valid_labels]

    best_det = None
    best_score = -1.0

    for det in detections:
        label = det["label"].lower()
        score = det.get("score", 1.0)

        if label in valid_labels and score >= threshold:
            if score > best_score:
                best_score = score
                best_det = det

    if best_det is not None:
        return True, best_det

    return False, None
