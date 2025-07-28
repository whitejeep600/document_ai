import numpy as np
from paddleocr import PPStructure

from src.types_ import DetectionOrAnnotation, BBox, PubLayNetCategory


def _raw_detection_to_annotation(raw_detection: dict) -> DetectionOrAnnotation | None:
    x_min, y_min, x_max, y_max = raw_detection["bbox"]
    bbox = BBox.from_xyxy(x_min, y_min, x_max, y_max)
    category = PubLayNetCategory.from_text(raw_detection["type"])
    if category is None:
        # Unrecognized category (PPStructure recognizes some categories that are not in PubLayNet).
        return None
    else:
        return DetectionOrAnnotation(category, bbox)


class LayoutDetector:
    """
    This is a callable class so that the model does not have to be initialized at
    every call.
    """
    model = PPStructure(layout=True)

    def __call__(self, image: np.ndarray) -> list[DetectionOrAnnotation]:
        raw_detections = self.model(image)
        detections = [
            _raw_detection_to_annotation(raw_detection) for raw_detection in raw_detections
        ]
        return [detection for detection in detections if detection is not None]