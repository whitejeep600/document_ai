from enum import Enum

import cv2
import numpy as np

from src.constants import PubLayNetLabel
from src.types_ import DocumentImageAnnotation

# "Normalized" here means that this is defined wrt. the size of the image, instead of
# a constant value in pixels.
_NORMALIZED_LINE_THICKNESS = 0.01

_LABEL_PLOTTING_OPACITY = 0.3


class Color(Enum):
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    PURPLE = (128, 0, 128)
    YELLOW = (255, 255, 0)


_LABEL_COLOR_MAPPING = {
    PubLayNetLabel.TEXT: Color.RED.value,
    PubLayNetLabel.TITLE: Color.BLUE.value,
    PubLayNetLabel.LIST: Color.GREEN.value,
    PubLayNetLabel.TABLE: Color.PURPLE.value,
    PubLayNetLabel.FIGURE: Color.YELLOW.value,
}


def _overlay_detection_on_image(
    image: np.ndarray, detection: DocumentImageAnnotation
) -> np.ndarray:
    """
    As a rectangle edge.
    """
    image = image.copy()
    line_thickness = int(_NORMALIZED_LINE_THICKNESS * min(image.shape[:2]))
    color = _LABEL_COLOR_MAPPING[detection.label]
    cv2.rectangle(
        image, detection.bbox.start_point(), detection.bbox.end_point(), color, line_thickness
    )
    return image


def _overlay_label_on_image(image: np.ndarray, label: DocumentImageAnnotation) -> np.ndarray:
    """
    As a semi-transparent, filled rectangle.
    """
    image = image.copy()
    image_with_opaque_label_bbox = image.copy()
    color = _LABEL_COLOR_MAPPING[label.label]
    cv2.rectangle(
        image_with_opaque_label_bbox, label.bbox.start_point(), label.bbox.end_point(), color, -1
    )
    image = cv2.addWeighted(
        image, 1 - _LABEL_PLOTTING_OPACITY, image_with_opaque_label_bbox, _LABEL_PLOTTING_OPACITY, 0
    )
    return image


def overlay_detections_on_image(
    image: np.ndarray, detections: list[DocumentImageAnnotation]
) -> np.ndarray:
    for detection in detections:
        image = _overlay_detection_on_image(image, detection)
    return image


def overlay_labels_on_image(image: np.ndarray, labels: list[DocumentImageAnnotation]) -> np.ndarray:
    for label in labels:
        image = _overlay_label_on_image(image, label)
    return image
