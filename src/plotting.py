from enum import Enum

import cv2
import numpy as np

from src.types_ import DetectionOrAnnotation, PubLayNetCategory

# "Normalized" here means that this is defined wrt. the size of the image, instead of
# a constant value in pixels.

_NORMALIZED_LINE_THICKNESS = 0.01

_DETECTION_OR_ANNOTATION_PLOTTING_OPACITY = 0.3


class Color(Enum):
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    PURPLE = (128, 0, 128)
    YELLOW = (255, 255, 0)


_CATEGORY_COLOR_MAPPING = {
    PubLayNetCategory.TEXT: Color.RED.value,
    PubLayNetCategory.TITLE: Color.BLUE.value,
    PubLayNetCategory.LIST: Color.GREEN.value,
    PubLayNetCategory.TABLE: Color.PURPLE.value,
    PubLayNetCategory.FIGURE: Color.YELLOW.value,
}


def _overlay_detection_on_image(
    image: np.ndarray, detection: DetectionOrAnnotation
) -> np.ndarray:
    """
    As a rectangle edge.
    """
    image = image.copy()
    line_thickness = int(_NORMALIZED_LINE_THICKNESS * min(image.shape[:2]))
    color = _CATEGORY_COLOR_MAPPING[detection.category]
    cv2.rectangle(
        image, detection.bbox.start_point(), detection.bbox.end_point(), color, line_thickness
    )
    return image


def _overlay_annotation_on_image(image: np.ndarray, annotation: DetectionOrAnnotation) -> np.ndarray:
    """
    As a semi-transparent, filled rectangle.
    """
    image = image.copy()
    image_with_opaque_annotation_bbox = image.copy()
    color = _CATEGORY_COLOR_MAPPING[annotation.category]
    cv2.rectangle(
        image_with_opaque_annotation_bbox, annotation.bbox.start_point(), annotation.bbox.end_point(), color, -1
    )
    image = cv2.addWeighted(
        image, 1 - _DETECTION_OR_ANNOTATION_PLOTTING_OPACITY, image_with_opaque_annotation_bbox, _DETECTION_OR_ANNOTATION_PLOTTING_OPACITY, 0
    )
    return image


def overlay_detections_on_image(
    image: np.ndarray, detections: list[DetectionOrAnnotation]
) -> np.ndarray:
    for detection in detections:
        image = _overlay_detection_on_image(image, detection)
    return image


def overlay_annotations_on_image(image: np.ndarray, annotations: list[DetectionOrAnnotation]) -> np.ndarray:
    for annotation in annotations:
        image = _overlay_annotation_on_image(image, annotation)
    return image
