from shapely import unary_union, box

from src.constants import PubLayNetLabel
from src.types_ import BBox, DocumentImageAnnotation


def _find_bbox_group_iou(bboxes_0: list[BBox], bboxes_1: list[BBox]) -> float | None:
    if len(bboxes_0) == 0 and len(bboxes_1) == 0:
        # undefined iou
        return None
    elif len(bboxes_0) == 0 or len(bboxes_1) == 0:
        return 0.0
    else:
        bboxes_0_union = unary_union([box(*bbox.to_xyxy()) for bbox in bboxes_0])
        bboxes_1_union = unary_union([box(*bbox.to_xyxy()) for bbox in bboxes_1])
        intersection = bboxes_0_union.intersection(bboxes_1_union).area
        union = bboxes_0_union.union(bboxes_1_union).area
        return round(intersection / union, 2)


def get_detection_metrics(
    detections: list[DocumentImageAnnotation], true_annotations: list[DocumentImageAnnotation]
) -> dict:
    label_to_iou: dict[PubLayNetLabel, float] = {}
    for label in PubLayNetLabel:
        label_detection_bboxes = [
            detection.bbox for detection in detections if detection.label == label
        ]
        label_annotation_bboxes = [
            annotation.bbox for annotation in true_annotations if annotation.label == label
        ]
        label_to_iou[label.to_text()] = _find_bbox_group_iou(
            label_detection_bboxes, label_annotation_bboxes
        )
    return {
        "label_iou": label_to_iou,
    }
