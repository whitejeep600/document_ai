from shapely import unary_union, box

from src.types_ import BBox, DetectionOrAnnotation, PubLayNetCategory


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
    detections: list[DetectionOrAnnotation], annotations: list[DetectionOrAnnotation]
) -> dict:
    category_to_iou: dict[PubLayNetCategory, float] = {}
    for category in PubLayNetCategory:
        category_detection_bboxes = [
            detection.bbox for detection in detections if detection.category == category
        ]
        category_annotation_bboxes = [
            annotation.bbox for annotation in annotations if annotation.category == category
        ]
        category_to_iou[category.value] = _find_bbox_group_iou(
            category_detection_bboxes, category_annotation_bboxes
        )
    return {
        "category_iou": category_to_iou,
    }
