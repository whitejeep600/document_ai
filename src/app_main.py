import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

from src.constants import HTTPMessageField
from src.layout_detector import LayoutDetector
from src.metrics import get_detection_metrics
from src.plotting import overlay_detections_on_image, overlay_annotations_on_image
from src.serialization import deserialize_annotations, serialize_image_for_http_response

document_ai = FastAPI()


@document_ai.post("/detect")
async def _detect(file: UploadFile = File(...)):
    file_contents = await file.read()
    image = cv2.imdecode(np.frombuffer(file_contents, np.uint8), cv2.IMREAD_COLOR)

    detections = LayoutDetector()(image)
    image_with_detections = overlay_detections_on_image(image, detections)

    serialized_image = serialize_image_for_http_response(image_with_detections)

    return {HTTPMessageField.PROCESSED_IMAGE: serialized_image}


@document_ai.post("/evaluate")
async def _evaluate(file: UploadFile = File(...), serialized_annotations: str = Form(...)):
    file_contents = await file.read()
    image = cv2.imdecode(np.frombuffer(file_contents, np.uint8), cv2.IMREAD_COLOR)
    annotations = deserialize_annotations(serialized_annotations)

    detections = LayoutDetector()(image)
    image_with_annotations = overlay_annotations_on_image(image, annotations)
    image_with_annotations_and_detections = overlay_detections_on_image(
        image_with_annotations, detections
    )
    metrics = get_detection_metrics(detections, annotations)

    serialized_image = serialize_image_for_http_response(image_with_annotations_and_detections)

    return {
        HTTPMessageField.METRICS: metrics,
        HTTPMessageField.PROCESSED_IMAGE: serialized_image,
    }
