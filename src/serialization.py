import base64
import io
import json

import cv2
import numpy as np

from src.types_ import DetectionOrAnnotation


def serialize_image_for_http_request(image: np.ndarray) -> dict:
    _, encoded_image = cv2.imencode(".jpg", image)
    image_bytes = io.BytesIO(encoded_image.tobytes())
    return {"file": ("image.jpg", image_bytes, "image/jpeg")}


def serialize_image_for_http_response(image: np.ndarray) -> str:
    _, encoded_image = cv2.imencode(".jpg", image)
    b64_img = base64.b64encode(encoded_image.tobytes()).decode("utf-8")
    return b64_img


def deserialize_image_from_http_response(image_bytes: bytes) -> np.ndarray:
    image_with_detections_array = np.frombuffer(base64.b64decode(image_bytes), dtype=np.uint8)
    return cv2.imdecode(image_with_detections_array, cv2.IMREAD_COLOR)


def serialize_annotations(annotations: list[DetectionOrAnnotation]) -> str:
    return json.dumps([annotation.as_dict() for annotation in annotations])


def deserialize_annotations(serialized_annotations: str) -> list[DetectionOrAnnotation]:
    separate_serialized_annotations: list[dict] = json.loads(serialized_annotations)
    return [DetectionOrAnnotation.from_dict(d) for d in separate_serialized_annotations]
