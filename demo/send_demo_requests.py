import base64
import io
import json
from pathlib import Path

import cv2
import numpy as np
import requests

from src.types_ import BBox, DetectionOrAnnotation, DocumentImageSample, PubLayNetCategory

_DETECT_URL = "http://localhost:8000/detect"
_EVALUATE_URL = "http://localhost:8000/evaluate"
_IMAGES_ROOT = Path("demo/publaynet_example_data/images")
_SAMPLES_DATA_PATH = Path("demo/publaynet_example_data/samples.json")
_RESULT_ROOT = Path("demo/results")
_DETECT_ENDPOINT_RESULT_PATH = _RESULT_ROOT / "detect_endpoint"
_EVALUATE_ENDPOINT_RESULT_PATH = _RESULT_ROOT / "evaluate_endpoint"
_EVALUATE_ENDPOINT_IMAGES_PATH = _EVALUATE_ENDPOINT_RESULT_PATH / "images"
_EVALUATE_ENDPOINT_METRICS_PATH = _EVALUATE_ENDPOINT_RESULT_PATH / "metrics"


def _read_publaynet_annotation(raw_annotation: dict) -> DetectionOrAnnotation:
    return DetectionOrAnnotation(
        category=PubLayNetCategory.from_category_code(raw_annotation["category_id"]),
        bbox=BBox.from_xywh(*[int(x) for x in raw_annotation["bbox"]]),
    )


def _read_image_samples() -> list[DocumentImageSample]:
    samples: list[DocumentImageSample] = []

    with open(_SAMPLES_DATA_PATH, "r") as f:
        sample_data = json.load(f)

    for image_path in _IMAGES_ROOT.iterdir():
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_data = [
            image_data
            for image_data in sample_data["images"]
            if image_data["file_name"] == image_path.name
        ][0]
        image_id = image_data["id"]
        raw_image_annotations = [
            annotation
            for annotation in sample_data["annotations"]
            if annotation["image_id"] == image_id
        ]
        image_annotations = [
            _read_publaynet_annotation(raw_annotation) for raw_annotation in raw_image_annotations
        ]
        samples.append(DocumentImageSample(image, image_annotations, image_path.name))

    return samples


def _demo_detect_endpoint(samples: list[DocumentImageSample]):
    _DETECT_ENDPOINT_RESULT_PATH.mkdir(exist_ok=True, parents=True)

    for sample in samples:
        _, encoded = cv2.imencode(".jpg", sample.image)
        image_bytes = io.BytesIO(encoded.tobytes())
        files = {
            "file": ("image.jpg", image_bytes, "image/jpeg")
        }

        response = requests.post(_DETECT_URL, files=files)
        data = response.json()
        image_with_detections_bytes = np.frombuffer(base64.b64decode(data["processed_image"]), dtype=np.uint8)
        image_with_detections = cv2.imdecode(image_with_detections_bytes, cv2.IMREAD_COLOR)
        cv2.imwrite(str(_DETECT_ENDPOINT_RESULT_PATH / sample.image_filename), image_with_detections)


def _demo_evaluate_endpoint(samples: list[DocumentImageSample]):
    for path in _EVALUATE_ENDPOINT_METRICS_PATH, _EVALUATE_ENDPOINT_IMAGES_PATH:
        path.mkdir(exist_ok=True, parents=True)

    for sample in samples:
        _, encoded = cv2.imencode(".jpg", sample.image)
        image_bytes = io.BytesIO(encoded.tobytes())
        files = {
            "file": ("image.jpg", image_bytes, "image/jpeg")
        }

        reference_annotations = {"annotations": json.dumps({"this": 1, "that": 2})}
        response = requests.post(_EVALUATE_URL, files=files, data=reference_annotations)
        data = response.json()
        metrics = data["metrics"]
        image_with_detections_bytes = np.frombuffer(base64.b64decode(data["processed_image"]), dtype=np.uint8)
        image_with_detections = cv2.imdecode(image_with_detections_bytes, cv2.IMREAD_COLOR)
        cv2.imwrite(str(_EVALUATE_ENDPOINT_IMAGES_PATH / sample.image_filename), image_with_detections)
        with open(str(_EVALUATE_ENDPOINT_METRICS_PATH / sample.image_filename.replace(".jpg", ".json")), "w") as f:
            json.dump(metrics, f)


def main():
    samples = _read_image_samples()
    _demo_detect_endpoint(samples)
    _demo_evaluate_endpoint(samples)


if __name__ == "__main__":
    main()
