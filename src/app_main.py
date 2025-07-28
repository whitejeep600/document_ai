import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

from src.constants import HTTPMessageField
from src.serialization import serialize_image_for_http_response

document_ai = FastAPI()


@document_ai.post("/detect")
async def _detect(file: UploadFile = File(...)):
    file_contents = await file.read()
    image = cv2.imdecode(np.frombuffer(file_contents, np.uint8), cv2.IMREAD_COLOR)

    # todo actual processing
    cv2.rectangle(image, (30, 30), (100, 100), (0, 0, 255), 20)

    serialized_image = serialize_image_for_http_response(image)

    return {HTTPMessageField.PROCESSED_IMAGE: serialized_image}


@document_ai.post("/evaluate")
async def _evaluate(file: UploadFile = File(...), annotations: str = Form(...)):
    file_contents = await file.read()
    image = cv2.imdecode(np.frombuffer(file_contents, np.uint8), cv2.IMREAD_COLOR)

    # todo actual processing
    cv2.rectangle(image, (30, 30), (100, 100), (0, 255, 255), 20)

    serialized_image = serialize_image_for_http_response(image)

    return {
        HTTPMessageField.METRICS: {"accuracy": 1},
        HTTPMessageField.PROCESSED_IMAGE: serialized_image,
    }
