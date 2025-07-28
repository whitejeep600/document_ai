import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

document_ai = FastAPI()


@document_ai.post("/detect")
async def _detect(file: UploadFile = File(...)):
    file_contents = await file.read()
    image = cv2.imdecode(np.frombuffer(file_contents, np.uint8), cv2.IMREAD_COLOR)

    # todo actual processing
    cv2.rectangle(image, (30, 30), (100, 100), (0, 0, 255), 20)

    _, encoded_image = cv2.imencode(".jpg", image)
    b64_img = base64.b64encode(encoded_image.tobytes()).decode("utf-8")

    return {"processed_image": b64_img}


@document_ai.post("/evaluate")
async def _evaluate(file: UploadFile = File(...), annotations: str = Form(...)):
    file_contents = await file.read()
    image = cv2.imdecode(np.frombuffer(file_contents, np.uint8), cv2.IMREAD_COLOR)

    # todo actual processing
    cv2.rectangle(image, (30, 30), (100, 100), (0, 255, 255), 20)

    _, encoded_image = cv2.imencode(".jpg", image)
    b64_img = base64.b64encode(encoded_image.tobytes()).decode("utf-8")

    return {"metrics": {"accuracy": 1}, "processed_image": b64_img}
