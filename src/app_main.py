from fastapi import FastAPI, Request


document_ai = FastAPI()


@document_ai.post("/detect")
async def detect(request: Request):
    data = await request.json()
    return {"just a placeholder endpoint for now": ""}


@document_ai.post("/evaluate")
async def evaluate(request: Request):
    data = await request.json()
    return {"just a placeholder endpoint for now": ""}