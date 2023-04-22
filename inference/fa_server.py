from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from inference import classify


class Item(BaseModel):
    data: str


app = FastAPI()


@app.get("/")
async def root() -> dict:
    return {"message": "Hello, Server is running"}


@app.post("/score")
async def score(item: Item) -> dict:
    """Scoring endpoint

    Returns:
        dict: predicted data
    """
    return classify(item.data)
