"""REST API:  uvicorn api:app --reload   (docs at http://127.0.0.1:8000/docs)"""
import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cyberguard import __version__
from cyberguard.modeling import explain, load_bundle, predict_urls

MODEL_PATH = os.getenv("CYBERGUARD_MODEL", "models/cyberguard_model.joblib")
MAX_URL_LEN = 2048

app = FastAPI(title="CyberGuard API", version=__version__,
              description="Educational ML URL scanner. Not a production security tool.")


@lru_cache(maxsize=1)
def bundle():
    try:
        return load_bundle(MODEL_PATH)
    except FileNotFoundError:
        raise HTTPException(503, "Model not trained yet. Run: python -m cyberguard train --data your.csv")


class URLIn(BaseModel):
    url: str = Field(..., min_length=1, max_length=MAX_URL_LEN, examples=["http://secure-login.example.xyz/verify"])


class BatchIn(BaseModel):
    urls: list[str] = Field(..., min_length=1, max_length=1000)


@app.get("/health")
def health():
    b = bundle()
    return {"status": "ok", "version": __version__, "model": b["name"], "demo_model": b.get("demo", False)}


@app.post("/predict")
def predict(body: URLIn):
    return explain(bundle(), body.url)


@app.post("/predict/batch")
def predict_batch(body: BatchIn):
    if any(len(u) > MAX_URL_LEN for u in body.urls):
        raise HTTPException(422, f"Each URL must be at most {MAX_URL_LEN} characters")
    return predict_urls(bundle(), body.urls).to_dict("records")
