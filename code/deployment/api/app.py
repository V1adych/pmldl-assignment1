import base64
import io
from datetime import datetime
from functools import wraps
from typing import List, Dict, Any
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, HTTPException, File, UploadFile

from pydantic import BaseModel

import numpy as np
import cv2

import logging

from detector import Detector
from cfg import DetectorConfig

detector = Detector(DetectorConfig())

CLASS_COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 0),
}

CLASS_MAPPING = {
    0: "broken",
    1: "healthy",
}


app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)

logger = logging.getLogger("app")


class ImageRequest(BaseModel):
    image: str


def validate_image(b64_data: str) -> None:
    try:
        raw = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(raw))
        img.verify()
    except (base64.binascii.Error, UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Image is corrupted or not valid")


def validate_image_bytes(image_bytes: bytes) -> None:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Image is corrupted or not valid")


def add_annotations(image: np.ndarray, annotations: List[Dict[str, Any]]) -> np.ndarray:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for annotation in annotations:
        x, y, w, h = annotation["box"]
        color = CLASS_COLORS[annotation["class"]]
        cv2.rectangle(
            image_bgr, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2
        )
        cv2.putText(
            image_bgr,
            CLASS_MAPPING[annotation["class"]],
            (x - w // 2, y - h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb


def run_detection_and_encode(image_np: np.ndarray) -> Dict[str, Any]:
    detections = detector(image_np)
    if len(detections) == 0:
        return {"success": 0, "spikes": []}

    image_with_annotations = add_annotations(image_np, detections)

    pil_image = Image.fromarray(image_with_annotations)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return {"success": 1, "spikes": detections, "image": img_str}


def perf_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger.info(f"{func.__name__}: starting")
        result = func(*args, **kwargs)
        end_time = datetime.now()
        logger.info(f"{func.__name__}: completed in {end_time - start_time}")
        result["perf_stats"] = {
            "request_received_timestamp": start_time.isoformat(timespec="milliseconds"),
            "request_completed_timestamp": end_time.isoformat(timespec="milliseconds"),
            "total_time_seconds": (end_time - start_time).total_seconds(),
        }
        return result

    return wrapper


def async_perf_logger(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger.info(f"{func.__name__}: starting")
        result = await func(*args, **kwargs)
        end_time = datetime.now()
        logger.info(f"{func.__name__}: completed in {end_time - start_time}")
        result["perf_stats"] = {
            "request_received_timestamp": start_time.isoformat(timespec="milliseconds"),
            "request_completed_timestamp": end_time.isoformat(timespec="milliseconds"),
            "total_time_seconds": (end_time - start_time).total_seconds(),
        }
        return result

    return wrapper


@app.post("/api/v1/detect_spikes")
@perf_logger
def analyze_thread(
    req: ImageRequest,
):
    validate_image(req.image)
    image_np = np.array(Image.open(io.BytesIO(base64.b64decode(req.image))))
    return run_detection_and_encode(image_np)


@app.post("/api/v1/bin/detect_spikes")
@async_perf_logger
async def analyze_thread_bin(
    image: UploadFile = File(...),
):
    contents = await image.read()
    validate_image_bytes(contents)
    image_np = np.array(Image.open(io.BytesIO(contents)))
    result = run_detection_and_encode(image_np)
    logger.info("/api/v1/bin/analyze_thread: thread pipeline completed")
    return result
