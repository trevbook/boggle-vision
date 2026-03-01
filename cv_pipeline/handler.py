"""AWS Lambda handler for Boggle Vision CV pipeline."""

import base64
import json
import os
import time

import cv2
import numpy as np

from .analyze import analyze_board
from .model import load_cnn_session

# ── Lazy model loading (runs once per container on first real request) ─────
_yolo_model = None
_cnn_session = None


_MODEL_FILES = ["yolov8s-seg.pt", "boggle_cnn_v2.onnx", "boggle_cnn_v2.onnx.data"]


def _resolve_models_dir() -> str:
    """Return the local directory containing model files.

    In deployed Lambda, models are downloaded from S3 to /tmp/models on first
    call.  In sst dev mode, the MODELS_DIR env var points at local files.
    """
    # sst dev — local files (path only exists on the developer's machine)
    models_dir_env = os.environ.get("MODELS_DIR")
    if models_dir_env and os.path.isdir(models_dir_env):
        return models_dir_env

    # Deployed Lambda — download from S3 into /tmp/models
    if "MODELS_BUCKET" in os.environ:
        models_dir = "/tmp/models"
        marker = os.path.join(models_dir, ".ready")
        if not os.path.exists(marker):
            import boto3

            os.makedirs(models_dir, exist_ok=True)
            s3 = boto3.client("s3")
            bucket = os.environ["MODELS_BUCKET"]
            for key in _MODEL_FILES:
                s3.download_file(bucket, key, os.path.join(models_dir, key))
            # Marker so subsequent warm invocations skip the download
            open(marker, "w").close()
        return models_dir

    # Fallback — resolve relative to this file (direct uv run)
    return os.path.join(os.path.dirname(__file__), "models")


def _get_models():
    global _yolo_model, _cnn_session
    if _yolo_model is None:
        from ultralytics import YOLO

        models_dir = _resolve_models_dir()
        _yolo_model = YOLO(os.path.join(models_dir, "yolov8s-seg.pt"))
        _cnn_session = load_cnn_session(os.path.join(models_dir, "boggle_cnn_v2.onnx"))
    return _yolo_model, _cnn_session
# ────────────────────────────────────────────────────────────────────────────

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
}


def _response(status_code, body):
    return {
        "statusCode": status_code,
        "headers": {**_CORS_HEADERS, "Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def handler(event, context):
    body = json.loads(event.get("body") or "{}")

    # Warm request — load models eagerly, then return
    if body.get("warm"):
        _get_models()
        return _response(200, {"warm": True})

    # Extract image
    image_b64 = body.get("image")
    if not image_b64:
        return _response(400, {"error": "No image provided"})

    image_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        return _response(400, {"error": "Failed to decode image"})

    # Run pipeline
    yolo_model, cnn_session = _get_models()
    t0 = time.time()
    result = analyze_board(image, yolo_model, cnn_session)
    pipeline_ms = (time.time() - t0) * 1000

    if "error" in result:
        return _response(422, {"success": False, "error": result["error"]})

    return _response(200, {
        "success": True,
        "analysis": {
            "letters": result["letters"],
            "gridSize": result["grid_size"],
            "confidences": result["confidences"],
            "meanConfidence": result["mean_confidence"],
            "minConfidence": result["min_confidence"],
            "detectionConfidence": result["det_conf"],
        },
        "timing": {"pipelineMs": round(pipeline_ms)},
    })
