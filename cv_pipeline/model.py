"""Stage 8: CNN tile classification via ONNX Runtime."""

import numpy as np
import onnxruntime as ort

from .constants import CLASS_LABELS


def load_cnn_session(model_path: str) -> ort.InferenceSession:
    """Load the BoggleCNN ONNX model into an inference session."""
    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def predict_tiles_batch(session: ort.InferenceSession, preprocessed_tiles):
    """Run CNN inference on a list of preprocessed tile arrays.

    Args:
        session: ONNX Runtime InferenceSession for BoggleCNN.
        preprocessed_tiles: list of (H, W) uint8 numpy arrays (output of preprocess_tile_v0).

    Returns:
        (letters, confidences) — parallel lists of str and float.
    """
    batch = np.stack(preprocessed_tiles).astype(np.float32)[:, np.newaxis, :, :]  # (N, 1, H, W)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: batch})[0]

    # Stable softmax
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    idxs = probs.argmax(axis=1)
    confs = probs[np.arange(len(idxs)), idxs]
    letters = [CLASS_LABELS[i] for i in idxs]
    return letters, confs.tolist()
