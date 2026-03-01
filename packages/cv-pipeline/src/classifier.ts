/**
 * Stage 8: CNN ONNX inference for tile classification.
 *
 * Ported from prototyping/notebooks/08-onnx-pipeline-validation.ipynb (predict_tiles_onnx).
 * Input: preprocessed 100x100 grayscale tile images (Uint8Array).
 * Output: predicted tile labels with confidence scores.
 */

import * as ort from "onnxruntime-node";
import { CLASS_LABELS, type TileLabel } from "./constants.js";

export interface TilePrediction {
  readonly label: TileLabel;
  readonly confidence: number;
}

/**
 * Create a reusable ONNX Runtime inference session for the BoggleCNN model.
 *
 * Call this once at startup (e.g. outside the Lambda handler) and reuse
 * the session across requests for fast inference.
 */
export async function createClassifierSession(modelPath: string): Promise<ort.InferenceSession> {
  return ort.InferenceSession.create(modelPath);
}

/**
 * Numerically stable softmax over a Float32Array segment.
 * Operates in-place on a pre-allocated output buffer.
 */
function softmax(logits: Float32Array, offset: number, length: number): Float32Array {
  const result = new Float32Array(length);

  // Find max for numerical stability
  let max = -Infinity;
  for (let i = 0; i < length; i++) {
    if (logits[offset + i] > max) max = logits[offset + i];
  }

  // exp(x - max) and sum
  let sum = 0;
  for (let i = 0; i < length; i++) {
    result[i] = Math.exp(logits[offset + i] - max);
    sum += result[i];
  }

  // Normalize
  for (let i = 0; i < length; i++) {
    result[i] /= sum;
  }

  return result;
}

/**
 * Classify a batch of preprocessed tile images using the BoggleCNN ONNX model.
 *
 * This is the TypeScript equivalent of `predict_tiles_onnx()` from notebook 08.
 *
 * @param session - ONNX Runtime session (from createClassifierSession).
 * @param tiles - Array of preprocessed tile images, each a Uint8Array of 100*100 pixels.
 * @returns Array of predictions with label and confidence for each tile.
 */
export async function classifyTiles(
  session: ort.InferenceSession,
  tiles: readonly Uint8Array[],
): Promise<readonly TilePrediction[]> {
  const n = tiles.length;
  if (n === 0) return [];

  const numClasses = CLASS_LABELS.length;
  const tileSize = 100 * 100;

  // Stack tiles into batch tensor: (N, 1, 100, 100) float32 in [0, 255] range
  // (matches Python: batch = np.stack(tiles).astype(np.float32)[:, np.newaxis, :, :])
  const batch = new Float32Array(n * tileSize);
  for (let i = 0; i < n; i++) {
    const tile = tiles[i];
    for (let j = 0; j < tileSize; j++) {
      batch[i * tileSize + j] = tile[j];
    }
  }

  const inputTensor = new ort.Tensor("float32", batch, [n, 1, 100, 100]);

  // Get the model's input name dynamically (matches Python: session.get_inputs()[0].name)
  const inputName = session.inputNames[0];
  const results = await session.run({ [inputName]: inputTensor });

  // Get the first output (logits): shape (N, 32)
  const outputName = session.outputNames[0];
  const logits = results[outputName].data as Float32Array;

  // Softmax + argmax → class labels (matches Python predict_tiles_onnx)
  const predictions: TilePrediction[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const probs = softmax(logits, i * numClasses, numClasses);

    // Argmax
    let maxIdx = 0;
    let maxProb = probs[0];
    for (let j = 1; j < numClasses; j++) {
      if (probs[j] > maxProb) {
        maxProb = probs[j];
        maxIdx = j;
      }
    }

    predictions[i] = {
      label: CLASS_LABELS[maxIdx],
      confidence: maxProb,
    };
  }

  return predictions;
}
