import { beforeAll, describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { InferenceSession } from "onnxruntime-node";
import { classifyTiles, createClassifierSession } from "./classifier.js";
import { CLASS_LABELS } from "./constants.js";

// ---------------------------------------------------------------------------
// Helpers: minimal NPY parser for uint8 and int64 arrays
// ---------------------------------------------------------------------------

interface NpyArray<T extends TypedArray> {
  shape: readonly number[];
  data: T;
}

type TypedArray = Uint8Array | Int32Array | BigInt64Array;

function parseNpy(buffer: Buffer): NpyArray<TypedArray> {
  // Magic: \x93NUMPY
  const magic = buffer.subarray(0, 6);
  if (magic[0] !== 0x93 || String.fromCharCode(...magic.subarray(1, 6)) !== "NUMPY") {
    throw new Error("Not a valid NPY file");
  }

  const headerLen = buffer.readUInt16LE(8);
  const headerStr = buffer.subarray(10, 10 + headerLen).toString("ascii");

  // Parse shape from header string like "{'descr': '<u1', 'fortran_order': False, 'shape': (1368, 100, 100), }"
  const shapeMatch = headerStr.match(/'shape':\s*\(([^)]*)\)/);
  if (!shapeMatch) throw new Error("Could not parse shape from NPY header");
  const shape = shapeMatch[1]
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number);

  const descrMatch = headerStr.match(/'descr':\s*'([^']+)'/);
  if (!descrMatch) throw new Error("Could not parse dtype from NPY header");
  const dtype = descrMatch[1];

  const dataOffset = 10 + headerLen;
  const rawData = buffer.subarray(dataOffset);

  if (dtype === "|u1" || dtype === "<u1") {
    return { shape, data: new Uint8Array(rawData.buffer, rawData.byteOffset, rawData.byteLength) };
  }
  if (dtype === "<i8") {
    const totalElements = shape.reduce((a, b) => a * b, 1);
    // Convert int64 → int32 (class indices are small)
    const int32 = new Int32Array(totalElements);
    for (let i = 0; i < totalElements; i++) {
      int32[i] = Number(rawData.readBigInt64LE(i * 8));
    }
    return { shape, data: int32 };
  }

  throw new Error(`Unsupported NPY dtype: ${dtype}`);
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const REPO_ROOT = resolve(import.meta.dir, "../../..");
const MODEL_PATH = resolve(REPO_ROOT, "prototyping/legacy/models/boggle_cnn_v2.onnx");
const TILES_PATH = resolve(REPO_ROOT, "prototyping/data/training-data-v2/tiles_raw.npy");
const LABELS_PATH = resolve(REPO_ROOT, "prototyping/data/training-data-v2/labels_raw.npy");

// ---------------------------------------------------------------------------
// Test data
// ---------------------------------------------------------------------------

let session: InferenceSession;
let allTiles: Uint8Array; // flat (1368 * 100 * 100)
let allLabels: Int32Array; // (1368,)
let numTiles: number;

beforeAll(async () => {
  session = await createClassifierSession(MODEL_PATH);

  const tilesNpy = parseNpy(readFileSync(TILES_PATH));
  allTiles = tilesNpy.data as Uint8Array;
  numTiles = tilesNpy.shape[0];

  const labelsNpy = parseNpy(readFileSync(LABELS_PATH));
  allLabels = labelsNpy.data as Int32Array;
});

/** Extract a single 100x100 tile from the flat buffer. */
function getTile(index: number): Uint8Array {
  const tileSize = 100 * 100;
  return allTiles.subarray(index * tileSize, (index + 1) * tileSize);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("classifier", () => {
  test("session loads successfully", () => {
    expect(session).toBeDefined();
    expect(session.inputNames.length).toBeGreaterThan(0);
    expect(session.outputNames.length).toBeGreaterThan(0);
  });

  test("classifies a single tile correctly", async () => {
    const tile = getTile(0);
    const expectedLabel = CLASS_LABELS[allLabels[0]];

    const predictions = await classifyTiles(session, [tile]);

    expect(predictions).toHaveLength(1);
    expect(predictions[0].label).toBe(expectedLabel);
    expect(predictions[0].confidence).toBeGreaterThan(0.5);
  });

  test("classifies a batch of tiles with high accuracy", async () => {
    // Sample first 36 tiles (one full board worth)
    const sampleSize = 36;
    const tiles: Uint8Array[] = [];
    for (let i = 0; i < sampleSize; i++) {
      tiles.push(getTile(i));
    }

    const predictions = await classifyTiles(session, tiles);

    expect(predictions).toHaveLength(sampleSize);

    let correct = 0;
    for (let i = 0; i < sampleSize; i++) {
      if (predictions[i].label === CLASS_LABELS[allLabels[i]]) {
        correct++;
      }
    }

    const accuracy = correct / sampleSize;
    // The CNN achieves ~99.5% on training data — expect at least 90% on raw tiles
    expect(accuracy).toBeGreaterThanOrEqual(0.9);
  });

  test("returns empty array for empty input", async () => {
    const predictions = await classifyTiles(session, []);
    expect(predictions).toHaveLength(0);
  });

  test("all predictions have valid labels", async () => {
    const tiles = [getTile(0), getTile(10), getTile(20)];
    const predictions = await classifyTiles(session, tiles);

    for (const pred of predictions) {
      expect(CLASS_LABELS).toContain(pred.label);
      expect(pred.confidence).toBeGreaterThan(0);
      expect(pred.confidence).toBeLessThanOrEqual(1);
    }
  });

  test("achieves >95% accuracy across all raw tiles", async () => {
    // Run inference on ALL raw tiles in batches
    const batchSize = 100;
    let correct = 0;

    for (let start = 0; start < numTiles; start += batchSize) {
      const end = Math.min(start + batchSize, numTiles);
      const tiles: Uint8Array[] = [];
      for (let i = start; i < end; i++) {
        tiles.push(getTile(i));
      }

      const predictions = await classifyTiles(session, tiles);

      for (let i = 0; i < predictions.length; i++) {
        if (predictions[i].label === CLASS_LABELS[allLabels[start + i]]) {
          correct++;
        }
      }
    }

    const accuracy = correct / numTiles;
    console.log(`Full dataset accuracy: ${correct}/${numTiles} = ${(accuracy * 100).toFixed(1)}%`);
    expect(accuracy).toBeGreaterThanOrEqual(0.95);
  });
});
