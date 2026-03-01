# @boggle-vision/cv-pipeline

Computer-vision pipeline for Boggle board analysis. Takes a raw board photo and returns a grid of classified tile labels.

## Status

**Stage 8 (CNN classifier)** is implemented. Stages 1-7 (YOLO detection, mask cleanup, geometry, grid detection, tile extraction, preprocessing) will be added in Phase 3.

## Usage

```typescript
import { createClassifierSession, classifyTiles } from "@boggle-vision/cv-pipeline";

// Create session once (reuse across requests)
const session = await createClassifierSession("path/to/boggle_cnn_v2.onnx");

// Classify preprocessed tile images
const tiles: Uint8Array[] = [/* 100x100 grayscale images */];
const predictions = await classifyTiles(session, tiles);
// [{ label: "A", confidence: 0.99 }, { label: "Qu", confidence: 0.97 }, ...]
```

## Architecture

| Stage | Module | Status |
|-------|--------|--------|
| 1-2 | `detection.ts` — YOLO inference + mask cleanup | Planned |
| 3-4 | `geometry.ts` — Quad fitting + perspective warp | Planned |
| 5 | `grid.ts` — Grid size detection + tile centers | Planned |
| 6-7 | `tiles.ts` — Tile extraction + preprocessing | Planned |
| 8 | `classifier.ts` — CNN ONNX inference | Done |

## Models

- **BoggleCNN v2** (`boggle_cnn_v2.onnx`, 1.7 MB) — 32-class tile classifier (A-Z, Qu, Er, Th, In, An, He, BLOCK)
- **YOLOv8s-seg** (`best.onnx`, ~24 MB) — Board segmentation (Phase 3)
