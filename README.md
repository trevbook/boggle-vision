# boggle-vision

Computer-vision powered Boggle solver

Boggle Vision is a computer-vision powered Boggle solver — snap a photo of a physical Boggle board, and it identifies the letters, finds every valid word, and gives you stats about the board. Built for real-time use during games of Super Big Boggle.

This is a ground-up rebuild of [the original Boggle Vision](https://github.com/trevbook/boggle-vision) (2023), which used a Python/FastAPI backend with OpenCV + a custom PyTorch CNN, a React/Redux frontend, and GCP hosting. The v2 uses a Next.js frontend with shadcn/ui, a Python CV pipeline (Ultralytics YOLO + OpenCV + ONNX Runtime) deployed as a containerized AWS Lambda, and a TypeScript board solver — all wired together with SST v3.

## Prerequisites

- [Bun](https://bun.sh) (latest)
- Node.js 22+ (for Next.js dev server)
- [uv](https://docs.astral.sh/uv/) (for Python CV pipeline)
- Docker (for building the Lambda container image)
- AWS CLI (configured) for SST deployment

## Getting Started

```bash
# Install dependencies
bun install

# Start the dev server
bun run --filter www dev

# Run tests
bun test
```

Or use `just` for shortcuts — run `just` to see all available commands.

## Available Scripts

| Command | Description |
|---------|-------------|
| `bun test` | Run tests (bun:test) |
| `bun test --watch` | Run tests in watch mode |
| `bun run lint` | Lint with Biome |
| `bun run lint:fix` | Lint and auto-fix |
| `bun run format` | Format with Biome |
| `bun run format:check` | Check formatting |
| `just ci` | Run all checks (lint, test) |

## Project Structure

```
boggle-vision/
  package.json              # Bun workspace (apps/*, packages/*)
  pyproject.toml            # uv workspace (cv_pipeline/)
  sst.config.ts             # SST entry point
  justfile                  # Task runner shortcuts
  apps/
    www/                    # Next.js + shadcn/ui frontend
  packages/                 # Internal TS packages (bun workspace)
    solver/                 # Pure TS board solver (trie + DFS)
  cv_pipeline/              # Python CV pipeline (containerized Lambda)
    handler.py              # Lambda entry point
    analyze.py              # 8-stage pipeline orchestrator
    detection.py            # YOLO segmentation + mask + quad fitting
    grid.py                 # Warp + tile center detection + grid assignment
    tiles.py                # Per-tile extraction + preprocessing
    model.py                # CNN classifier (ONNX Runtime)
    Dockerfile              # Lambda container image
  prototyping/              # Python ML environment (uv, Jupyter, Ultralytics)
    legacy/                 # Quarantined v0 code for bootstrapping
    notebooks/              # Training pipeline notebooks
    data/                   # Board photos, labels, synthetic data
  infra/                    # SST infrastructure
  .github/workflows/
    ci.yml                  # PR/push validation
```

## Architecture

The app has two main backends:

1. **CV Pipeline** (Python Lambda) — An 8-stage pipeline that takes a board photo and returns detected letters + confidences. Uses a YOLOv8-seg model for board segmentation, OpenCV for image processing (warping, thresholding, tile extraction), and a custom CNN (via ONNX Runtime) for letter classification into one of 32 tile classes (A–Z, digraphs Qu/Th/Er/In/An/He, BLOCK). Deployed as a containerized Lambda behind API Gateway.
2. **Board Solver** (TypeScript) — Pure TypeScript package that finds all valid words via DFS over a trie. Scores words using Boggle rules. Runs client-side or in a Next.js API route (<50ms).

The frontend is a Next.js app (`apps/www/`) using shadcn/ui components. Infrastructure is defined with SST v3 (`infra/`) targeting serverless AWS.

## License

UNLICENSED
