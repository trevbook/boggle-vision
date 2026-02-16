# boggle-vision

Computer-vision powered Boggle solver

Boggle Vision is a computer-vision powered Boggle solver — snap a photo of a physical Boggle board, and it identifies the letters, finds every valid word, and gives you stats about the board. Built for real-time use during games of Super Big Boggle.

This is a ground-up rebuild of [the original Boggle Vision](https://github.com/trevbook/boggle-vision) (2023), which used a Python/FastAPI backend with OpenCV + a custom PyTorch CNN, a React/Redux frontend, and GCP hosting. The v2 swaps all of that for a TypeScript monorepo, Next.js UI with shadcn/ui, and serverless AWS infrastructure via SST.

## Prerequisites

- [Bun](https://bun.sh) (latest)
- Node.js 22+ (for Next.js dev server)
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
  package.json              # Monorepo root with dev tooling
  tsconfig.json             # TypeScript base config (bundler, strict)
  biome.json                # Biome linting and formatting
  bunfig.toml               # Bun configuration
  lefthook.yml              # Git hook automation
  justfile                  # Task runner shortcuts
  apps/
    www/  # Next.js + shadcn/ui
  packages/                 # Internal shared packages
  prototyping/              # Python ML environment (uv, Jupyter, Ultralytics)
    legacy/                 # Quarantined v0 code for bootstrapping
    notebooks/              # Training pipeline notebooks
    data/                   # Board photos, labels, synthetic data
  infra/                    # SST infrastructure
  sst.config.ts             # SST entry point
  .github/workflows/
    ci.yml                  # PR/push validation
```

## Architecture

The app's core pipeline has three stages:

1. **Board Detection** — Given a photo, detect the Boggle board region and extract individual tile images. The original used hand-tuned OpenCV contour detection; the v2 approach is TBD (exploring modern segmentation models like SAM).
2. **Letter Recognition** — Classify each tile image into one of 32 possible tiles (A–Z, plus digraphs like Qu, Th, Er, In, An, He, and BLOCK). The original trained a small custom CNN; v2 may leverage better models or multimodal LLMs.
3. **Board Solving** — Find all valid words via DFS over a trie built from a curated dictionary. Score words using Boggle rules and compute board statistics (total points, word count, rarity, z-scores vs. simulated distributions).

The frontend is a Next.js app (`apps/www/`) using shadcn/ui components. Shared logic lives in `packages/`. Infrastructure is defined with SST v3 (`infra/`) targeting serverless AWS.

## License

UNLICENSED
