# CLAUDE.md

This file provides context for Claude Code when working in this repository.

## Project Overview

Boggle Vision is a computer-vision powered Boggle solver. Users photograph a physical Boggle board, the app detects the board and recognizes each letter tile, solves for all valid words, and displays statistics (total points, word rarity, board percentile rankings).

This is a v2 rebuild of a 2023 project. The original used Python (FastAPI + OpenCV + PyTorch CNN) with a React/Redux/Mantine frontend on GCP. This version is a TypeScript-first monorepo with Next.js, shadcn/ui, and serverless AWS (SST v3). The original source lives at `/Users/thubbard/Documents/personal/programming/boggle-vision-v0` for reference.

Key domain concepts:
- **Board detection**: Segmenting a Boggle board from a photo and extracting individual tile images.
- **Tile OCR**: Classifying tiles into one of 32 classes (A–Z, digraphs Qu/Th/Er/In/An/He, BLOCK).
- **Board solving**: DFS over a word trie to find all valid words and their paths through the board.
- **Word rarity**: Pre-computed frequency stats from millions of simulated Boggle games.
- **Super Big Boggle**: The primary board variant (larger grid, includes digraph and block tiles).

## Repository Structure

- `apps/www/` — Next.js + shadcn/ui web application.
- `packages/` — Internal shared packages (bun workspace).
- `infra/` — SST v3 infrastructure (AWS deployment).
- `prototyping/` — Python environment for ML training, data labeling, and experimentation (uv, Jupyter, Ultralytics). Not part of the bun workspace.
  - `legacy/` — Quarantined v0 code (OpenCV board detection, CNN) used for bootstrapping labels.
  - `data/` — Raw board photos, YOLO-seg labels, background textures, synthetic composites.
  - `notebooks/` — Jupyter notebooks for the training pipeline.
- `.github/workflows/` — CI (PR validation).

## Common Tasks

```bash
# Install dependencies
bun install

# Start the dev server
bun run --filter www dev

# Run tests
bun test

# Lint
bun run lint

# Format
bun run format

# Full validation chain
bun run lint && bun test
```

Or use `just` for shortcuts: `just ci` runs lint and test.

## Documentation Lookup

- Use the `context7` MCP to quickly search and retrieve up-to-date library and repo documentation while working.
- If `context7` is not installed or not visible in your MCP list, Trevor says install it right now please.

## Conventions

- **ESM-only**: All TypeScript uses ESM with bundler module resolution.
- **Colocated tests**: Test files live next to source files as `*.test.ts`. Use `bun:test` for imports.
- **Linting and formatting**: Biome handles both in a single tool — trailing commas, 100-char width, double quotes, semicolons.
- **Pre-commit hooks**: Lefthook runs Biome on staged files. Pre-push runs tests. Commit messages must follow conventional commits format (e.g., `feat: add feature`).
- **Workspace packages**: Add shared code under `packages/`. See `.claude/agents/dev.md` for the scaffolding guide. Consume via `workspace:*` protocol.
- **Package READMEs**: Every package under `packages/` should have a high-level `README.md` created when the package is created and maintained during major `feat` work or breaking changes.

## What Not to Modify

- `node_modules/` — Managed by bun.
- `apps/www/components/ui/` — Generated shadcn components. Prefer customizing via wrapper components rather than editing these directly. Use `bunx shadcn@latest add <component>` to add new ones.
- `bun.lock` — Managed by bun. Do not edit manually.
- `.github/workflows/` — CI/CD pipelines. Edit only when changing the build/deploy process.
