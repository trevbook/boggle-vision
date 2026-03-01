import type { TileLabel } from "./types";

export const toRow = (index: number, gridSize: number) => Math.floor(index / gridSize);

export const toCol = (index: number, gridSize: number) => index % gridSize;

export const reshapeToGrid = (letters: TileLabel[], gridSize: number): TileLabel[][] =>
  Array.from({ length: gridSize }, (_, r) => letters.slice(r * gridSize, (r + 1) * gridSize));
