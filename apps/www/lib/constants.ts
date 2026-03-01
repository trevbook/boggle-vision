import type { TileLabel } from "./types";

export const SINGLE_LETTERS: TileLabel[] = [
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "H",
  "I",
  "J",
  "K",
  "L",
  "M",
  "N",
  "O",
  "P",
  "R",
  "S",
  "T",
  "U",
  "V",
  "W",
  "X",
  "Y",
  "Z",
];

export const DIGRAPHS: TileLabel[] = ["Qu", "Er", "Th", "In", "An", "He"];

export const ALL_TILE_LABELS: TileLabel[] = [...SINGLE_LETTERS, ...DIGRAPHS, "BLOCK"];

export const CONFIDENCE_THRESHOLDS = {
  LOW: 0.7,
  MEDIUM: 0.85,
} as const;

export const QUIP_INTERVAL_MS = 2500;
export const PATH_TRANSITION_MS = 200;
export const MAX_SAVED_BOARDS = 10;
export const IMAGE_MAX_DIMENSION = 1920;
