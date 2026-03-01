/** 32-class label set for the BoggleCNN tile classifier. */
export const CLASS_LABELS = [
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
  "Qu",
  "Er",
  "Th",
  "In",
  "An",
  "He",
  "BLOCK",
] as const;

export type TileLabel = (typeof CLASS_LABELS)[number];

/** COCO class IDs used for board detection (remote, keyboard, cell phone). */
export const BOARD_CLASSES = new Set([65, 66, 67]);

/** YOLO confidence threshold (with fallback cascade in detection). */
export const YOLO_CONF = 0.25;

/** YOLO input image size. */
export const YOLO_IMGSZ = 640;

/** Preprocessed tile size (pixels). */
export const TARGET_TILE_SIZE = 100;

/** Margin ratio from grid edges when locating tile centers. */
export const GRID_INSET_RATIO = 0.1;

/** Padding outside quad corners during perspective warp. */
export const WARP_PAD_PCT = 0.07;
