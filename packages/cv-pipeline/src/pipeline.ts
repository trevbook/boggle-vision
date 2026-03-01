/**
 * Pipeline orchestrator — ties together all CV pipeline stages.
 *
 * Currently only Stage 8 (classification) is implemented.
 * Stages 1-7 (detection, geometry, grid, tiles) will be added in Phase 3.
 */

export type { TilePrediction } from "./classifier.js";
export { classifyTiles, createClassifierSession } from "./classifier.js";
