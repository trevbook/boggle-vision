import type { SolvedWord, SolveResult } from "@boggle-vision/solver";

export type { SolvedWord, SolveResult };

/** The 32 tile classes recognized by the CNN classifier. */
export type TileLabel =
  | "A"
  | "B"
  | "C"
  | "D"
  | "E"
  | "F"
  | "G"
  | "H"
  | "I"
  | "J"
  | "K"
  | "L"
  | "M"
  | "N"
  | "O"
  | "P"
  | "R"
  | "S"
  | "T"
  | "U"
  | "V"
  | "W"
  | "X"
  | "Y"
  | "Z"
  | "Qu"
  | "Er"
  | "Th"
  | "In"
  | "An"
  | "He"
  | "BLOCK";

export type AppScreen = "capture" | "processing" | "results";
export type SortOption = "length" | "points" | "alpha";

export interface AnalysisData {
  letters: TileLabel[];
  gridSize: number;
  confidences: number[];
  meanConfidence: number;
  minConfidence: number;
  detectionConfidence: number;
}

export interface TimingData {
  pipelineMs: number;
  solverMs: number;
}

export interface AnalyzeResponse {
  success: true;
  analysis: AnalysisData;
  timing: { pipelineMs: number };
}

export interface SavedBoard {
  id: string;
  timestamp: string;
  letters: TileLabel[];
  gridSize: number;
  words: SolvedWord[];
  totalPoints: number;
  wordCount: number;
}

export type AppAction =
  | { type: "START_PROCESSING"; image: Blob }
  | {
      type: "PROCESSING_COMPLETE";
      analysis: AnalysisData;
      solution: SolveResult;
      timing: TimingData;
    }
  | { type: "ANALYZE_ERROR"; error: string }
  | { type: "CANCEL_PROCESSING" }
  | { type: "SELECT_WORD"; index: number }
  | { type: "DESELECT_WORD" }
  | { type: "TOGGLE_OVERLAY" }
  | { type: "ENTER_EDIT" }
  | { type: "EDIT_TILE"; tileIndex: number; label: TileLabel }
  | { type: "EXIT_EDIT"; solution: SolveResult }
  | { type: "SET_SORT"; sortBy: SortOption }
  | { type: "SET_MIN_POINTS"; minPoints: number }
  | { type: "CLEAR" }
  | { type: "LOAD_BOARD"; board: SavedBoard };

export interface AppState {
  screen: AppScreen;
  capturedImage: Blob | null;
  analysis: AnalysisData | null;
  solution: SolveResult | null;
  timing: TimingData | null;
  selectedWordIndex: number | null;
  isEditMode: boolean;
  editedLetters: TileLabel[] | null;
  letterOverlayVisible: boolean;
  sortBy: SortOption;
  minPointsFilter: number;
}
