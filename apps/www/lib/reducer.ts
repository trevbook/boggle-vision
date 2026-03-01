import type { AppAction, AppState } from "./types";

export const initialState: AppState = {
  screen: "capture",
  capturedImage: null,
  analysis: null,
  solution: null,
  timing: null,
  selectedWordIndex: null,
  isEditMode: false,
  editedLetters: null,
  letterOverlayVisible: true,
  sortBy: "length",
  minPointsFilter: 1,
};

export function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "START_PROCESSING":
      return {
        ...initialState,
        screen: "processing",
        capturedImage: action.image,
      };

    case "PROCESSING_COMPLETE":
      return {
        ...state,
        screen: "results",
        analysis: action.analysis,
        solution: action.solution,
        timing: action.timing,
      };

    case "ANALYZE_ERROR":
    case "CANCEL_PROCESSING":
      return { ...initialState };

    case "SELECT_WORD":
      return { ...state, selectedWordIndex: action.index };

    case "DESELECT_WORD":
      return { ...state, selectedWordIndex: null };

    case "TOGGLE_OVERLAY":
      return { ...state, letterOverlayVisible: !state.letterOverlayVisible };

    case "ENTER_EDIT":
      return {
        ...state,
        isEditMode: true,
        editedLetters: state.editedLetters ?? [...(state.analysis?.letters ?? [])],
        selectedWordIndex: null,
      };

    case "EDIT_TILE": {
      const letters = [...(state.editedLetters ?? state.analysis?.letters ?? [])];
      letters[action.tileIndex] = action.label;
      return { ...state, editedLetters: letters };
    }

    case "EXIT_EDIT":
      return {
        ...state,
        isEditMode: false,
        solution: action.solution,
        selectedWordIndex: null,
        minPointsFilter: 1,
      };

    case "SET_SORT":
      return { ...state, sortBy: action.sortBy, selectedWordIndex: null };

    case "SET_MIN_POINTS":
      return { ...state, minPointsFilter: action.minPoints, selectedWordIndex: null };

    case "CLEAR":
      return { ...initialState };

    case "LOAD_BOARD":
      return {
        ...initialState,
        screen: "results",
        analysis: {
          letters: action.board.letters,
          gridSize: action.board.gridSize,
          confidences: action.board.letters.map(() => 1),
          meanConfidence: 1,
          minConfidence: 1,
          detectionConfidence: 1,
        },
        solution: {
          words: action.board.words,
          totalPoints: action.board.totalPoints,
        },
      };

    default:
      return state;
  }
}
