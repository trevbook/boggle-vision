import type { AppAction, AppState, BoardDisplayMode } from "./types";

export const initialState: AppState = {
  screen: "capture",
  capturedImage: null,
  analysis: null,
  solution: null,
  timing: null,
  selectedWordIndex: null,
  isEditMode: false,
  editedLetters: null,
  boardImage: null,
  boardDisplayMode: "photo",
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
        boardImage: action.boardImage,
        boardDisplayMode: action.boardImage ? "photo" : "letters",
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

    case "CYCLE_BOARD_DISPLAY": {
      const hasBoardImage = state.boardImage !== null;
      const modes: BoardDisplayMode[] = hasBoardImage
        ? ["photo", "letters", "photo-only"]
        : ["letters", "photo-only"];
      const currentIndex = modes.indexOf(state.boardDisplayMode);
      const nextIndex = (currentIndex + 1) % modes.length;
      return { ...state, boardDisplayMode: modes[nextIndex] };
    }

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
        boardImage: null,
        boardDisplayMode: "letters",
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
