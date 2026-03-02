"use client";

import { useCallback, useEffect, useReducer } from "react";
import { toast } from "sonner";

import { CaptureScreen } from "@/components/capture/capture-screen";
import { ProcessingScreen } from "@/components/processing/processing-screen";
import { ResultsScreen } from "@/components/results/results-screen";
import { useBoardHistory } from "@/hooks/use-board-history";
import { useSolver } from "@/hooks/use-solver";
import { analyzeImage, warmLambda } from "@/lib/api";
import { appReducer, initialState } from "@/lib/reducer";
import type { SavedBoard } from "@/lib/types";

export default function Home() {
  const [state, dispatch] = useReducer(appReducer, initialState);
  const { solve } = useSolver();
  const { boards, saveBoard } = useBoardHistory();

  // Warm Lambda + preload dictionary on mount
  useEffect(() => {
    warmLambda();
  }, []);

  // Processing pipeline: analyze image -> solve board
  useEffect(() => {
    if (state.screen !== "processing" || !state.capturedImage) return;

    const image = state.capturedImage;
    let cancelled = false;

    (async () => {
      try {
        const resp = await analyzeImage(image);
        if (cancelled) return;

        const { result, solverMs } = await solve(resp.analysis.letters, resp.analysis.gridSize);
        if (cancelled) return;

        dispatch({
          type: "PROCESSING_COMPLETE",
          analysis: resp.analysis,
          boardImage: resp.boardImage ?? null,
          solution: result,
          timing: { pipelineMs: resp.timing.pipelineMs, solverMs },
        });
      } catch (err) {
        if (!cancelled) {
          toast.error(err instanceof Error ? err.message : "Something went wrong");
          dispatch({ type: "ANALYZE_ERROR", error: String(err) });
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [state.screen, state.capturedImage, solve]);

  // Save board to history when clearing
  const handleClear = useCallback(() => {
    if (state.analysis && state.solution) {
      const board: SavedBoard = {
        id: crypto.randomUUID(),
        timestamp: new Date().toISOString(),
        letters: state.editedLetters ?? state.analysis.letters,
        gridSize: state.analysis.gridSize,
        words: [...state.solution.words],
        totalPoints: state.solution.totalPoints,
        wordCount: state.solution.words.length,
      };
      saveBoard(board);
    }
    dispatch({ type: "CLEAR" });
  }, [state.analysis, state.solution, state.editedLetters, saveBoard]);

  switch (state.screen) {
    case "capture":
      return (
        <CaptureScreen
          onCapture={(image) => dispatch({ type: "START_PROCESSING", image })}
          savedBoards={boards}
          onLoadBoard={(board) => dispatch({ type: "LOAD_BOARD", board })}
        />
      );

    case "processing": {
      if (!state.capturedImage) return null;
      return (
        <ProcessingScreen
          image={state.capturedImage}
          onCancel={() => dispatch({ type: "CANCEL_PROCESSING" })}
        />
      );
    }

    case "results": {
      if (!state.analysis || !state.solution) return null;
      return (
        <ResultsScreen
          analysis={state.analysis}
          solution={state.solution}
          editedLetters={state.editedLetters}
          selectedWordIndex={state.selectedWordIndex}
          isEditMode={state.isEditMode}
          boardImage={state.boardImage}
          boardDisplayMode={state.boardDisplayMode}
          sortBy={state.sortBy}
          minPointsFilter={state.minPointsFilter}
          dispatch={(action) => {
            if (action.type === "CLEAR") {
              handleClear();
            } else {
              dispatch(action);
            }
          }}
        />
      );
    }
  }
}
