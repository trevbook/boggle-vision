"use client";

import { useCallback, useMemo, useState } from "react";

import { useSolver } from "@/hooks/use-solver";
import type {
  AnalysisData,
  AppAction,
  BoardDisplayMode,
  SolvedWord,
  SolveResult,
  SortOption,
  TileLabel,
} from "@/lib/types";

import { ActionButtons } from "./action-buttons";
import { BoardDisplay } from "./board-display";
import { StatsBar } from "./stats-bar";
import { TileEditDrawer } from "./tile-edit-drawer";
import { WordDetailBar } from "./word-detail-bar";
import { WordList } from "./word-list";
import { WordListControls } from "./word-list-controls";

interface ResultsScreenProps {
  analysis: AnalysisData;
  solution: SolveResult;
  editedLetters: TileLabel[] | null;
  selectedWordIndex: number | null;
  isEditMode: boolean;
  boardImage: string | null;
  boardDisplayMode: BoardDisplayMode;
  sortBy: SortOption;
  minPointsFilter: number;
  dispatch: React.Dispatch<AppAction>;
}

function sortWords(words: readonly SolvedWord[], sortBy: SortOption): SolvedWord[] {
  const sorted = [...words];
  switch (sortBy) {
    case "length":
      return sorted.sort((a, b) => b.length - a.length || a.word.localeCompare(b.word));
    case "points":
      return sorted.sort((a, b) => b.points - a.points || b.length - a.length);
    case "alpha":
      return sorted.sort((a, b) => a.word.localeCompare(b.word));
    default:
      return sorted;
  }
}

export function ResultsScreen({
  analysis,
  solution,
  editedLetters,
  selectedWordIndex,
  isEditMode,
  boardImage,
  boardDisplayMode,
  sortBy,
  minPointsFilter,
  dispatch,
}: ResultsScreenProps) {
  const { solveSync } = useSolver();
  const [editingTileIndex, setEditingTileIndex] = useState<number | null>(null);

  const activeLetters = editedLetters ?? analysis.letters;

  const displayedWords = useMemo(
    () =>
      sortWords(
        solution.words.filter((w) => w.points >= minPointsFilter),
        sortBy,
      ),
    [solution.words, minPointsFilter, sortBy],
  );

  const filteredTotalPoints = useMemo(
    () => displayedWords.reduce((sum, w) => sum + w.points, 0),
    [displayedWords],
  );

  const maxPoints = useMemo(
    () => Math.max(1, ...solution.words.map((w) => w.points)),
    [solution.words],
  );

  const selectedWord: SolvedWord | null =
    selectedWordIndex !== null ? (displayedWords[selectedWordIndex] ?? null) : null;

  const handlePrev = useCallback(() => {
    if (selectedWordIndex === null || displayedWords.length === 0) return;
    const prev = (selectedWordIndex - 1 + displayedWords.length) % displayedWords.length;
    dispatch({ type: "SELECT_WORD", index: prev });
  }, [selectedWordIndex, displayedWords.length, dispatch]);

  const handleNext = useCallback(() => {
    if (selectedWordIndex === null || displayedWords.length === 0) return;
    const next = (selectedWordIndex + 1) % displayedWords.length;
    dispatch({ type: "SELECT_WORD", index: next });
  }, [selectedWordIndex, displayedWords.length, dispatch]);

  const handleTileTap = useCallback(
    (index: number) => {
      if (isEditMode) {
        setEditingTileIndex(index);
      }
    },
    [isEditMode],
  );

  const handleTileSelect = useCallback(
    (label: TileLabel) => {
      if (editingTileIndex !== null) {
        dispatch({ type: "EDIT_TILE", tileIndex: editingTileIndex, label });
        setEditingTileIndex(null);
      }
    },
    [editingTileIndex, dispatch],
  );

  const handleDoneEdit = useCallback(() => {
    const result = solveSync(editedLetters ?? analysis.letters, analysis.gridSize);
    if (result) {
      dispatch({ type: "EXIT_EDIT", solution: result });
    }
  }, [solveSync, editedLetters, analysis, dispatch]);

  return (
    <div className="flex flex-col h-dvh">
      {/* Board */}
      {(() => {
        const effectiveMode = isEditMode ? "letters" : boardDisplayMode;
        const showPhoto =
          (effectiveMode === "photo" || effectiveMode === "photo-only") && boardImage !== null;
        const showLetters = effectiveMode === "photo" || effectiveMode === "letters";
        const boardImageSrc = boardImage ? `data:image/jpeg;base64,${boardImage}` : null;

        const toggleLabel =
          effectiveMode === "photo"
            ? "Show grid"
            : effectiveMode === "letters"
              ? "Hide letters"
              : boardImage
                ? "Show photo"
                : "Show letters";

        return (
          <>
            <BoardDisplay
              letters={activeLetters}
              gridSize={analysis.gridSize}
              confidences={analysis.confidences}
              selectedWord={selectedWord}
              showLetters={showLetters}
              showPhoto={showPhoto}
              boardImageSrc={boardImageSrc}
              isEditMode={isEditMode}
              onTileTap={handleTileTap}
            />

            {/* Toggle display mode button */}
            {!isEditMode && (
              <div className="flex justify-center py-1">
                <button
                  type="button"
                  onClick={() => dispatch({ type: "CYCLE_BOARD_DISPLAY" })}
                  className="text-xs text-muted-foreground underline underline-offset-2"
                >
                  {toggleLabel}
                </button>
              </div>
            )}
          </>
        );
      })()}

      {/* Stats */}
      <StatsBar totalPoints={filteredTotalPoints} wordCount={displayedWords.length} />

      {/* Controls */}
      {!isEditMode && (
        <WordListControls
          sortBy={sortBy}
          onSortChange={(s) => dispatch({ type: "SET_SORT", sortBy: s })}
          minPoints={minPointsFilter}
          maxPoints={maxPoints}
          onMinPointsChange={(m) => dispatch({ type: "SET_MIN_POINTS", minPoints: m })}
        />
      )}

      {/* Word list */}
      {!isEditMode && (
        <WordList
          words={displayedWords}
          selectedIndex={selectedWordIndex}
          onSelectWord={(i) => dispatch({ type: "SELECT_WORD", index: i })}
        />
      )}

      {/* Edit mode spacer */}
      {isEditMode && (
        <div className="flex-1 flex items-center justify-center p-4">
          <p className="text-sm text-muted-foreground">Tap a tile to change it</p>
        </div>
      )}

      {/* Word detail bar */}
      {selectedWord && !isEditMode && (
        <WordDetailBar
          word={selectedWord}
          currentIndex={selectedWordIndex ?? 0}
          totalWords={displayedWords.length}
          onPrev={handlePrev}
          onNext={handleNext}
          onDeselect={() => dispatch({ type: "DESELECT_WORD" })}
        />
      )}

      {/* Action buttons */}
      <ActionButtons
        isEditMode={isEditMode}
        onEdit={() => dispatch({ type: "ENTER_EDIT" })}
        onDoneEdit={handleDoneEdit}
        onClear={() => dispatch({ type: "CLEAR" })}
      />

      {/* Tile edit drawer */}
      <TileEditDrawer
        open={editingTileIndex !== null}
        currentLabel={editingTileIndex !== null ? activeLetters[editingTileIndex] : null}
        onSelect={handleTileSelect}
        onClose={() => setEditingTileIndex(null)}
      />
    </div>
  );
}
