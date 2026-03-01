"use client";

import { CONFIDENCE_THRESHOLDS } from "@/lib/constants";
import type { SolvedWord, TileLabel } from "@/lib/types";
import { cn } from "@/lib/utils";

interface BoardDisplayProps {
  letters: TileLabel[];
  gridSize: number;
  confidences: number[];
  selectedWord: SolvedWord | null;
  letterOverlayVisible: boolean;
  isEditMode: boolean;
  onTileTap: (index: number) => void;
}

export function BoardDisplay({
  letters,
  gridSize,
  confidences,
  selectedWord,
  letterOverlayVisible,
  isEditMode,
  onTileTap,
}: BoardDisplayProps) {
  const highlightedTiles = new Set(selectedWord?.path ?? []);
  const hasSelection = selectedWord !== null;

  return (
    <div className="px-2 pt-2">
      <div
        className="grid gap-1 mx-auto"
        style={{
          gridTemplateColumns: `repeat(${gridSize}, 1fr)`,
          maxWidth: `${gridSize * 56}px`,
        }}
      >
        {letters.map((letter, i) => {
          const confidence = confidences[i];
          const isInPath = highlightedTiles.has(i);
          const pathOrder = selectedWord?.path.indexOf(i) ?? -1;
          const isDimmed = hasSelection && !isInPath;
          const isBlock = letter === "BLOCK";
          const row = Math.floor(i / gridSize);
          const col = i % gridSize;

          return (
            <button
              key={`${row}-${col}`}
              type="button"
              onClick={() => onTileTap(i)}
              disabled={!isEditMode && !isBlock}
              className={cn(
                "relative aspect-square rounded-md flex items-center justify-center font-bold text-lg transition-all select-none",
                isBlock ? "bg-muted" : "bg-amber-50 border border-amber-200 text-stone-800",
                // Confidence indicators
                !isBlock && confidence < CONFIDENCE_THRESHOLDS.LOW && "ring-2 ring-red-400",
                !isBlock &&
                  confidence >= CONFIDENCE_THRESHOLDS.LOW &&
                  confidence < CONFIDENCE_THRESHOLDS.MEDIUM &&
                  "ring-2 ring-amber-400",
                // Path highlighting
                isInPath && "ring-2 ring-blue-500 bg-blue-50 z-10",
                isDimmed && "opacity-30",
                // Edit mode
                isEditMode && !isBlock && "cursor-pointer hover:ring-2 hover:ring-primary",
                // Transitions
                "duration-200 ease-in-out",
              )}
            >
              {/* Letter */}
              {letterOverlayVisible && !isBlock && (
                <span className={cn("leading-none", letter.length > 1 && "text-sm")}>{letter}</span>
              )}

              {/* Path order number */}
              {isInPath && pathOrder >= 0 && (
                <span className="absolute -top-1 -right-1 bg-blue-500 text-white text-[10px] font-bold rounded-full w-4 h-4 flex items-center justify-center z-20">
                  {pathOrder + 1}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
