"use client";

import { CONFIDENCE_THRESHOLDS } from "@/lib/constants";
import type { SolvedWord, TileLabel } from "@/lib/types";
import { cn } from "@/lib/utils";

interface BoardDisplayProps {
  letters: TileLabel[];
  gridSize: number;
  confidences: number[];
  selectedWord: SolvedWord | null;
  showLetters: boolean;
  showPhoto: boolean;
  boardImageSrc: string | null;
  isEditMode: boolean;
  onTileTap: (index: number) => void;
}

export function BoardDisplay({
  letters,
  gridSize,
  confidences,
  selectedWord,
  showLetters,
  showPhoto,
  boardImageSrc,
  isEditMode,
  onTileTap,
}: BoardDisplayProps) {
  const highlightedTiles = new Set(selectedWord?.path ?? []);
  const hasSelection = selectedWord !== null;

  return (
    <div className="px-2 pt-2">
      <div className="relative mx-auto" style={{ maxWidth: `${gridSize * 56}px` }}>
        {/* Board photo background */}
        {showPhoto && boardImageSrc && (
          // biome-ignore lint/performance/noImgElement: base64 data URI, no optimization possible
          <img
            src={boardImageSrc}
            alt=""
            className="absolute inset-0 w-full h-full rounded-md object-cover pointer-events-none"
          />
        )}

        {/* Tile grid */}
        <div
          className="grid gap-1 relative"
          style={{
            gridTemplateColumns: `repeat(${gridSize}, 1fr)`,
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
                  // Background + text color
                  showPhoto
                    ? isBlock
                      ? "bg-muted/50"
                      : "bg-transparent text-white drop-shadow-[0_1px_2px_rgba(0,0,0,0.8)]"
                    : isBlock
                      ? "bg-muted"
                      : "bg-amber-50 border border-amber-200 text-stone-800",
                  // Confidence indicators
                  !isBlock && confidence < CONFIDENCE_THRESHOLDS.LOW && "ring-2 ring-red-400",
                  !isBlock &&
                    confidence >= CONFIDENCE_THRESHOLDS.LOW &&
                    confidence < CONFIDENCE_THRESHOLDS.MEDIUM &&
                    "ring-2 ring-amber-400",
                  // Path highlighting
                  isInPath && showPhoto && "ring-2 ring-blue-500 bg-blue-500/20 z-10",
                  isInPath && !showPhoto && "ring-2 ring-blue-500 bg-blue-50 z-10",
                  isDimmed && "opacity-30",
                  // Edit mode
                  isEditMode && !isBlock && "cursor-pointer hover:ring-2 hover:ring-primary",
                  // Transitions
                  "duration-200 ease-in-out",
                )}
              >
                {/* Letter */}
                {showLetters && !isBlock && (
                  <span className={cn("leading-none", letter.length > 1 && "text-sm")}>
                    {letter}
                  </span>
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
    </div>
  );
}
