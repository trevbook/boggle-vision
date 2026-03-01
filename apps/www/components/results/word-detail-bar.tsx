"use client";

import { ChevronLeft, ChevronRight, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import type { SolvedWord } from "@/lib/types";

interface WordDetailBarProps {
  word: SolvedWord;
  currentIndex: number;
  totalWords: number;
  onPrev: () => void;
  onNext: () => void;
  onDeselect: () => void;
}

export function WordDetailBar({
  word,
  currentIndex,
  totalWords,
  onPrev,
  onNext,
  onDeselect,
}: WordDetailBarProps) {
  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-primary text-primary-foreground">
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8 shrink-0 text-primary-foreground hover:text-primary-foreground/80 hover:bg-primary-foreground/10"
        onClick={onPrev}
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>

      <div className="flex-1 text-center min-w-0">
        <p className="font-bold text-lg truncate">{word.word}</p>
        <p className="text-xs opacity-80">
          {word.length} letters &middot; {word.points} pts &middot; {currentIndex + 1}/{totalWords}
        </p>
      </div>

      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8 shrink-0 text-primary-foreground hover:text-primary-foreground/80 hover:bg-primary-foreground/10"
        onClick={onNext}
      >
        <ChevronRight className="h-4 w-4" />
      </Button>

      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8 shrink-0 text-primary-foreground hover:text-primary-foreground/80 hover:bg-primary-foreground/10"
        onClick={onDeselect}
      >
        <X className="h-4 w-4" />
      </Button>
    </div>
  );
}
