"use client";

import { ScrollArea } from "@/components/ui/scroll-area";
import type { SolvedWord } from "@/lib/types";

import { WordRow } from "./word-row";

interface WordListProps {
  words: SolvedWord[];
  selectedIndex: number | null;
  onSelectWord: (index: number) => void;
}

export function WordList({ words, selectedIndex, onSelectWord }: WordListProps) {
  if (words.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <p className="text-sm text-muted-foreground">No words match this filter.</p>
      </div>
    );
  }

  return (
    <ScrollArea className="flex-1">
      <div className="divide-y">
        {words.map((w, i) => (
          <WordRow
            key={w.word}
            word={w.word}
            length={w.length}
            points={w.points}
            isSelected={i === selectedIndex}
            onSelect={() => onSelectWord(i)}
          />
        ))}
      </div>
    </ScrollArea>
  );
}
