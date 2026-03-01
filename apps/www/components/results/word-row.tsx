import { cn } from "@/lib/utils";

interface WordRowProps {
  word: string;
  length: number;
  points: number;
  isSelected: boolean;
  onSelect: () => void;
}

export function WordRow({ word, length, points, isSelected, onSelect }: WordRowProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "w-full flex items-center justify-between px-3 py-2 text-left transition-colors",
        isSelected
          ? "bg-primary/10 border-l-2 border-primary"
          : "hover:bg-accent border-l-2 border-transparent",
      )}
    >
      <span className="font-semibold">{word}</span>
      <span className="text-xs text-muted-foreground tabular-nums shrink-0 ml-2">
        {length} letters &middot; {points} pts
      </span>
    </button>
  );
}
