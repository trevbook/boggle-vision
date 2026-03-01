"use client";

import { Slider } from "@/components/ui/slider";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { SortOption } from "@/lib/types";

interface WordListControlsProps {
  sortBy: SortOption;
  onSortChange: (sort: SortOption) => void;
  minPoints: number;
  maxPoints: number;
  onMinPointsChange: (min: number) => void;
}

export function WordListControls({
  sortBy,
  onSortChange,
  minPoints,
  maxPoints,
  onMinPointsChange,
}: WordListControlsProps) {
  return (
    <div className="flex items-center gap-3 px-3 py-2 border-b">
      <ToggleGroup
        type="single"
        size="sm"
        value={sortBy}
        onValueChange={(v) => {
          if (v) onSortChange(v as SortOption);
        }}
        className="shrink-0"
      >
        <ToggleGroupItem value="length" className="text-xs px-2">
          Length
        </ToggleGroupItem>
        <ToggleGroupItem value="points" className="text-xs px-2">
          Points
        </ToggleGroupItem>
        <ToggleGroupItem value="alpha" className="text-xs px-2">
          A-Z
        </ToggleGroupItem>
      </ToggleGroup>

      <div className="flex items-center gap-2 flex-1 min-w-0">
        <span className="text-xs text-muted-foreground whitespace-nowrap">{minPoints}+ pts</span>
        <Slider
          min={1}
          max={maxPoints}
          step={1}
          value={[minPoints]}
          onValueChange={([v]) => onMinPointsChange(v)}
          className="flex-1"
        />
      </div>
    </div>
  );
}
