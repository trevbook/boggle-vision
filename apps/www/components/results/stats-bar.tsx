import { Separator } from "@/components/ui/separator";

interface StatsBarProps {
  totalPoints: number;
  wordCount: number;
}

export function StatsBar({ totalPoints, wordCount }: StatsBarProps) {
  return (
    <div className="flex items-center justify-center gap-3 py-2 text-sm">
      <span className="font-bold text-lg tabular-nums">{totalPoints}</span>
      <span className="text-muted-foreground">pts</span>
      <Separator orientation="vertical" className="h-4" />
      <span className="font-bold text-lg tabular-nums">{wordCount}</span>
      <span className="text-muted-foreground">words</span>
    </div>
  );
}
