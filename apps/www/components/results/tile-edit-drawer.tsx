"use client";

import { Drawer, DrawerContent, DrawerHeader, DrawerTitle } from "@/components/ui/drawer";
import { DIGRAPHS, SINGLE_LETTERS } from "@/lib/constants";
import type { TileLabel } from "@/lib/types";
import { cn } from "@/lib/utils";

interface TileEditDrawerProps {
  open: boolean;
  currentLabel: TileLabel | null;
  onSelect: (label: TileLabel) => void;
  onClose: () => void;
}

export function TileEditDrawer({ open, currentLabel, onSelect, onClose }: TileEditDrawerProps) {
  return (
    <Drawer
      open={open}
      onOpenChange={(o) => {
        if (!o) onClose();
      }}
    >
      <DrawerContent>
        <DrawerHeader>
          <DrawerTitle>Select tile</DrawerTitle>
        </DrawerHeader>

        <div className="px-4 pb-6 space-y-4">
          {/* Single letters */}
          <div className="grid grid-cols-7 gap-1.5">
            {SINGLE_LETTERS.map((label) => (
              <TileOption
                key={label}
                label={label}
                isSelected={label === currentLabel}
                onSelect={() => onSelect(label)}
              />
            ))}
          </div>

          {/* Digraphs + BLOCK */}
          <div className="grid grid-cols-7 gap-1.5">
            {DIGRAPHS.map((label) => (
              <TileOption
                key={label}
                label={label}
                isSelected={label === currentLabel}
                onSelect={() => onSelect(label)}
              />
            ))}
            <TileOption
              label="BLOCK"
              isSelected={currentLabel === "BLOCK"}
              onSelect={() => onSelect("BLOCK")}
            />
          </div>
        </div>
      </DrawerContent>
    </Drawer>
  );
}

function TileOption({
  label,
  isSelected,
  onSelect,
}: {
  label: TileLabel;
  isSelected: boolean;
  onSelect: () => void;
}) {
  const isBlock = label === "BLOCK";

  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        "aspect-square rounded-md flex items-center justify-center font-bold text-sm transition-all",
        isBlock
          ? "bg-muted text-muted-foreground text-[10px]"
          : "bg-amber-50 border border-amber-200 text-stone-800",
        isSelected && "ring-2 ring-primary",
      )}
    >
      {isBlock ? "BLK" : label}
    </button>
  );
}
