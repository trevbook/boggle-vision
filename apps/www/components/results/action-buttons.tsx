import { Check, Pencil, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";

interface ActionButtonsProps {
  isEditMode: boolean;
  onEdit: () => void;
  onDoneEdit: () => void;
  onClear: () => void;
}

export function ActionButtons({ isEditMode, onEdit, onDoneEdit, onClear }: ActionButtonsProps) {
  return (
    <div className="flex gap-2 px-3 py-2 border-t">
      {isEditMode ? (
        <Button className="flex-1 gap-2" onClick={onDoneEdit}>
          <Check className="h-4 w-4" />
          Done
        </Button>
      ) : (
        <>
          <Button variant="outline" className="flex-1 gap-2" onClick={onEdit}>
            <Pencil className="h-4 w-4" />
            Edit
          </Button>
          <Button variant="outline" className="flex-1 gap-2" onClick={onClear}>
            <Trash2 className="h-4 w-4" />
            Clear
          </Button>
        </>
      )}
    </div>
  );
}
