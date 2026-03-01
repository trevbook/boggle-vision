"use client";

import { Camera } from "lucide-react";
import { useState } from "react";

import { AppHeader } from "@/components/app-header";
import { Button } from "@/components/ui/button";
import type { SavedBoard } from "@/lib/types";

import { CameraViewfinder } from "./camera-viewfinder";
import { UploadButton } from "./upload-button";

interface CaptureScreenProps {
  onCapture: (image: Blob) => void;
  savedBoards: SavedBoard[];
  onLoadBoard: (board: SavedBoard) => void;
}

export function CaptureScreen({ onCapture, savedBoards, onLoadBoard }: CaptureScreenProps) {
  const [showCamera, setShowCamera] = useState(false);

  if (showCamera) {
    return (
      <CameraViewfinder
        onCapture={(blob) => {
          setShowCamera(false);
          onCapture(blob);
        }}
        onClose={() => setShowCamera(false)}
      />
    );
  }

  return (
    <div className="flex flex-col items-center min-h-dvh px-4">
      <AppHeader />

      <div className="flex-1 flex flex-col items-center justify-center gap-4 max-w-sm w-full">
        <Button size="lg" className="w-full h-14 text-lg gap-2" onClick={() => setShowCamera(true)}>
          <Camera className="h-5 w-5" />
          Take Photo
        </Button>

        <UploadButton onUpload={onCapture} />
      </div>

      {savedBoards.length > 0 && (
        <div className="w-full max-w-sm pb-8">
          <h2 className="text-sm font-medium text-muted-foreground mb-3">Recent Boards</h2>
          <div className="space-y-2">
            {savedBoards.map((board) => (
              <button
                key={board.id}
                type="button"
                onClick={() => onLoadBoard(board)}
                className="w-full text-left p-3 rounded-lg border hover:bg-accent transition-colors"
              >
                <div className="flex justify-between items-center">
                  <span className="font-semibold">{board.totalPoints} pts</span>
                  <span className="text-sm text-muted-foreground">
                    {board.wordCount} words &middot; {board.gridSize}&times;{board.gridSize}
                  </span>
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {new Date(board.timestamp).toLocaleDateString()}
                </div>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
