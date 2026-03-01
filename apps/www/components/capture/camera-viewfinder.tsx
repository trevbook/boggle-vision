"use client";

import { X } from "lucide-react";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { useCamera } from "@/hooks/use-camera";

interface CameraViewfinderProps {
  onCapture: (blob: Blob) => void;
  onClose: () => void;
}

export function CameraViewfinder({ onCapture, onClose }: CameraViewfinderProps) {
  const { videoRef, isActive, error, start, stop, capture } = useCamera();

  useEffect(() => {
    start();
    return () => stop();
  }, [start, stop]);

  const handleCapture = async () => {
    const blob = await capture();
    if (blob) {
      stop();
      onCapture(blob);
    }
  };

  if (error) {
    return (
      <div className="fixed inset-0 z-50 bg-black flex flex-col items-center justify-center gap-4 text-white px-6">
        <p className="text-center text-sm">
          Camera access denied. Please allow camera permissions or upload a photo instead.
        </p>
        <Button variant="secondary" onClick={onClose}>
          Go Back
        </Button>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 bg-black flex flex-col">
      {/* Video feed */}
      <div className="relative flex-1 overflow-hidden">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="absolute inset-0 w-full h-full object-cover"
        />

        {/* Alignment guide */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="w-[75vw] max-w-[320px] aspect-square border-2 border-white/40 rounded-2xl" />
        </div>

        {/* Instruction text */}
        <div className="absolute bottom-20 left-0 right-0 text-center pointer-events-none">
          <p className="text-white/60 text-sm">Center your board in the frame</p>
        </div>

        {/* Close button */}
        <button
          type="button"
          onClick={() => {
            stop();
            onClose();
          }}
          className="absolute top-4 right-4 p-2 rounded-full bg-black/40 text-white"
        >
          <X className="h-6 w-6" />
        </button>
      </div>

      {/* Capture button */}
      <div className="flex justify-center py-6 bg-black">
        <button
          type="button"
          onClick={handleCapture}
          disabled={!isActive}
          className="h-18 w-18 rounded-full border-4 border-white bg-white/20 active:bg-white/40 disabled:opacity-50 transition-colors"
          aria-label="Take photo"
        />
      </div>
    </div>
  );
}
