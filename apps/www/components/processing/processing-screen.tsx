"use client";

import { X } from "lucide-react";
import { useMemo } from "react";

import { Button } from "@/components/ui/button";

import { LoadingQuip } from "./loading-quip";

interface ProcessingScreenProps {
  image: Blob;
  onCancel: () => void;
}

export function ProcessingScreen({ image, onCancel }: ProcessingScreenProps) {
  const imageUrl = useMemo(() => URL.createObjectURL(image), [image]);

  return (
    <div className="flex flex-col items-center justify-center min-h-dvh px-4 gap-6">
      {/* Image preview with shimmer overlay */}
      <div className="relative w-full max-w-sm aspect-square rounded-xl overflow-hidden">
        {/* biome-ignore lint/performance/noImgElement: blob URL, can't use next/image */}
        <img src={imageUrl} alt="Captured board" className="w-full h-full object-cover" />
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-shimmer" />
      </div>

      <LoadingQuip />

      <Button variant="ghost" size="sm" onClick={onCancel} className="gap-1">
        <X className="h-4 w-4" />
        Cancel
      </Button>
    </div>
  );
}
