"use client";

import { useEffect, useState } from "react";

import { QUIP_INTERVAL_MS } from "@/lib/constants";
import { LOADING_QUIPS } from "@/lib/quips";

export function LoadingQuip() {
  const [index, setIndex] = useState(() => Math.floor(Math.random() * LOADING_QUIPS.length));

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % LOADING_QUIPS.length);
    }, QUIP_INTERVAL_MS);
    return () => clearInterval(interval);
  }, []);

  return (
    <p className="text-sm text-muted-foreground animate-pulse text-center min-h-[1.25rem]">
      {LOADING_QUIPS[index]}
    </p>
  );
}
