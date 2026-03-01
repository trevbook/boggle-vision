"use client";

import { useCallback, useEffect, useState } from "react";

import { MAX_SAVED_BOARDS } from "@/lib/constants";
import type { SavedBoard } from "@/lib/types";

const STORAGE_KEY = "boggle-vision-boards";

function loadBoards(): SavedBoard[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export function useBoardHistory() {
  const [boards, setBoards] = useState<SavedBoard[]>([]);

  useEffect(() => {
    setBoards(loadBoards());
  }, []);

  const saveBoard = useCallback((board: SavedBoard) => {
    setBoards((prev) => {
      const next = [board, ...prev].slice(0, MAX_SAVED_BOARDS);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  const deleteBoard = useCallback((id: string) => {
    setBoards((prev) => {
      const next = prev.filter((b) => b.id !== id);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  }, []);

  return { boards, saveBoard, deleteBoard };
}
