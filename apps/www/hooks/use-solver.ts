"use client";

import { buildTrie, type SolveResult, solveBoard, type TrieNode } from "@boggle-vision/solver";
import { useCallback, useRef } from "react";

import { reshapeToGrid } from "@/lib/board-utils";
import type { TileLabel } from "@/lib/types";

/** Module-level singleton so the trie is built at most once across re-renders. */
let cachedTrie: { root: TrieNode; wordCount: number } | null = null;

async function loadTrie() {
  if (cachedTrie) return cachedTrie;
  const resp = await fetch("/enable1.txt");
  const text = await resp.text();
  cachedTrie = buildTrie(text);
  return cachedTrie;
}

export function useSolver() {
  const trieRef = useRef(cachedTrie);

  /** Async solve — loads the trie if it hasn't been loaded yet. */
  const solve = useCallback(
    async (
      letters: TileLabel[],
      gridSize: number,
    ): Promise<{ result: SolveResult; solverMs: number }> => {
      if (!trieRef.current) {
        trieRef.current = await loadTrie();
      }
      const board = reshapeToGrid(letters, gridSize);
      const t0 = performance.now();
      const result = solveBoard(board, trieRef.current.root);
      const solverMs = Math.round(performance.now() - t0);
      return { result, solverMs };
    },
    [],
  );

  /** Sync solve — only works after the trie has been loaded by a prior `solve()` call. */
  const solveSync = useCallback((letters: TileLabel[], gridSize: number): SolveResult | null => {
    if (!trieRef.current) return null;
    const board = reshapeToGrid(letters, gridSize);
    return solveBoard(board, trieRef.current.root);
  }, []);

  return { solve, solveSync };
}
