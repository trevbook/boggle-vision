/**
 * DFS-based Boggle board solver.
 *
 * Finds all valid words on a Boggle board by traversing adjacent tiles
 * using depth-first search, pruning paths via trie lookups.
 *
 * Ported from boggle-vision-v0/utils/board_solving.py.
 */

import { scoreWord } from "./scoring";
import type { TrieNode } from "./trie";

/** A word found on the board with its scoring information. */
export interface SolvedWord {
  /** The word as a lowercase string (digraphs expanded, e.g. "queen"). */
  readonly word: string;
  /** Number of characters in the word. */
  readonly length: number;
  /** Boggle point value. */
  readonly points: number;
  /** Tile indices (row-major flat index) forming the word path on the board. */
  readonly path: readonly number[];
}

/** Result of solving a board. */
export interface SolveResult {
  /** All valid words found, sorted by length descending then alphabetically. */
  readonly words: readonly SolvedWord[];
  /** Total points across all words. */
  readonly totalPoints: number;
}

/** 8-directional adjacency offsets (up, down, left, right, and diagonals). */
const DIRECTIONS: readonly (readonly [number, number])[] = [
  [-1, -1],
  [-1, 0],
  [-1, 1],
  [0, -1],
  [0, 1],
  [1, -1],
  [1, 0],
  [1, 1],
];

/**
 * Expand a tile label to its lowercase character representation.
 * Single letters → lowercase. Digraphs → lowercase (e.g. "Qu" → "qu").
 */
function tileLabelToChars(label: string): string {
  return label.toLowerCase();
}

/**
 * Solve a Boggle board: find all valid words of length >= minLength.
 *
 * @param board - 2D array of tile labels (e.g. [["A", "B", ...], ["Qu", "BLOCK", ...], ...]).
 *               "BLOCK" tiles are impassable.
 * @param trieRoot - Root of the word trie (built from dictionary).
 * @param minLength - Minimum word length (default: 4, per standard Boggle rules).
 * @returns SolveResult with all found words and total points.
 */
export function solveBoard(
  board: readonly (readonly string[])[],
  trieRoot: TrieNode,
  minLength = 4,
): SolveResult {
  const rows = board.length;
  const cols = board[0].length;
  const visited: boolean[][] = Array.from({ length: rows }, () => new Array(cols).fill(false));

  // Use a Map to deduplicate words (keeping the first path found)
  const foundWords = new Map<string, readonly number[]>();

  function dfs(x: number, y: number, node: TrieNode, word: string, path: number[]): void {
    visited[x][y] = true;
    const flatIdx = x * cols + y;
    path.push(flatIdx);

    // If this node marks a complete word, record it
    if (node.isEnd && word.length >= minLength && !foundWords.has(word)) {
      foundWords.set(word, [...path]);
    }

    // Explore all 8 neighbors
    for (const [dx, dy] of DIRECTIONS) {
      const nx = x + dx;
      const ny = y + dy;

      if (nx < 0 || nx >= rows || ny < 0 || ny >= cols) continue;
      if (visited[nx][ny]) continue;

      const tile = board[nx][ny];
      if (tile === "BLOCK") continue;

      // Check if this tile's label exists as a child in the trie
      const child = node.children.get(tile);
      if (!child) continue;

      dfs(nx, ny, child, word + tileLabelToChars(tile), path);
    }

    // Backtrack
    path.pop();
    visited[x][y] = false;
  }

  // Start DFS from every non-BLOCK tile
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const tile = board[i][j];
      if (tile === "BLOCK") continue;

      const child = trieRoot.children.get(tile);
      if (!child) continue;

      dfs(i, j, child, tileLabelToChars(tile), []);
    }
  }

  // Build result array, sorted by length desc then alphabetically
  const words: SolvedWord[] = [];
  for (const [word, path] of foundWords) {
    words.push({
      word,
      length: word.length,
      points: scoreWord(word.length),
      path,
    });
  }

  words.sort((a, b) => b.length - a.length || a.word.localeCompare(b.word));

  const totalPoints = words.reduce((sum, w) => sum + w.points, 0);

  return { words, totalPoints };
}
