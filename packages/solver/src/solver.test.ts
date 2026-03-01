import { beforeAll, describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { solveBoard } from "./solver.js";
import type { TrieNode } from "./trie.js";
import { buildTrie } from "./trie.js";

const DICT_PATH = resolve(import.meta.dir, "../data/enable1.txt");

let trieRoot: TrieNode;

beforeAll(() => {
  const wordList = readFileSync(DICT_PATH, "utf-8");
  const { root } = buildTrie(wordList);
  trieRoot = root;
});

describe("solveBoard", () => {
  // Real board from labeled-boards.csv (easy-01):
  // Y  G  R  L  H  N
  // E  T  T  N  T  O
  // Th F  E  E  E  N
  // C  L  E  T  H  O
  // J  R  T  R  E  L
  // D  I  T  H  E  J
  const BOARD_EASY_01: string[][] = [
    ["Y", "G", "R", "L", "H", "N"],
    ["E", "T", "T", "N", "T", "O"],
    ["Th", "F", "E", "E", "E", "N"],
    ["C", "L", "E", "T", "H", "O"],
    ["J", "R", "T", "R", "E", "L"],
    ["D", "I", "T", "H", "E", "J"],
  ];

  test("finds words on a real 6x6 board", () => {
    const result = solveBoard(BOARD_EASY_01, trieRoot);

    expect(result.words.length).toBeGreaterThan(50);
    expect(result.totalPoints).toBeGreaterThan(0);
  });

  test("all found words have valid lengths and scores", () => {
    const result = solveBoard(BOARD_EASY_01, trieRoot);

    for (const word of result.words) {
      expect(word.length).toBeGreaterThanOrEqual(4);
      expect(word.points).toBeGreaterThan(0);
      expect(word.word.length).toBe(word.length);
    }
  });

  test("words are sorted by length desc then alphabetically", () => {
    const result = solveBoard(BOARD_EASY_01, trieRoot);

    for (let i = 1; i < result.words.length; i++) {
      const prev = result.words[i - 1];
      const curr = result.words[i];
      if (prev.length === curr.length) {
        expect(prev.word.localeCompare(curr.word)).toBeLessThanOrEqual(0);
      } else {
        expect(prev.length).toBeGreaterThan(curr.length);
      }
    }
  });

  test("totalPoints equals sum of individual word points", () => {
    const result = solveBoard(BOARD_EASY_01, trieRoot);
    const sum = result.words.reduce((acc, w) => acc + w.points, 0);
    expect(result.totalPoints).toBe(sum);
  });

  test("paths have valid tile indices", () => {
    const result = solveBoard(BOARD_EASY_01, trieRoot);
    const maxIdx = 6 * 6 - 1;

    for (const word of result.words) {
      // Path length should match tile count (digraphs count as 1 tile)
      expect(word.path.length).toBeGreaterThanOrEqual(1);
      for (const idx of word.path) {
        expect(idx).toBeGreaterThanOrEqual(0);
        expect(idx).toBeLessThanOrEqual(maxIdx);
      }
      // No repeated tiles in a path
      const unique = new Set(word.path);
      expect(unique.size).toBe(word.path.length);
    }
  });

  test("finds specific expected words", () => {
    const result = solveBoard(BOARD_EASY_01, trieRoot);
    const wordSet = new Set(result.words.map((w) => w.word));

    // These words should be findable on this board via adjacency
    expect(wordSet.has("teen")).toBe(true);
    expect(wordSet.has("feet")).toBe(true);
    expect(wordSet.has("tree")).toBe(true);
  });

  test("handles BLOCK tiles", () => {
    const boardWithBlocks: string[][] = [
      ["T", "E", "S", "T"],
      ["BLOCK", "BLOCK", "BLOCK", "BLOCK"],
      ["W", "O", "R", "D"],
      ["BLOCK", "BLOCK", "BLOCK", "BLOCK"],
    ];

    const result = solveBoard(boardWithBlocks, trieRoot);

    // "test" should be found (top row), but no words should cross the BLOCK row
    const wordSet = new Set(result.words.map((w) => w.word));
    expect(wordSet.has("test")).toBe(true);
    expect(wordSet.has("word")).toBe(true);

    // No path should include indices from BLOCK rows (row 1: 4,5,6,7 or row 3: 12,13,14,15)
    const blockIndices = new Set([4, 5, 6, 7, 12, 13, 14, 15]);
    for (const word of result.words) {
      for (const idx of word.path) {
        expect(blockIndices.has(idx)).toBe(false);
      }
    }
  });

  test("handles small 4x4 board", () => {
    const board: string[][] = [
      ["T", "E", "S", "T"],
      ["A", "R", "E", "A"],
      ["L", "I", "N", "E"],
      ["S", "T", "A", "R"],
    ];

    const result = solveBoard(board, trieRoot);
    expect(result.words.length).toBeGreaterThan(10);
  });

  test("returns empty for impossible board", () => {
    const board: string[][] = [
      ["Z", "Z"],
      ["Z", "Z"],
    ];

    const result = solveBoard(board, trieRoot);
    expect(result.words).toHaveLength(0);
    expect(result.totalPoints).toBe(0);
  });

  test("digraph tiles work in solver", () => {
    // Board designed so "queen" could be spelled
    const board: string[][] = [
      ["Qu", "E", "X", "X"],
      ["X", "E", "X", "X"],
      ["X", "N", "X", "X"],
      ["X", "X", "X", "X"],
    ];

    const result = solveBoard(board, trieRoot);
    const wordSet = new Set(result.words.map((w) => w.word));
    expect(wordSet.has("queen")).toBe(true);
  });
});
