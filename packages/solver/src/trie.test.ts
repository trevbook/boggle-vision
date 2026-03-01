import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { _wordToTileKeys, buildTrie, lookupTileKeys, lookupWord } from "./trie.js";

const DICT_PATH = resolve(import.meta.dir, "../data/enable1.txt");
const wordList = readFileSync(DICT_PATH, "utf-8");

describe("wordToTileKeys", () => {
  test("decomposes simple word", () => {
    expect(_wordToTileKeys("test")).toEqual(["T", "E", "S", "T"]);
  });

  test("decomposes word with qu digraph", () => {
    expect(_wordToTileKeys("queen")).toEqual(["Qu", "E", "E", "N"]);
  });

  test("decomposes word with th digraph", () => {
    expect(_wordToTileKeys("the")).toEqual(["Th", "E"]);
  });

  test("decomposes word with multiple digraphs", () => {
    expect(_wordToTileKeys("there")).toEqual(["Th", "Er", "E"]);
  });

  test("decomposes word with in digraph", () => {
    expect(_wordToTileKeys("thing")).toEqual(["Th", "In", "G"]);
  });

  test("rejects standalone q (no u)", () => {
    // "qi" has q not followed by u
    expect(_wordToTileKeys("qi")).toBeNull();
  });
});

describe("buildTrie", () => {
  const { root, wordCount } = buildTrie(wordList);

  test("loads substantial word count from ENABLE", () => {
    // ENABLE has ~173K words; after filtering 4+ letters, should be 100K+
    expect(wordCount).toBeGreaterThan(100_000);
    expect(wordCount).toBeLessThan(175_000);
  });

  test("finds common words", () => {
    expect(lookupWord(root, "test")).toBe(true);
    expect(lookupWord(root, "hello")).toBe(true);
    expect(lookupWord(root, "word")).toBe(true);
    expect(lookupWord(root, "tree")).toBe(true);
  });

  test("finds words containing digraphs", () => {
    expect(lookupWord(root, "queen")).toBe(true);
    expect(lookupWord(root, "there")).toBe(true);
    expect(lookupWord(root, "thing")).toBe(true);
    expect(lookupWord(root, "other")).toBe(true);
  });

  test("rejects short words (< 4 letters)", () => {
    expect(lookupWord(root, "the")).toBe(false);
    expect(lookupWord(root, "an")).toBe(false);
    expect(lookupWord(root, "a")).toBe(false);
  });

  test("rejects non-words", () => {
    expect(lookupWord(root, "asdfghjkl")).toBe(false);
    expect(lookupWord(root, "zzzzz")).toBe(false);
  });

  test("lookupTileKeys works with tile-format keys", () => {
    expect(lookupTileKeys(root, ["T", "E", "S", "T"])).toBe(true);
    expect(lookupTileKeys(root, ["Qu", "E", "E", "N"])).toBe(true);
    expect(lookupTileKeys(root, ["Z", "Z", "Z", "Z"])).toBe(false);
  });
});
