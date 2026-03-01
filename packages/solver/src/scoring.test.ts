import { describe, expect, test } from "bun:test";
import { scoreWord } from "./scoring.js";

describe("scoreWord", () => {
  test("words shorter than 4 letters score 0", () => {
    expect(scoreWord(1)).toBe(0);
    expect(scoreWord(2)).toBe(0);
    expect(scoreWord(3)).toBe(0);
  });

  test("standard scoring table", () => {
    expect(scoreWord(4)).toBe(1);
    expect(scoreWord(5)).toBe(2);
    expect(scoreWord(6)).toBe(3);
    expect(scoreWord(7)).toBe(5);
    expect(scoreWord(8)).toBe(11);
  });

  test("9+ letters score 2 * length", () => {
    expect(scoreWord(9)).toBe(18);
    expect(scoreWord(10)).toBe(20);
    expect(scoreWord(12)).toBe(24);
  });
});
