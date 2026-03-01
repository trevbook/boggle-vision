/**
 * Boggle scoring rules.
 *
 * Standard Super Big Boggle scoring:
 * - < 4 letters: 0 points (not valid)
 * - 4 letters: 1 point
 * - 5 letters: 2 points
 * - 6 letters: 3 points
 * - 7 letters: 5 points
 * - 8 letters: 11 points
 * - 9+ letters: 2 × length points
 */

const SCORE_TABLE: Record<number, number> = {
  4: 1,
  5: 2,
  6: 3,
  7: 5,
  8: 11,
};

/**
 * Score a single Boggle word by its letter count.
 *
 * Note: `length` here is the number of *characters* in the word (not tiles).
 * Digraphs like "Qu" count as 2 characters for scoring purposes,
 * matching standard Boggle rules.
 */
export function scoreWord(wordLength: number): number {
  if (wordLength < 4) return 0;
  if (wordLength <= 8) return SCORE_TABLE[wordLength];
  return 2 * wordLength;
}
