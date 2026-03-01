/**
 * Trie construction for Boggle word lookup.
 *
 * The trie maps tile sequences (not raw characters) to valid words.
 * Digraph tiles (Qu, Th, Er, In, An, He) are treated as single keys,
 * so "queen" is stored as Qu → e → e → n, matching how tiles appear on the board.
 */

/** Map from lowercase digraph to its canonical tile form. */
const DIGRAPH_TO_TILE = new Map<string, string>([
  ["qu", "Qu"],
  ["er", "Er"],
  ["th", "Th"],
  ["in", "In"],
  ["an", "An"],
  ["he", "He"],
]);

export interface TrieNode {
  /** Child nodes keyed by tile value (single uppercase letter or digraph). */
  readonly children: Map<string, TrieNode>;
  /** True if this node marks the end of a valid word. */
  isEnd: boolean;
}

function createNode(): TrieNode {
  return { children: new Map(), isEnd: false };
}

/**
 * Decompose a lowercase word into a sequence of tile keys.
 *
 * Greedily matches digraphs first (qu, th, er, in, an, he),
 * then falls back to single characters. Characters not representable
 * by any tile (e.g. "q" not followed by "u") cause the word to be
 * rejected (returns null).
 */
function wordToTileKeys(word: string): readonly string[] | null {
  const keys: string[] = [];
  let i = 0;

  while (i < word.length) {
    let matched = false;

    // Try digraphs first (all are 2 chars)
    if (i + 1 < word.length) {
      const pair = word.substring(i, i + 2);
      const tile = DIGRAPH_TO_TILE.get(pair);
      if (tile) {
        keys.push(tile);
        i += 2;
        matched = true;
      }
    }

    if (!matched) {
      const ch = word[i].toUpperCase();
      // "Q" without "u" can't be represented — reject the word
      if (word[i] === "q") return null;
      keys.push(ch);
      i += 1;
    }
  }

  return keys;
}

/**
 * Insert a word into the trie, decomposing it into tile keys.
 * Returns true if the word was inserted, false if it was rejected
 * (contains characters that can't be represented by tiles).
 */
function insertWord(root: TrieNode, word: string): boolean {
  const keys = wordToTileKeys(word);
  if (!keys) return false;

  let node = root;
  for (const key of keys) {
    let child = node.children.get(key);
    if (!child) {
      child = createNode();
      node.children.set(key, child);
    }
    node = child;
  }
  node.isEnd = true;
  return true;
}

/**
 * Build a trie from a newline-separated word list.
 *
 * Words are lowercased and decomposed into tile sequences.
 * Words shorter than `minLength` are excluded (Boggle requires 4+ letters).
 * Words containing characters not representable by tiles are silently skipped.
 *
 * @returns The root TrieNode and the count of words inserted.
 */
export function buildTrie(
  wordList: string,
  minLength = 4,
): { readonly root: TrieNode; readonly wordCount: number } {
  const root = createNode();
  let wordCount = 0;

  const lines = wordList.split("\n");
  for (const raw of lines) {
    const word = raw.trim().toLowerCase();
    if (word.length < minLength) continue;
    if (insertWord(root, word)) {
      wordCount++;
    }
  }

  return { root, wordCount };
}

/**
 * Check if a word exists in the trie.
 * The word should be provided as tile keys (e.g. ["Qu", "E", "E", "N"]).
 */
export function lookupTileKeys(root: TrieNode, keys: readonly string[]): boolean {
  let node: TrieNode | undefined = root;
  for (const key of keys) {
    node = node.children.get(key);
    if (!node) return false;
  }
  return node.isEnd;
}

/**
 * Check if a raw lowercase word exists in the trie.
 */
export function lookupWord(root: TrieNode, word: string): boolean {
  const keys = wordToTileKeys(word.toLowerCase());
  if (!keys) return false;
  return lookupTileKeys(root, keys);
}

// Re-export for testing
export { wordToTileKeys as _wordToTileKeys };
