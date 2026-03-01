# @boggle-vision/solver

Trie-based DFS Boggle board solver. Given a grid of tile labels, finds all valid English words and scores them.

## Usage

```typescript
import { buildTrie, solveBoard } from "@boggle-vision/solver";

// Build the trie once from ENABLE word list
const wordList = await Bun.file("data/enable1.txt").text();
const { root, wordCount } = buildTrie(wordList);

// Solve a 6x6 board
const board = [
  ["Y", "G", "R", "L", "H", "N"],
  ["E", "T", "T", "N", "T", "O"],
  ["Th", "F", "E", "E", "E", "N"],
  ["C", "L", "E", "T", "H", "O"],
  ["J", "R", "T", "R", "E", "L"],
  ["D", "I", "T", "H", "E", "J"],
];

const result = solveBoard(board, root);
console.log(`Found ${result.words.length} words for ${result.totalPoints} points`);
```

## Dictionary

Uses the **ENABLE** (Enhanced North American Benchmark Lexicon) word list — the standard dictionary for word game software. ~173K words, no proper nouns.

## Scoring

Standard Super Big Boggle rules:

| Word Length | Points |
|-------------|--------|
| 4 letters | 1 |
| 5 letters | 2 |
| 6 letters | 3 |
| 7 letters | 5 |
| 8 letters | 11 |
| 9+ letters | 2 × length |

## Features

- Handles digraph tiles (Qu, Th, Er, In, An, He) as atomic units
- Skips BLOCK tiles during traversal
- 8-directional adjacency (standard Boggle rules)
- Trie-based pruning for fast solving
- Returns word paths through the board for visualization
