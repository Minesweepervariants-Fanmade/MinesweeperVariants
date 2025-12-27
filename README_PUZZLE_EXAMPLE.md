# Generated Puzzle Example

This document showcases the puzzle generated during deployment testing.

## Puzzle Board (demo.png)

The puzzle presents a 5x5 grid with various clues:
- Numbers with rule indicators (V, 1K)
- Question marks indicating unknown cells
- Checkerboard coloring pattern

![Puzzle Board](output/demo.png)

## Solution Board (answer.png)

The solution shows the complete board with all mine locations marked (雷):

![Solution Board](output/answer.png)

## Puzzle Specifications

- **Size**: 5x5
- **Total Mines**: 5 out of 19 cells
- **Rules Applied**:
  - `[2F]` - 花田 (Flower Field): Colored cells with mines must have exactly 1 mine in surrounding 4 cells
  - `[1K]` - 骑士 (Knight): Numbers indicate total mines in knight-move positions
  - `[V]` - 标准扫雷 (Standard): Numbers indicate mines in surrounding 8 cells
- **Coloring**: `[@c]` - Checkerboard pattern
- **Generation Time**: 0.47 seconds
- **Seed**: 3132466

## Rule Explanations

### Standard Minesweeper (V)
Cells marked with "V" follow standard minesweeper rules - the number indicates how many mines are in the surrounding 8 cells.

### Knight (1K)
Cells marked with "1K" count mines in knight-move positions (like a chess knight) - 8 cells that are 2 squares away in one direction and 1 square perpendicular.

### Flower Field (2F)
This rule applies to the colored (gray) cells. If a colored cell contains a mine, exactly 1 of its 4 adjacent cells (up, down, left, right) must also contain a mine.

## How to Play

1. Start with the cells that have clues
2. Consider all three rules when deducing mine locations
3. Use logical reasoning to determine where mines must be
4. The puzzle has a unique solution!

## Interesting Observations

From the solution:
- The 5 mines are located at: B3, C3, E4, D5, E5
- The mines form an interesting cluster pattern
- The "?" at C1 remains unknown in the puzzle, requiring logical deduction
- Multiple overlapping rules create challenging but solvable constraints

This demonstrates the power of combining multiple minesweeper variant rules to create engaging logical puzzles!
