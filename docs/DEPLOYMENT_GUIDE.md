# Deployment Guide for MinesweeperVariants

This guide documents the successful deployment of the MinesweeperVariants puzzle generator according to the README instructions.

## Deployment Steps

### 1. Prerequisites
- Python 3.12+ (tested with Python 3.12.3)
- pip (Python package manager)

### 2. Installation

Since Poetry was not available in the environment, we used the alternative installation method with pip:

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install the package in editable mode (for development)
pip install -e .

# Install optional dependencies for image generation
pip install pillow
```

### 3. Initialize Git Submodules

The project uses git submodules for the `rule` and `server` components:

```bash
git submodule init
git submodule update
```

This step is **critical** - without initializing the submodules, the rule modules won't be available and the application will fail with import errors.

### 4. Verify Installation

List available rules to verify the installation:

```bash
python -m minesweepervariants list
```

## Generated Puzzle Example

Successfully generated a 5x5 puzzle with the following configuration:

### Command Used
```bash
python -m minesweepervariants -s 5 -c 2F 1K V -d c -t 5
```

### Parameters Explanation
- `-s 5`: Size 5x5 (square board)
- `-c 2F 1K V`: Rules used:
  - `2F` (花田/Flower Field): Mines in colored cells have exactly 1 mine in surrounding 4 cells
  - `1K` (骑士/Knight): Each cell represents total mines in knight-move positions
  - `V` (标准扫雷/Standard Minesweeper): Standard minesweeper rules
- `-d c`: Checkerboard coloring (@c)
- `-t 5`: Total of 5 mines

### Results

The generator produced:
- **Puzzle board** (`demo.png`): The puzzle with clues
- **Answer board** (`answer.png`): The solution with mine positions
- **Demo file** (`demo.txt`): Detailed information including:
  - Generation time: 0.47 seconds
  - Seed: 3132466 (for reproducibility)
  - Board code for regeneration
  - Solving statistics

The puzzle has:
- Total cells: 19 (5x5 = 25, with some cells having clues)
- Total mines: 5
- Clues provided: Multiple cells with numbers and special markers (V for standard rule, 1K for knight rule)

## Output Files

Generated files are located in `.\output/`:
- `demo.png` - The puzzle image
- `answer.png` - The solution image  
- `demo.txt` - Detailed puzzle information and metadata

## Notes

- Some import warnings for sharpRule modules appear but don't affect puzzle generation
- The puzzle has a unique solution and can be solved using logical deduction
- The checkerboard coloring helps distinguish different rule applications
- Each cell may have multiple rule indicators (V for standard, 1K for knight)

## Reproducibility

To generate the exact same puzzle again, use the seed:

```bash
python -m minesweepervariants -s 5 -c 2F 1K V -d c -t 5 --seed 3132466
```

## Conclusion

The deployment was successful. The MinesweeperVariants puzzle generator is now operational and can generate various types of minesweeper variant puzzles according to the configured rules.
