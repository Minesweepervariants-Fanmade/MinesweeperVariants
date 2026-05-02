# MinesweeperVariants

 [简体中文](./README_zh-CN.md)

> To browse rules, visit the [Minesweeper Variants Rule List](https://minesweepervariants-fanmade.github.io/rule/)
> For the front-end project, use [Minesweeper Variants-Vue](https://koolshow.github.io/MinesweeperVariants-Vue/)

## Version 1.3

Minesweeper Variants, also known as 14mv, is a puzzle generator and game server for Minesweeper variants. It supports multiple rule combinations and can generate paper-style puzzles with a unique solution.

---

## Installation

### Requirements

Python 3.13 is recommended. Download the installer for your platform from [https://www.python.org](https://www.python.org).

> If you use a virtual environment such as venv/virtualenv or Poetry, install the package there instead. An isolated environment is recommended to avoid conflicts with system packages.

### Install with pip

```bash
python -m pip install --upgrade pip
python -m pip install minesweepervariants
```

---

## Running

There are two common entry points: the server and the generator CLI.

Start the server:

```bash
python -m minesweepervariants.server
# Keep it running, then open https://koolshow.github.io/MinesweeperVariants-Vue/ in your browser.
```

Run the generator CLI:

```bash
python -m minesweepervariants
```

You can add command-line options such as `-s` for size and `-t` for total mines. See the examples and options below.

---

## Development Setup

Poetry is recommended for dependency management and environment setup.

### 1. Install Python 3.13

On Windows, download the official installer from [https://www.python.org](https://www.python.org).

### 2. Install Poetry

Follow the official installation guide: [https://python-poetry.org/docs/#installation](https://python-poetry.org/docs/#installation)

### 3. Install dependencies and create a virtual environment

From the project root:

```bash
poetry install
```

Poetry will create an isolated virtual environment and install all dependencies automatically.

### 4. Run the project

You do not need to activate the virtual environment manually. Use Poetry to run scripts directly:

```bash
poetry run python run.py [arguments]
```

You can use the same pattern for other commands: `poetry run <command>`.

### 5. C extension build tools on Windows

Some dependencies may require compiled C extensions. Install:

* Visual C++ Build Tools

Download it from:

> [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

Make sure these components are selected:

* C++ build tools, including MSVC and the Windows SDK
* CMake

Restart the terminal after installation.

---

## Usage

### Run command

```bash
run [arguments]
```

This invokes the main generator.

---

### Common run options

| Option                          | Type              | Description                                                                                            |
| ------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------ |
| `-s, --size`                  | integer, required | Puzzle size                                                                                            |
| `-t, --total`                 | integer           | Total mines                                                                                            |
| `-c, --rules`                 | string list       | All rule names, such as `2F 1Q V 1K 1F`; they are automatically grouped into left/middle/right rules |
| `-E, --early-rules`           | string list       | Extra left-rule names used only during the initial generation phase; multiple allowed                  |
| `-d, --dye`                   | string            | Dye rule name, such as `@c`                                                                          |
| `-m, --mask`                  | string            | Mask dye rule name, such as `@c`                                                                     |
| `-r, --used-r`                | flag              | Enable R deduction (disabled by default)                                                               |
| `-a, --attempts`              | integer           | Maximum number of generation attempts                                                                  |
| `-q, --query`                 | string range      | Clue-count filter, such as `5-8`, `-8`, or `5`                                                   |
| `-e, --early-stop`            | flag              | Stop early when the query target is reached (may produce an incorrect clue board)                      |
| `-v, --vice-board`            | flag              | Allow removing vice-board information during generation                                                |
| `-T, --test`                  | flag              | Generate only one answer board using the rules                                                         |
| `-S, --seed`                  | integer           | Random seed (setting it explicitly forces attempts to 1)                                               |
| `-O, --onseed`                | flag              | Use a reproducible seed for generation; this is slower                                                 |
| `-L, --log-lv`                | string            | Log level, such as `DEBUG`, `INFO`, or `WARNING`                                                 |
| `-B, --board-class`           | string            | Board class / board name; the default is usually fine                                                  |
| `-I, --no-image`              | flag              | Do not generate images                                                                                 |
| `-F, --file-name`             | string            | Output file name prefix                                                                                |
| `-D, --dynamic-dig-rounds`    | integer           | Dynamic clue-removal rounds (auto-detected when omitted)                                               |
| `-M, --dynamic-dig-max-batch` | integer           | Maximum number of cell changes per dynamic clue-removal round                                          |
| `--output-path`               | string            | Output directory for generated images                                                                  |
| `--log-path`                  | string            | Output directory for logs                                                                              |
| `--lang`                      | string            | Output language code, for example en_US or zh_CN                                                       |
| `list`                        |                   | Show all implemented rule documentation                                                                |

---

### Example runs

```bash
run -s 5 -c 2F 1k 1q V -d c -r -q 2-4 -I
# Example with extra left-rule names during initial generation only
run -s 5 -c 1Q V -E 2F 3L -I
```

> Generate a 5x5 board with checkerboard dyeing, using 2F and 1Q as left rules, and V and 1K as right rules.
> Enable R deduction, keep only boards whose clue count is between 2 and 4, and disable image output.
> Note: rule names are case-insensitive.

Additional options for `list`:

| Option     | Type | Description                       |
| ---------- | ---- | --------------------------------- |
| `--json` | flag | Output rule documentation as JSON |

---

### Output files

Successful runs create the following files under `output/`:

```
output/
├─ output.png   (default image output from img)
├─ demo.txt     (historical deduction text)
├─ demo.png     (puzzle image)
└─ answer.png   (solution image)
```

`demo.txt` contains:

* Generation time
* Clue table, when `-q` is used
* Time spent generating
* Total mines, formatted as total mines / total cells
* Seed / puzzle ID as an integer string
* Puzzle content
* Solution and non-question-mark content
* The generated command for the puzzle image, prefixed with `img`
* The generated command for the answer image

---

### Image output command

```bash
img [arguments]
```

This invokes the image output subcommand.

---

### img options

| Option                | Type    | Description                                         |
| --------------------- | ------- | --------------------------------------------------- |
| `-c, --code`        | string  | Board bytecode representing fixed board content     |
| `-r, --rule-text`   | string  | Rule string; quote it if it contains spaces         |
| `-s, --size`        | integer | Cell size                                           |
| `-o, --output`      | string  | Output file name without extension                  |
| `-w, --white-base`  | flag    | Use a white background                              |
| `-b, --board-class` | string  | Underlying board class; the default is usually fine |

---

### Example image command

```bash
img -c ... -r "[V]-R*/15-4395498" -o demo -s 100 -w
```

> Generate an image using `[V]-R*/15-4395498` as the bottom text and save it to `output/demo.png`.
> Each cell is 100x100 pixels on a white background.

> Note: replace `...` with the board code value, which is saved in `output/demo.txt`.

---

## Developer documentation

The full developer documentation lives in the [`doc/`](./doc) directory:

| Document                                      | Description                           |
| --------------------------------------------- | ------------------------------------- |
| [README.md](./doc/README.md)                     | Entry point                           |
| [dev/rule_mines.md](./doc/dev/rule_mines.md)     | Left-rule interface documentation     |
| [dev/rule_clue_mines.md](./doc/dev/rule_clue.md) | Middle-rule interface documentation   |
| [dev/rule_clue.md](./doc/dev/rule_clue.md)       | Right-rule interface documentation    |
| [dev/board_api.md](./doc/dev/board_api.md)       | Board structure and coordinate system |
| [dev/utils.md](./doc/dev/utils.md)               | Utility module interface              |
