#!/usr/bin/env python3
"""从output/demo.txt末条题板生成 ControlNet Inpaint 输入图(1024x1024).

输出:
  clue_mask.png  — Inpaint 蒙版 (白=生成主体, 黑=保留背景)
  clue_base.png  — 纯白底图 (作为 init_image)
用法 (diffusers):
  pipe(prompt=..., image=clue_base, mask_image=clue_mask, control_image=make_inpaint_condition(clue_base, clue_mask))
"""
import json as std_json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from minesweepervariants.impl.impl_obj import decode_board
from minesweepervariants.json_object import json_loads

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "output"
DEMO_FILE = OUTPUT_PATH / "demo.txt"
MASK_OUTPUT = OUTPUT_PATH / "clue_mask.png"
BASE_OUTPUT = OUTPUT_PATH / "clue_base.png"


def find_last_board_json(text: str) -> str:
    """Extract JSON string from last 题板代码: entry."""
    marker = "题板代码: \n"
    idx = text.rfind(marker)
    if idx == -1:
        print("Error: no 题板代码 found in demo.txt", file=sys.stderr)
        sys.exit(1)

    rest = text[idx + len(marker):].lstrip()
    decoder = std_json.JSONDecoder()
    try:
        _, end = decoder.raw_decode(rest)
    except std_json.JSONDecodeError as e:
        print(f"Error: JSON parse failed: {e}", file=sys.stderr)
        sys.exit(1)

    return rest[:end]


def main():
    if not DEMO_FILE.exists():
        print(f"Error: {DEMO_FILE} not found. Run generate_puzzle first.", file=sys.stderr)
        sys.exit(1)

    text = DEMO_FILE.read_text("utf-8")
    json_str = find_last_board_json(text)
    board = decode_board(json_loads(json_str))

    # Get first master board size
    keys = board.get_interactive_keys()
    if not keys:
        keys = board.get_board_keys()
    master_key = keys[0]

    size = board.get_config(master_key, "size")
    cols, rows = size.cols, size.rows

    TARGET = 1024
    img = Image.new("L", (TARGET, TARGET), 255)  # white bg
    draw = ImageDraw.Draw(img)

    # Scale cell to fit largest dim, center grid
    max_dim = max(cols, rows)
    cell_size = TARGET / max_dim
    grid_w = cols * cell_size
    grid_h = rows * cell_size
    offset_x = (TARGET - grid_w) / 2
    offset_y = (TARGET - grid_h) / 2

    # Paint clue cells black
    for pos, _ in board(key=master_key, mode="none"):
        if board.get_type(pos) == "C":  # clue
            x0 = offset_x + pos.col * cell_size
            y0 = offset_y + pos.row * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=0)

    # Morphological smoothing with elliptical kernel
    arr = np.array(img, dtype=np.uint8)
    k = int(cell_size * 0.12) | 1  # odd kernel size ~12% cell
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
    arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)
    img = Image.fromarray(arr)

    img.save(MASK_OUTPUT)
    # Generate pure white base image for ControlNet Inpaint reference
    base = Image.new("L", (TARGET, TARGET), 255)
    base.save(BASE_OUTPUT)

    print(f"[ControlNet Inpaint 蒙版]   → {MASK_OUTPUT}")
    print(f"  白=生成主体(非线索格), 黑=保留背景(线索格)")
    print(f"[ControlNet Inpaint 原图]   → {BASE_OUTPUT}")
    print(f"  纯白底图, 配合蒙版使用")
    print(f"棋盘 {cols}x{rows}, 格子 {cell_size:.2f}px, 网格 {grid_w:.0f}x{grid_h:.0f} 居中")


if __name__ == "__main__":
    main()
