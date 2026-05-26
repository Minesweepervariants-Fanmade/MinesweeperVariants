#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/06/19 23:05
# @Author  : Wu_RH
# @FileName: img.py
# @Version : 1.0.0
import sys
import argparse

from minesweepervariants.abs.board import json_loads, decompress
from minesweepervariants.impl.impl_obj import get_board, decode_board
from minesweepervariants.utils.image_create import draw_board


from minesweepervariants.config.config import DEFAULT_CONFIG

# ==== 获取默认值 ====
defaults = {}
defaults.update(DEFAULT_CONFIG)

# ==== 调用生成 ====


def main(
    code: str,
    rule_text: str,
    output: str,
    white_base: bool,
    size: int,
    because: list[str],
    deduced: list[str],
):
    board = decode_board(data=json_loads(decompress(code)))

    hint_because = []
    hint_deduced = []
    for pos, _ in board(mode="none"):
        if str(pos) in because:
            hint_because.append(pos)
        if str(pos) in deduced:
            hint_deduced.append(pos)

    draw_board(
        board=board,
        bottom_text=rule_text,
        output=output,
        background_white=white_base,
        cell_size=size,
        hint_because=hint_because,
        hint_deduced=hint_deduced,
    )
