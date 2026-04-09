#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/06/19 23:05
# @Author  : Wu_RH
# @FileName: img.py
# @Version : 1.0.0
import sys
import argparse

from minesweepervariants.impl.impl_obj import get_board, decode_board
from minesweepervariants.utils.image_create import draw_board


from minesweepervariants.config.config import DEFAULT_CONFIG

# ==== 获取默认值 ====
defaults = {}
defaults.update(DEFAULT_CONFIG)

# ==== 参数解析 ====
parser = argparse.ArgumentParser(description="")

parser.add_argument("-c", "--code", type=str,
                    help="题板字节码")  # 字符串类型
parser.add_argument("-r", "--rule-text", type=str, default="",
                    help="规则字符串，有空格需要带引号")  # 字符串类型
parser.add_argument("-o", "--output", type=str, default=defaults["output_file"],
                    help="输出的文件名（不含后缀）")  # 字符串类型
parser.add_argument("-s", "--size", type=int, default=defaults["cell_size"],
                    help="单元格像素数")  # 整数
parser.add_argument("-b", "--board-class", type=str, default=defaults["board_class"],
                    help="题板类的名称代号")
parser.add_argument("-w", "--white-base", action="store_true", default=defaults["white_base"],
                    help="题板背景是否是白底的")

args = parser.parse_args()

# ==== 调用生成 ====

if args.code is None:
    parser.print_help()
    sys.exit(0)

try:
    board = decode_board(args.code)
except:
    code = bytes.fromhex(args.code)
    board = get_board(args.board_class)(rules={}, code=code)


draw_board(
    board=board,
    bottom_text=args.rule_text,
    output=args.output,
    background_white=args.white_base,
    cell_size=args.size
)
