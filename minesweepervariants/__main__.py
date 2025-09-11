#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/06/03 05:23
# @Author  : Wu_RH
# @FileName: run.py
# @Version : 1.0.0
import shutil
import sys
import argparse
import textwrap
from importlib.util import find_spec

from minesweepervariants import puzzle_query
from minesweepervariants import puzzle
from minesweepervariants import test

from minesweepervariants.config.config import DEFAULT_CONFIG

# ==== 获取默认值 ====
defaults = {}
defaults.update(DEFAULT_CONFIG)

# ==== 参数解析 ====
parser = argparse.ArgumentParser(description="")

subparsers = parser.add_subparsers(dest='command', required=False)

parser_list = subparsers.add_parser('list', help='列出所有规则的文档说明')

parser.add_argument("-s", "--size", nargs="+",
                    help="纸笔的题板边长")
parser.add_argument("-t", "--total", type=int, default=defaults.get("total"),
                    help="总雷数")
parser.add_argument("-c", "--rules", nargs="+", default=[],
                    help="所有规则名")
parser.add_argument("-d", "--dye", default=defaults.get("dye"),
                    help="染色规则名称，如 @c")
parser.add_argument("-m", "--mask",  default=defaults.get("dye"),
                    help="染色规则名称，如 @c")
parser.add_argument("-r", "--used-r", action="store_true", default=defaults.get("used_r"),
                    help="推理是否加R")
parser.add_argument("-a", "--attempts", type=int, default=defaults.get("attempts"),
                    help="尝试生成题板次数")
parser.add_argument("-q", "--query", default="",
                    help="生成题板的最高线索数范围 使用x-y表示(包含), 例如 5-8 表示线索数在5到8之间, -8表示不超过8, 5表示不少于5")
parser.add_argument("-e", "--early-stop", action="store_true", default=False,
                    help="生成题板的时候达到指定线索数量推理的时候 直接退出 这会导致线索图不正确")
parser.add_argument("-v", "--vice-board", action="store_true", default=False,
                    help="启用后生成题板的时候可以删除副板的信息")
parser.add_argument("--test", action="store_true", default=False,
                    help="启用后将仅生成一份使用了规则的答案题板")
parser.add_argument("--seed", type=int, default=defaults.get("seed"),
                    help="随机种子")
parser.add_argument("--onseed",  action="store_true", default=False,
                    help="启用可循的种子来生成题板,速度会大幅降低")
parser.add_argument("--log-lv", default=defaults.get("log_lv"),
                    help="日志等级，如 DEBUG、INFO、WARNING")
parser.add_argument("--board-class", default=defaults.get("board_class"),
                    help="题板的类名/题板的名称 通常使用默认值即可")
parser.add_argument("--no-image", action="store_true", default=defaults.get("no_image"),
                    help="是否不生成图片")
parser.add_argument("--file-name", default="",
                    help="文件名的前缀")
parser_list.add_argument("--shell", action="store_true", default=False)
args = parser.parse_args()

# ==== 调用生成 ====


def print_with_indent(text, indent="\t"):
    width = shutil.get_terminal_size(fallback=(80, 24)).columns // 2
    # 减去缩进长度，避免超宽
    effective_width = width - len(indent.expandtabs())
    lines = text.splitlines()
    for line in lines:
        wrapped = textwrap.fill(line, width=effective_width,
                                initial_indent=indent,
                                subsequent_indent=indent)
        print(wrapped, flush=True)
    print()


if args.command == "list":
    from minesweepervariants.impl import rule
    rule_list = rule.get_all_rules()
    # print(rule_list)

    if args.shell:
        import random
        encode = "utf-8"
        split_symbol = ''.join([chr(random.randint(33, 126)) for _ in range(50)])
        result = split_symbol.encode(encode)
        for rule_line in ["L", "M", "R"]:
            for name in rule_list[rule_line].keys():
                if not rule_list[rule_line][name]['module_doc']:
                    unascii_name = [n for n in rule_list[rule_line][name]["names"] if not n.isascii()]
                    zh_name = unascii_name[0] if unascii_name else ""
                    part = f"[{name}]{zh_name}: " + rule_list[rule_line][name]["doc"]
                else:
                    part = rule_list[rule_line][name]['module_doc']
                result += part.encode(encode)
                result += split_symbol.encode(encode)  # 如果原 join 是用分隔符连接
            result += (split_symbol * 2).encode(encode)
        print("hex_start:" + result.hex() + ":hex_end", end="", flush=True)
        # print(result.decode(encode))
        sys.stdout.buffer.flush()
        sys.exit(0)

    for rule_line, rule_line_name in [
        ("L", "\n\n左线规则:"),
        ("M", "\n\n中线规则:"),
        ("R", "\n\n右线规则:"),
    ]:
        if rule_list[rule_line]:
            print(rule_line_name, flush=True)
        for name in rule_list[rule_line]:
            doc = rule_list[rule_line][name]["module_doc"]
            unascii_name = [n for n in rule_list[rule_line][name]["names"] if not n.isascii()]
            zh_name = unascii_name[0] if unascii_name else ""
            if not doc:
                doc = f"[{name}]{zh_name}: " + rule_list[rule_line][name]["doc"]
            print_with_indent(doc)

    sys.exit(0)

if args.size is None:
    parser.print_help()
    sys.exit(0)
else:
    if len(args.size) == 0:
        parser.print_help()
        sys.exit(0)
    elif len(args.size) == 1:
        size = (int(args.size[0]), int(args.size[0]))
    else:
        size = (int(args.size[0]), int(args.size[1]))

if args.seed != defaults.get("seed"):
    args.attempts = 1

for rule_name in args.rules:
    if "$" in rule_name:
        args.rules[args.rules.index(rule_name)] = rule_name.replace("$", "^")

if args.test:
    test(
        log_lv=args.log_lv,
        seed=args.seed,
        size=size,
        total=args.total,
        rules=args.rules,
        dye=args.dye,
        mask_dye=args.mask,
        board_class=args.board_class,
        unseed=not args.onseed,
        image=not args.no_image,
    )
elif not args.query:
    if not args.no_image and find_spec("PIL") is None:
        print("可选依赖`image`未安装，请使用`pip install minesweepervariants[image]`安装, 或者添加--no-image参数不绘制图片.")
        exit(1)
    puzzle(
        log_lv=args.log_lv,
        seed=args.seed,
        attempts=args.attempts,
        size=size,
        total=args.total,
        rules=args.rules,
        dye=args.dye,
        mask_dye=args.mask,
        drop_r=(not args.used_r),
        board_class=args.board_class,
        vice_board=args.vice_board,
        unseed=not args.onseed,
        image=not args.no_image,
        file_name=args.file_name,
    )
else:
    puzzle_query(
        log_lv=args.log_lv,
        seed=args.seed,
        size=size,
        total=args.total,
        rules=args.rules,
        query=args.query,
        attempts=args.attempts,
        dye=args.dye,
        mask_dye=args.mask,
        drop_r=(not args.used_r),
        early_stop=args.early_stop,
        board_class=args.board_class,
        vice_board=args.vice_board,
        unseed=not args.onseed,
        file_name=args.file_name,
        image=not args.no_image,
    )
