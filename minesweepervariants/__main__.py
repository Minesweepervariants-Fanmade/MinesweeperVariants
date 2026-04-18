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
import json
from pathlib import Path
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
parser.add_argument("-E", "--early-rules", nargs="+", default=[],
                    help="仅在初始题板生成阶段使用的左线规则名，可多个")
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
parser.add_argument("-T", "--test", action="store_true", default=False,
                    help="启用后将仅生成一份使用了规则的答案题板")
parser.add_argument("-S", "--seed", type=int, default=defaults.get("seed"),
                    help="随机种子")
parser.add_argument("-O", "--onseed",  action="store_true", default=False,
                    help="启用可循的种子来生成题板,速度会大幅降低")
parser.add_argument("-L", "--log-lv", default=defaults.get("log_lv"),
                    help="日志等级，如 DEBUG、INFO、WARNING")
parser.add_argument("-B", "--board-class", default=defaults.get("board_class"),
                    help="题板的类名/题板的名称 通常使用默认值即可")
parser.add_argument("-I", "--no-image", action="store_true", default=defaults.get("no_image"),
                    help="是否不生成图片")
parser.add_argument("-F", "--file-name", default="",
                    help="文件名的前缀")
parser.add_argument("-D", "--dynamic-dig-rounds", type=int, default=None,
                    help="动态删线索模式迭代轮数。未指定时: 动态规则自动100轮, 其他规则为0; 显式指定则强制使用")
parser.add_argument("-M", "--dynamic-dig-max-batch", type=int, default=defaults.get("dynamic_dig_max_batch"),
                    help="动态删线索每轮最大改动格数")
parser_list.add_argument("-H", "--shell", action="store_true", default=False)
parser_list.add_argument("--json", action="store_true", default=False)
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


def build_rule_doc(name, rule_info, include_author_tag=True, image_name=""):
    unascii_name = [n for n in rule_info["names"] if not n.isascii()]
    zh_name = unascii_name[0] if unascii_name else ""
    names = ", ".join(i for i in rule_info["names"] if i not in [name, zh_name])
    author = rule_info.get("author", "")
    if isinstance(author, tuple):
        author = f"{author[0]}({author[1]})" if author else ""
    author_part = f"[@Author={author}]" if include_author_tag and author else ""
    image_part = f"[@Image={image_name}]" if image_name else ""
    return (
        f"[{name}]{zh_name}{('(' + names + ')') if names else ''}"
        f"{author_part}{image_part}: "
    ) + rule_info["doc"]


def collect_rule_images(rule_list):
    image_dir = Path(__file__).resolve().parent / "impl" / "rule" / "image"
    image_names = []
    if image_dir.exists() and image_dir.is_dir():
        image_names = sorted(p.name for p in image_dir.iterdir() if p.is_file())

    image_map = {}
    for rule_line in ["L", "M", "R"]:
        for name in rule_list[rule_line].keys():
            prefix = f"{name}_"
            image_name = ""
            for candidate in image_names:
                if candidate.startswith(prefix):
                    image_name = candidate
                    break
            image_map[name] = image_name
    return image_map


def traverse_rule_list(rule_list, on_rule, on_line_start=None, on_line_end=None):
    for rule_line in ["L", "M", "R"]:
        if on_line_start:
            on_line_start(rule_line)
        for name, rule_info in rule_list[rule_line].items():
            on_rule(rule_line, name, rule_info)
        if on_line_end:
            on_line_end(rule_line)


def handle_list_shell_output(rule_list, image_map):
    import random

    encode = "utf-8"
    split_symbol = ''.join([chr(random.randint(33, 126)) for _ in range(50)])
    split_name_symbol = ''.join([chr(random.randint(33, 126)) for _ in range(10)])
    split_symbol_bytes = split_symbol.encode(encode)
    result = bytearray(split_symbol_bytes + split_name_symbol.encode(encode))

    def on_rule(_, name, rule_info):
        author = rule_info.get("author", ("", ""))
        image_name = image_map.get(name, "")
        part = build_rule_doc(name, rule_info, include_author_tag=True, image_name=image_name)
        part = split_name_symbol.join(
            [name] + rule_info["names"] +
            (list(author) if author else ["", ""]) + [image_name, part]
        )
        result.extend(part.encode(encode))
        result.extend(split_symbol_bytes)

    def on_line_end(_):
        result.extend(split_symbol_bytes)

    traverse_rule_list(rule_list, on_rule=on_rule, on_line_end=on_line_end)

    print("hex_start:" + result.hex() + ":hex_end", end="", flush=True)


def handle_list_text_output(rule_list, image_map):
    rule_line_name_map = {
        "L": "\n\n左线规则:",
        "M": "\n\n中线规则:",
        "R": "\n\n右线规则:",
    }

    def on_line_start(rule_line):
        if rule_list[rule_line]:
            print(rule_line_name_map[rule_line], flush=True)

    def on_rule(_, name, rule_info):
        doc = build_rule_doc(
            name,
            rule_info,
            include_author_tag=True,
            image_name=image_map.get(name, ""),
        )
        print_with_indent(doc)

    traverse_rule_list(rule_list, on_rule=on_rule, on_line_start=on_line_start)


def handle_list_json_output(rule_list, image_map):
    result = {
        "L": [],
        "M": [],
        "R": [],
    }

    def on_rule(rule_line, name, rule_info):
        image_name = image_map.get(name, "")
        names = list(rule_info.get("names", []))
        raw_author = rule_info.get("author", "")
        author_name = ""
        author_id = ""
        if isinstance(raw_author, tuple):
            if len(raw_author) > 0:
                author_name = str(raw_author[0])
            if len(raw_author) > 1:
                author_id = str(raw_author[1])
        elif isinstance(raw_author, str):
            author_name = raw_author

        result[rule_line].append({
            "rule_line": rule_line,
            "name": name,
            "names": names,
            "author": {
                "name": author_name,
                "id": author_id,
            },
            "image": image_name,
            "doc": rule_info.get("doc", ""),
            "display": build_rule_doc(
                name,
                rule_info,
                include_author_tag=True,
                image_name=image_name,
            ),
        })

    traverse_rule_list(rule_list, on_rule=on_rule)
    print(json.dumps(result, ensure_ascii=False), end="", flush=True)

def main():
    if args.command == "list":
        from minesweepervariants.impl import rule
        rule_list = rule.get_all_rules()
        image_map = collect_rule_images(rule_list)
        # print(rule_list)

        if args.shell:
            handle_list_shell_output(rule_list, image_map)
        elif args.json:
            handle_list_json_output(rule_list, image_map)
        else:
            handle_list_text_output(rule_list, image_map)

        return

    if args.size is None:
        parser.print_help()
        return
    else:
        if len(args.size) == 0:
            parser.print_help()
            return
        elif len(args.size) == 1:
            size = (int(args.size[0]), int(args.size[0]))
        else:
            size = (int(args.size[0]), int(args.size[1]))

    if args.seed != defaults.get("seed"):
        args.attempts = 1

    for rule_index in range(len(args.rules)):
        rule_name = args.rules[rule_index]
        if "$0" in rule_name:
            rule_name = rule_name.replace("$0", "$")
        if "$1" in rule_name:
            rule_name = rule_name.replace("$1", "^")
        if "$2" in rule_name:
            rule_name = rule_name.replace("$2", "|")
        if "$3" in rule_name:
            rule_name = rule_name.replace("$3", "&")
        if "$4" in rule_name:
            rule_name = rule_name.replace("$4", ">")
        if "$5" in rule_name:
            rule_name = rule_name.replace("$5", "<")
        if "$6" in rule_name:
            rule_name = rule_name.replace("$6", "%")
        args.rules[rule_index] = rule_name

    for rule_index in range(len(args.early_rules)):
        rule_name = args.early_rules[rule_index]
        if "$0" in rule_name:
            rule_name = rule_name.replace("$0", "$")
        if "$1" in rule_name:
            rule_name = rule_name.replace("$1", "^")
        if "$2" in rule_name:
            rule_name = rule_name.replace("$2", "|")
        if "$3" in rule_name:
            rule_name = rule_name.replace("$3", "&")
        if "$4" in rule_name:
            rule_name = rule_name.replace("$4", ">")
        if "$5" in rule_name:
            rule_name = rule_name.replace("$5", "<")
        if "$6" in rule_name:
            rule_name = rule_name.replace("$6", "%")
        args.early_rules[rule_index] = rule_name

    if args.test:
        test(
            log_lv=args.log_lv,
            seed=args.seed,
            size=size,
            total=args.total,
            rules=args.rules,
            early_rules=args.early_rules,
            dye=args.dye,
            mask_dye=args.mask,
            board_class=args.board_class,
            unseed=not args.onseed,
            image=not args.no_image,
        )
    elif not args.query:
        if not args.no_image and find_spec("PIL") is None:
            print("可选依赖`image`未安装，请使用`pip install minesweepervariants[image]`安装, 或者添加--no-image参数不绘制图片.")
            return
        puzzle(
            log_lv=args.log_lv,
            seed=args.seed,
            attempts=args.attempts,
            size=size,
            total=args.total,
            rules=args.rules,
            early_rules=args.early_rules,
            dye=args.dye,
            mask_dye=args.mask,
            drop_r=(not args.used_r),
            board_class=args.board_class,
            vice_board=args.vice_board,
            unseed=not args.onseed,
            image=not args.no_image,
            file_name=args.file_name,
            dynamic_dig_rounds=args.dynamic_dig_rounds,
            dynamic_dig_max_batch=args.dynamic_dig_max_batch,
        )
    else:
        puzzle_query(
            log_lv=args.log_lv,
            seed=args.seed,
            size=size,
            total=args.total,
            rules=args.rules,
            early_rules=args.early_rules,
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
            dynamic_dig_rounds=args.dynamic_dig_rounds,
            dynamic_dig_max_batch=args.dynamic_dig_max_batch,
        )

if __name__ == "__main__":
    main()