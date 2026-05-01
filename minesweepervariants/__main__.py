#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/06/03 05:23
# @Author  : Wu_RH
# @FileName: run.py
# @Version : 1.0.0
import argparse
import json
import shutil
import textwrap
import sys
import locale
from pathlib import Path
from importlib.util import find_spec

from minesweepervariants import puzzle_query
from minesweepervariants import puzzle
from minesweepervariants import test

from minesweepervariants.config.config import DEFAULT_CONFIG
from minesweepervariants.utils.tool import get_logger
from minesweepervariants.utils import tool

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
parser.add_argument("-m", "--mask", default=defaults.get("dye"),
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
parser.add_argument("-O", "--onseed", action="store_true", default=False,
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

parser.add_argument("--output-path", default=None,
                    help="图片输出目录路径，图片将保存到此目录（默认使用配置中的路径）")
parser.add_argument("--log-path", default=None,
                    help="日志输出目录路径，日志将保存到此目录（默认使用配置中的路径）")
parser.add_argument("--lang", default=defaults.get("lang"),
                    help="输出语言代码，例如: en_US, zh_CN。")
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


def _build_list_display(rule_info, rule_key):
    """构建 list 命令输出的 display 字符串。"""
    import locale as locale_mod

    # 从 name_map 中选择本地化名称；优先当前 locale，再短码、再 default，最后回退到任意已有名称或 rule_key
    name_map = rule_info.get("name", {})
    if isinstance(name_map, str):
        display_name = name_map
    else:
        lang = locale_mod.getlocale()[0]
        display_name = name_map.get(lang) or name_map.get("default", rule_info.get("id", "Unknown"))


    # 构建作者文本
    author = rule_info.get("author", {})
    author_text = ""
    if isinstance(author, dict):
        a_name = author.get("name", "")
        a_id = author.get("id", "")
        if a_name and a_id:
            author_text = f"{a_name}({a_id})"
        else:
            author_text = a_name or a_id

    # 构建 doc 文本
    doc_map = rule_info.get("doc", {})
    doc_text = ""
    if isinstance(doc_map, dict):
        doc_text = doc_map.get("default", "") or (next(iter(doc_map.values()), "") if doc_map else "")

    # 构建最终 display 字符串
    rule_id = rule_info.get("id", rule_key)
    image = rule_info.get("image", "")

    author_part = f"[@Author={author_text}]" if author_text else ""
    image_part = f"[@Image={image}]" if image else ""

    return f"[{rule_id}]{display_name}{author_part}{image_part}: {doc_text}"


def handle_list_text_output(rule_list):
    rule_line_name_map = {
        "L": "\n\n左线规则:",
        "M": "\n\n中线规则:",
        "R": "\n\n右线规则:",
    }

    for rule_line in ["L", "M", "R"]:
        if rule_list.get(rule_line):
            print(rule_line_name_map[rule_line], flush=True)
        for rule_info in rule_list.get(rule_line, []):
            rule_id = rule_info.get("id", "")
            display = _build_list_display(rule_info, rule_id)
            print_with_indent(display)


def handle_list_json_output(rule_list):
    print(json.dumps(rule_list, ensure_ascii=False), end="", flush=True)


def main():
    if args.log_path:
        output_path = Path(args.log_path).expanduser().absolute()
        output_path.mkdir(parents=True, exist_ok=True)
        DEFAULT_CONFIG["log_path"] = str(output_path)
        # 重新初始化 logger 以使用新路径
        tool.LOGGER = None
        get_logger()

    if args.lang:
        try:
            locale.setlocale(locale.LC_ALL, args.lang)
            DEFAULT_CONFIG["lang"] = args.lang
        except locale.Error as le:
            get_logger().warning(f"无法设置语言: {args.lang}，使用系统默认。 错误: {le}")
        get_logger().info(f"输出语言: {locale.getlocale()}")

    if args.output_path:
        output_path = Path(args.output_path).expanduser().absolute()
        output_path.mkdir(parents=True, exist_ok=True)
        DEFAULT_CONFIG["output_path"] = str(output_path)

    if args.command == "list":
        from minesweepervariants.impl import rule
        rule_list = rule.get_all_rules()

        if args.json:
            handle_list_json_output(rule_list)
        else:
            handle_list_text_output(rule_list)

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

    DEFAULT_CONFIG["log_file_name"] = args.file_name
    tool.LOGGER = None
    get_logger().info(
        f"题板信息: "
        f"SIZE:{args.size} "
        f"TOTAL:{args.total if args.total != -1 else '自动'} "
        f"DYE:{args.dye if args.dye else '空'} "
        f"MASK:{args.mask if args.mask else '空'}"
    )
    get_logger().info(f"使用规则: RULE:{args.rules} EARLY_RULE:{args.early_rules}")

    try:
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
                print(
                    "可选依赖`image`未安装，请使用`pip install minesweepervariants[image]`安装, 或者添加--no-image参数不绘制图片.")
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
    except Exception as e:
        import traceback
        get_logger().error("\n" + ''.join(traceback.format_exception(type(e), e, e.__traceback__)))
        raise e
    finally:
        get_logger().info("minesweepervariants END")


if __name__ == "__main__":
    main()
