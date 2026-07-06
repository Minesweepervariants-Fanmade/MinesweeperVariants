#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/06/03 05:23
# @Author  : Wu_RH
# @FileName: run.py
# @Version : 1.0.0
import argparse
import json
import gettext
import shutil
import textwrap
import sys
import locale
from pathlib import Path
from importlib.util import find_spec
from typing import Callable

from minesweepervariants import puzzle_query
from minesweepervariants import puzzle
from minesweepervariants import test
from minesweepervariants import hint
from minesweepervariants import img
from minesweepervariants import game
from minesweepervariants import ocr

from minesweepervariants.config.config import DEFAULT_CONFIG
from minesweepervariants.size import Size
from minesweepervariants.utils.tool import get_logger
from minesweepervariants.utils import tool
from minesweepervariants.utils.i18n import init_gettext

# ==== 获取默认值 ====
defaults = {}
defaults.update(DEFAULT_CONFIG)

pre_parser = argparse.ArgumentParser(add_help=False)
pre_parser.add_argument("--lang", default=defaults.get("lang"))
pre_args = pre_parser.parse_known_args()[0]
_: Callable[..., str] = init_gettext(pre_args.lang)

# ==== 参数解析 ====
parser = argparse.ArgumentParser(description="")

subparsers = parser.add_subparsers(dest='command', required=False)

parser_list = subparsers.add_parser('list', help=_('CLI_LIST_RULE_DOCS'))

parser_hint = subparsers.add_parser("hint", help="根据输入的内容进行逐步提示操作")

parser_img = subparsers.add_parser("img", help="根据输入的内容进行逐步提示操作")

parser_ocr = subparsers.add_parser("ocr", help="根据输入的图片路径使用ocr将其转为内置的board模板")

parser_img.add_argument("-c", "--code", type=str,
                        help="题板字节码")  # 字符串类型
parser_img.add_argument("-r", "--rule-text", type=str, default="",
                        help="规则字符串，有空格需要带引号")  # 字符串类型
parser_img.add_argument("-o", "--output", type=str, default=defaults["output_file"],
                        help="输出的文件名（不含后缀）")  # 字符串类型
parser_img.add_argument("-s", "--size", type=int, default=defaults["cell_size"],
                        help="单元格像素数")  # 整数
parser_img.add_argument("-b", "--because", nargs="+", default=[],
                        help="从什么格子推出")
parser_img.add_argument("-d", "--deduced", nargs="+", default=[],
                        help="可推出什么格子")
parser_img.add_argument("-w", "--white-base", action="store_true", default=defaults["white_base"],
                        help="题板背景是否是白底的")

parser.add_argument("-s", "--size", nargs="+",
                    help=_("CLI_BOARD_SIZE"))
parser.add_argument("-t", "--total", type=int, default=defaults.get("total"),
                    help=_("CLI_TOTAL_MINES"))
parser.add_argument("-c", "--rules", nargs="+", default=[],
                    help=_("CLI_RULE_NAMES"))
parser.add_argument("-E", "--early-rules", nargs="+", default=[],
                    help=_("CLI_EARLY_RULE_NAMES"))
parser.add_argument("-d", "--dye", default=defaults.get("dye"),
                    help=_("CLI_DYE_RULE_NAME"))
parser.add_argument("-m", "--mask", default=defaults.get("dye"),
                    help=_("CLI_MASK_RULE_NAME"))
parser.add_argument("-W", "--workes-number", type=int, default=defaults.get("workes_number"),
                    help="多线程数量")
parser.add_argument("-r", "--used-r", action="store_true", default=defaults.get("used_r"),
                    help=_("CLI_USED_R"))
parser.add_argument("-a", "--attempts", type=int, default=defaults.get("attempts"),
                    help=_("CLI_ATTEMPTS"))
parser.add_argument("-q", "--query", default="",
                    help=_("CLI_QUERY_RANGE"))
parser.add_argument("-e", "--early-stop", action="store_true", default=False,
                    help=_("CLI_EARLY_STOP"))
parser.add_argument("-v", "--vice-board", action="store_true", default=False,
                    help=_("CLI_VICE_BOARD"))
parser.add_argument("-T", "--test", action="store_true", default=False,
                    help=_("CLI_TEST_ONE_BOARD"))
parser.add_argument("-S", "--seed", type=int, default=defaults.get("seed"),
                    help=_("CLI_SEED"))
parser.add_argument("-O", "--onseed", action="store_true", default=False,
                    help=_("CLI_REPRODUCIBLE_SEED"))
parser.add_argument("-L", "--log-lv", default=defaults.get("log_lv"),
                    help=_("CLI_LOG_LEVEL"))
parser.add_argument("-B", "--board-class", default=defaults.get("board_class"),
                    help=_("CLI_BOARD_CLASS"))
parser.add_argument("-I", "--no-image", action="store_true", default=defaults.get("no_image"),
                    help=_("CLI_NO_IMAGE"))
parser.add_argument("-F", "--file-name", default="",
                    help=_("CLI_FILE_PREFIX"))
parser.add_argument("-D", "--dynamic-dig-rounds", type=int, default=None,
                    help=_("CLI_DYNAMIC_DIG_ROUNDS"))
parser.add_argument("-M", "--dynamic-dig-max-batch", type=int, default=defaults.get("dynamic_dig_max_batch"),
                    help=_("CLI_DYNAMIC_DIG_MAX_BATCH"))
parser.add_argument("--mus-min-size", action="store_true", default=False,
                    help="如果启用的话 那么表示必须找到最少亮起线索数量的题板(仅在线索实现合法且启用with-mus参数时生效)")
parser.add_argument("--with-mus", action="store_true", default=False,
                    help="如果启用的话 那么表示使用最小冲突集来寻找题板怎么摆放线索(仅在线索实现合法时生效)")

parser.add_argument("--output-path", default=None,
                    help=_("CLI_OUTPUT_PATH"))
parser.add_argument("--log-path", default=None,
                    help=_("CLI_LOG_PATH"))
parser.add_argument("--lang", default=defaults.get("lang"),
                    help=_("CLI_LANG"))

parser_hint.add_argument("-b", "--board-code", type=str,
                         help="题板字节码")  # 字符串类型
parser_hint.add_argument("-a", "--answer-code", type=str,
                         help="答案题板字节码")  # 字符串类型
parser_hint.add_argument("-c", "--rules", nargs="+", default=[],
                         help="规则字符串，有空格/单引号/双引号等特殊字符需要带引号")  # 字符串类型
parser_hint.add_argument("-r", "--used-r", action="store_true", default=defaults.get("used_r"),
                         help=_("CLI_USED_R"))
parser_hint.add_argument("-m", "--game-mode", type=str, default=defaults.get("game_mode"),
                         help="枚举值, game的游戏模式[专家:EXPERT/终极:ULTIMATE/纸笔:PUZZLE]")
parser_hint.add_argument("-t", "--total", type=int, default=-1,
                         help="总雷数的数量")
parser_hint.add_argument("-W", "--workes-number", type=int, default=defaults.get("workes_number"),
                         help="多线程数量")

parser_hint.add_argument("-B", "--board-class", default=defaults.get("board_class"),
                         help="题板的类名/题板的名称 通常使用默认值即可")
parser_hint.add_argument("-I", "--no-image", action="store_true", default=defaults.get("no_image"),
                         help="是否不生成图片")
parser_hint.add_argument("-L", "--log-lv", default=defaults.get("log_lv"),
                         help="日志输出目录路径，日志将保存到此目录（默认使用配置中的路径）")
parser_hint.add_argument("-F", "--file-name", default=defaults.get("hint_file"),
                         help="文件名的前缀")
parser_hint.add_argument("--output-path", default=defaults["output_file"],
                         help="图片输出目录路径，图片将保存到此目录（默认使用配置中的路径）")
parser_hint.add_argument("--log-path", default=None,
                         help="日志输出目录路径，日志将保存到此目录（默认使用配置中的路径）")

parser_game = subparsers.add_parser("game", help="交互式游戏 CLI，命令: r/f/h/q")

parser_game.add_argument("-b", "--board-code", type=str,
                         help="题板字节码")
parser_game.add_argument("-a", "--answer-code", type=str,
                         help="答案题板字节码")
parser_game.add_argument("-c", "--rules", nargs="+", default=[],
                         help="规则字符串，有空格/单引号/双引号等特殊字符需要带引号")
parser_game.add_argument("-r", "--used-r", action="store_true", default=defaults.get("used_r"),
                         help=_("CLI_USED_R"))
parser_game.add_argument("-m", "--game-mode", type=str, default=defaults.get("game_mode"),
                         help="游戏模式: EXPERT/ULTIMATE/PUZZLE")
parser_game.add_argument("-t", "--total", type=int, default=-1,
                         help="总雷数的数量")
parser_game.add_argument("-B", "--board-class", default=defaults.get("board_class"),
                         help="题板的类名/题板的名称")
parser_game.add_argument("-I", "--no-image", action="store_true", default=defaults.get("no_image"),
                         help="是否不生成图片")
parser_game.add_argument("-L", "--log-lv", default=defaults.get("log_lv"),
                         help="日志等级")
parser_game.add_argument("-F", "--file-name", default=defaults.get("hint_file"),
                         help="图片文件名前缀")
parser_game.add_argument("-W", "--workes-number", type=int, default=defaults.get("workes_number"),
                         help="多线程数量")

parser_ocr.add_argument("-p", "-i", "--img-path", type=str, help="图片的路径参数")

parser_ocr.add_argument("-c", "--clue-rule", nargs="+", default=[],
                        help="右线的规则参数, 用以匹配ocr(不支持#等多右)")
parser_ocr.add_argument("-L", "--log-lv", default=defaults.get("log_lv"),
                        help="日志输出目录路径，日志将保存到此目录（默认使用配置中的路径）")
parser_ocr.add_argument("-f", "--file-name", default="",
                        help="图片文件名前缀")

parser_list.add_argument("--json", action="store_true", default=False)
parser_list.add_argument("-F", "--file-path", type=str, default="")

args, unknown = parser.parse_known_args()
if args.command != "ocr":
    args = parser.parse_args()


# ==== 调用生成 ====


def print_with_indent(text, file_stream, indent="\t"):
    width = shutil.get_terminal_size(fallback=(80, 24)).columns // 2
    # 减去缩进长度，避免超宽
    effective_width = width - len(indent.expandtabs())
    lines = text.splitlines()
    for line in lines:
        wrapped = textwrap.fill(line, width=effective_width,
                                initial_indent=indent,
                                subsequent_indent=indent)
        print(wrapped, flush=True, file=file_stream)
    print(file=file_stream)


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


def handle_list_text_output(rule_list, file=None):
    rule_line_name_map = {
        "L": "\n\n" + _("OUT_LEFT_RULES"),
        "M": "\n\n" + _("OUT_MIDDLE_RULES"),
        "R": "\n\n" + _("OUT_RIGHT_RULES"),
    }
    if type(file) is str:
        file: str
        file_stream = open(file, "w", encoding="utf-8")
    elif file is None:
        file_stream = sys.stdout
    else:
        file_stream = file

    for rule_line in ["L", "M", "R"]:
        if rule_list.get(rule_line):
            print(rule_line_name_map[rule_line], flush=True, file=file_stream)
        for rule_info in rule_list.get(rule_line, []):
            rule_id = rule_info.get("id", "")
            display = _build_list_display(rule_info, rule_id)
            print_with_indent(display, file_stream=file_stream)


def handle_list_json_output(rule_list, file=None):
    if type(file) is str:
        file: str
        file_stream = open(file, "w", encoding="utf-8")
    elif file is None:
        file_stream = sys.stdout
    else:
        file_stream = file
    print(json.dumps(rule_list, ensure_ascii=False), end="", flush=True, file=file_stream)


def main():
    if args.log_path:
        output_path = Path(args.log_path).expanduser().absolute()
        output_path.mkdir(parents=True, exist_ok=True)
        DEFAULT_CONFIG["log_path"] = str(output_path)
        # 重新初始化 logger 以使用新路径
        tool.LOGGER = None
        get_logger()

    if args.with_mus:
        DEFAULT_CONFIG["with_mus"] = True

    if args.mus_min_size:
        if args.with_mus:
            DEFAULT_CONFIG["mus_min_size"] = True
        else:
            get_logger().warning("未启用--with-mus参数 mus_min_size不生效")

    if args.output_path:
        output_path = Path(args.output_path).expanduser().absolute()
        output_path.mkdir(parents=True, exist_ok=True)
        DEFAULT_CONFIG["output_path"] = str(output_path)

    if args.command == "list":
        from minesweepervariants.impl import rule
        rule_list = rule.get_all_rules()

        if args.json:
            handle_list_json_output(rule_list, file=args.file_path or None)
        else:
            handle_list_text_output(rule_list, file=args.file_path or None)

        return
    DEFAULT_CONFIG["workes_number"] = args.workes_number

    if args.command == "img":
        img(
            code=args.code,
            rule_text=args.rule_text,
            output=args.output,
            white_base=args.white_base,
            size=args.size,
            because=args.because,
            deduced=args.deduced,
        )
        return
    if args.command == "hint":
        hint(
            board_code=args.board_code,
            answer=args.answer_code,
            rules=args.rules,
            drop_r=not args.used_r,
            board_class=args.board_class,
            file_name=args.file_name,
            log_lv=args.log_lv,
            no_image=args.no_image,
            game_mode=args.game_mode,
            total=args.total,
        )
        return

    if args.command == "game":
        game(
            board_code=args.board_code,
            answer=args.answer_code,
            rules=args.rules,
            board_class=args.board_class,
            file_name=args.file_name,
            log_lv=args.log_lv,
            no_image=args.no_image,
            drop_r=not args.used_r,
            game_mode=args.game_mode,
            total=args.total,
        )
        return

    if args.command == "ocr":
        ocr(
            img_path=args.img_path,
            rules_id=args.clue_rule,
            log_lv=args.log_lv,
            file_name=args.file_name,
        )
        return

    if args.size is None:
        parser.print_help()
        return
    else:
        if len(args.size) == 0:
            parser.print_help()
            return
        elif len(args.size) == 1:
            size = Size(int(args.size[0]), int(args.size[0]))
        else:
            size = Size(cols=int(args.size[0]), rows=int(args.size[1]))

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
        f"{_('OUT_BOARD_INFO')}"
        f"{_('OUT_SIZE')}{args.size} "
        f"{_('OUT_TOTAL')}{args.total if args.total != -1 else _('OUT_AUTO')} "
        f"{_('OUT_DYE')}{args.dye if args.dye else _('OUT_EMPTY')} "
        f"{_('OUT_MASK')}{args.mask if args.mask else _('OUT_EMPTY')}"
    )
    get_logger().info(f"{_('OUT_USED_RULES')}{args.rules}{_('OUT_EARLY_RULE')}{args.early_rules}")
    get_logger().info(f"{_('OUT_USED_R')}[USED_R]: {args.used_r}")

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
                attempts=args.attempts,
                file_name=args.file_name,
            )
        elif not args.query:
            if not args.no_image and find_spec("PIL") is None:
                print(_("OUT_IMAGE_MISSING"))
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
        get_logger().info(_("OUT_END"))


if __name__ == "__main__":
    main()
