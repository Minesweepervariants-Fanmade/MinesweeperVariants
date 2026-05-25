#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/07/06 05:43
# @Author  : Wu_RH
# @FileName: generate_test.py
import os
import threading
import time

from minesweepervariants.abs.board import Size, compress, json_dumps
from minesweepervariants.impl.summon import Summon
from minesweepervariants.utils.image_create import draw_board
from minesweepervariants.utils.impl_obj import get_seed
from minesweepervariants.utils.tool import get_logger, get_random
from minesweepervariants.utils import tool

from minesweepervariants.config.config import DEFAULT_CONFIG, PUZZLE_CONFIG


def _build_rule_text(board) -> str:
    rule_names = []
    seen = set(rule_names)
    hidden_defaults = {"R", "V", "F", "F#", "#"}

    rules = getattr(board, "rules", {})
    if isinstance(rules, dict):
        for rule_group in rules.values():
            for rule in rule_group:
                if rule is None:
                    continue
                rule_name = rule.get_name()
                if rule_name in seen or rule_name in hidden_defaults:
                    continue
                seen.add(rule_name)
                rule_names.append(rule_name)

    if not rule_names:
        rule_names = ["V"]

    return "".join(f"[{rule}]" for rule in rule_names)


def main(
        log_lv: str,  # 日志等级
        seed: int,  # 随机种子
        size: tuple[int, int],  # 题板尺寸
        total: int,  # 总雷数
        rules: list[str],  # 规则id集合
        early_rules: list[str],  # 仅初始生成阶段生效的左线规则
        dye: str,  # 染色规则
        mask_dye: str,   # 异形题板
        board_class: str,  # 题板的名称
        unseed: bool,   # 是否不启用种子
        image: bool,     # 是否生成图片
        attempts: int = -1,     # 最大尝试次数
):
    DEFAULT_CONFIG["log_file_name"] = ""
    tool.LOGGER = None
    logger = get_logger(log_lv=log_lv)
    get_random(seed, new=True)
    attempt_index = 0
    if isinstance(size, tuple):
        size = Size(size[0], size[1])
    s = Summon(
        size=size, total=total, rules=rules, early_rules=early_rules,
        board=board_class, mask=mask_dye, dye=dye, unseed=unseed
    )
    unseed = s.unseed
    total = s.total
    logger.info(f"total mines: {total}")
    _board = None
    while True:
        if attempt_index == attempts:
            break
        attempt_index += 1
        logger.info(f"尝试第{attempt_index}次minesweepervariants..", end="\r")
        get_random(seed, new=True)
        a_time = time.time()
        _board = s.summon_board()
        if _board is None and not unseed:
            raise ValueError("左线/总雷数非法")
        if _board is None:
            continue
        logger.info(f"<{attempt_index}>生成用时:{(time_used := time.time() - a_time)}s")
        logger.info(f"总雷数: {s.total}")
        logger.info("\n" + _board.show_board())

        rule_text = _build_rule_text(s.answer_board)
        if dye:
            rule_text += f"[@{dye}]"
        rule_text += f"{size.cols}x{size.rows}"

        if not os.path.exists(DEFAULT_CONFIG["output_path"]):
            os.mkdir(DEFAULT_CONFIG["output_path"])

        with open(os.path.join(DEFAULT_CONFIG["output_path"], "demo.txt"), "a", encoding="utf-8") as f:
            f.write("\n" + ("=" * 100) + "\n\n生成时间" + logger.get_time() + "\n")
            f.write(f"生成用时:{time_used}s\n")
            f.write(f"总雷数: {s.total}\n")
            f.write(f"种子: {get_seed()}\n")
            f.write(rule_text)
            f.write("\n"+_board.show_board())

            f.write(f"\n答案: img -c {compress(json_dumps(_board.json()))} ")
            f.write(f"-r \"{rule_text}-R{s.total}")
            if unseed:
                f.write(" ")
            else:
                f.write(f"-{get_seed()}\" ")
            f.write("-o answer\n")

        if image:
            def d():
                draw_board(
                    board=_board, cell_size=100, output="answer",
                    bottom_text=rule_text + f"-R{s.total}-{get_seed()}\n"
                )
            if attempt_index != attempts:
                threading.Thread(target=d, daemon=True).start()
            else:
                d()

        logger.info("\n\n" + "=" * 20 + "\n")
        logger.info("\n生成时间" + logger.get_time() + "\n")
        logger.info(f"生成用时:{time_used}s\n")
        logger.info(f"总雷数: {s.total}\n")
        logger.info("\n" + _board.show_board() + "\n")

        if attempt_index != attempts:
            input("检查完毕后输入回车继续尝试 使用ctrl+c终止进程\r")
