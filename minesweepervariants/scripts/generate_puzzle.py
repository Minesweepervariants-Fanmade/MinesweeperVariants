#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/06/15 18:07
# @Author  : Wu_RH
# @FileName: generate_puzzle.py.py
import base64
import os
import time

from minesweepervariants.impl.impl_obj import get_board, ModelGenerateError, encode_board
from minesweepervariants.impl.summon import Summon
from minesweepervariants.impl.summon.summon import GenerateError
from minesweepervariants.utils import timer
from minesweepervariants.utils.image_create import draw_board
from minesweepervariants.utils.tool import get_logger, get_random
from minesweepervariants.utils import tool
from minesweepervariants.utils.impl_obj import get_seed

from minesweepervariants.config.config import DEFAULT_CONFIG, PUZZLE_CONFIG

# ==== 获取默认值 ====
CONFIG = {}
CONFIG.update(DEFAULT_CONFIG)
CONFIG.update(PUZZLE_CONFIG)


def main(
        log_lv: str,  # 日志等级
        seed: int,  # 随机种子
        attempts: int,  # 尝试次数
        size: tuple[int, int],  # 题板尺寸
        total: int,  # 总雷数
        rules: list[str],  # 规则id集合
        dye: str,  # 染色规则
        mask_dye: str,  # 异形题板
        drop_r: bool,  # 在推理时候是否隐藏R推理
        board_class: str,  # 题板的名称
        vice_board: bool,  # 启用删除副板
        unseed: bool,  # 是否抛弃seed来生成
        image: bool,  # 是否生成图片
        file_name: str,  # 文件已什么开头
):
    rule_code = rules[:]
    tool.LOGGER = None
    logger = get_logger(log_lv=log_lv)
    get_random(seed, new=True)
    s = Summon(size=size, total=total, rules=rules, board=board_class,
               drop_r=drop_r, mask=mask_dye, dye=dye, vice_board=vice_board)
    if unseed:
        s.unseed = True
    total = s.total
    logger.info(f"total mines: {total}")
    _board = None
    info_list = []
    print(rule_code)
    attempt_index = 0
    while attempts == -1 or attempt_index < attempts:
        attempt_index += 1
        s = Summon(size=size, total=total, rules=rule_code[:], board=board_class,
                   drop_r=drop_r, mask=mask_dye, dye=dye, vice_board=vice_board)
        if unseed:
            s.unseed = True
        logger.info(f"尝试第{attempt_index}次minesweepervariants..", end="\r")
        get_random(seed, new=True)
        a_time = time.time()
        try:
            _board = s.create_puzzle()
        except ModelGenerateError:
            continue
        except GenerateError:
            continue
        b_time = time.time()
        if _board is None:
            continue
        n_num = len([None for _ in _board("N")])
        logger.info(f"<{attempt_index}>生成用时:{b_time - a_time}s")
        logger.info(f"总雷数: {total}/{n_num}")
        logger.info("\n" + _board.show_board())
        if len([None for _ in _board("NF")]) == total:
            logger.warn("题板生成失败 线索填充无法覆盖全盘")
            continue
        info_list.append([
            b_time - a_time,
            n_num,
            "\n" + _board.show_board(),
            _board.encode(),
            "\n" + s.answer_board_str,
            s.answer_board_code,
            _board
        ])

        break

    if not info_list:
        raise ValueError("未在有效次数内得出结果")

    info_list.sort(key=lambda x: x[0])
    time_used, n_num, board_str, board_code, answer, answer_code, _board = info_list[0]

    rule_text = ""
    for rule in rules:
        rule_text += "[" + rule + "]"
    if rule_text == "":
        rule_text = "[V]"
    if dye:
        rule_text += f"[@{dye}]"
    if mask_dye:
        rule_text += f"[&{mask_dye}]"
    size_a = 0
    size_b = 0
    size_c = len(_board.get_interactive_keys())
    for key in _board.get_interactive_keys():
        bound = _board.boundary(key)
        size_a = max(size_a, bound.x + 1)
        size_b = max(size_b, bound.y + 1)
    rule_text += f"{size_a}x{size_b}"
    if size_c > 1:
        rule_text += f"x{size_c}"

    if not os.path.exists(CONFIG["output_path"]):
        os.mkdir(CONFIG["output_path"])

    i, mask = 1, 0
    for key in _board.get_board_keys():
        for _, obj in _board(key=key):
            if obj is None:
                mask += i
            i <<= 1

    # 计算需要的字节长度
    byte_length = (mask.bit_length() + 7) // 8  # 计算所需字节数
    byte_length = max(byte_length, 1)  # 确保至少 1 字节

    mask = mask.to_bytes(byte_length, "big", signed=False)
    rule_code = [base64.urlsafe_b64encode(rule.encode("utf-8")).decode("utf-8") for rule in rule_code]

    with open(os.path.join(CONFIG["output_path"], f"{file_name or 'demo'}.txt"), "a", encoding="utf-8") as f:
        f.write("\n" + ("=" * 100) + "\n\n生成时间" + logger.get_time() + "\n")
        f.write(f"生成用时:{time_used}s\n")
        f.write(f"总雷数: {total}/{n_num}\n")
        f.write(f"种子: {get_seed()}\n")
        f.write(rule_text)
        f.write(board_str)
        f.write(answer)

        f.write(f"\n答案: img -c {encode_board(answer_code)} ")
        f.write(f"-r \"{rule_text}-R{total}/")
        f.write(f"{n_num}")
        if unseed:
            f.write(f"-{get_seed()}\" ")
        else:
            f.write(" ")
        f.write("-o answer\n")

        f.write(f"\n题板: img -c {encode_board(board_code)} ")
        f.write(f"-r \"{rule_text}-R{'*' if drop_r else total}/")
        f.write(f"{n_num}")
        if unseed:
            f.write(f"-{get_seed()}\" ")
        else:
            f.write(" ")
        f.write("-o demo\n")

        f.write(f"\n题板代码: \n{encode_board(answer_code)}:{mask.hex()}:{':'.join(rule_code)}\n")

        f.write(f"\n总求解用时: \n{sum(d['time'] for d in timer.HISTORY)}s\n")

        f.write("\n求解记录:\n")
        for d in timer.HISTORY:
            f.write(f"求解状态: {d['result']}, 用时: {d['time']}s\n")

    if image:
        draw_board(board=get_board(board_class)(code=board_code), cell_size=100, output=file_name + "demo",
                bottom_text=(rule_text +
                                f"-R{'*' if drop_r else total}/{n_num}" +
                                ("\n" if unseed else f"-{get_seed()}\n")))

        draw_board(board=get_board(board_class)(code=answer_code), output=file_name + "answer", cell_size=100,
                bottom_text=(rule_text +
                                f"-R{total}/{n_num}" +
                                ("\n" if unseed else f"-{get_seed()}\n")))

    logger.info("\n\n" + "=" * 20 + "\n")
    logger.info("\n生成时间" + logger.get_time() + "\n")
    logger.info(f"生成用时:{time_used}s\n")
    logger.info(f"总雷数: {total}/{n_num}\n")
    logger.info(board_str + "\n")
    logger.info(answer + "\n")
    logger.info(f"{board_code}")
