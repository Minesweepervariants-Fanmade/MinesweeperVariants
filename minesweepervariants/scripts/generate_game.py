#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/06/17 08:52
# @Author  : xxx
# @FileName: generate_game.py
"""
普通模式-随机生成题板
"""
import base64
import os
import time

from minesweepervariants.impl.impl_obj import ModelGenerateError, get_board, encode_board
from minesweepervariants.impl.summon import Summon
from minesweepervariants.impl.summon.game import GameSession, PUZZLE
from minesweepervariants.impl.summon.summon import GenerateError
from minesweepervariants.utils import tool
from minesweepervariants.utils.image_create import draw_board
from minesweepervariants.utils.impl_obj import get_seed
from minesweepervariants.utils.tool import get_logger, get_random

from minesweepervariants.config.config import DEFAULT_CONFIG, PUZZLE_CONFIG

# ==== 获取默认值 ====
CONFIG = {}
CONFIG.update(DEFAULT_CONFIG)
CONFIG.update(PUZZLE_CONFIG)


def main(
        log_lv: str,  # 日志等级
        seed: int,  # 随机种子
        size: tuple[int, int],  # 题板尺寸
        total: int,  # 总雷数
        rules: list[str],  # 所有的规则集合
        query: str,  # 最高线索数范围 x-y表示
        attempts: int,  # 尝试次数
        dye: str,  # 染色规则
        mask_dye: str,  # 异形题板
        drop_r: bool,  # 在推理时候是否隐藏R推理
        early_stop: bool,  # 如果达到指定的线索数量 直接跳出
        board_class: str,  # 题板的类名
        vice_board: bool,  # 启用删除副板
        unseed: bool,  # 是否禁用种子来快速生成题目
        file_name: str = "",
        image: bool = True,  # 是否生成图片
):
    if "-" in query:
        a, b, *_ = query.split("-")
        a = int(a) if a.isdigit() else None
        b = int(b) if b.isdigit() else None
        query = [a, b]
    else:
        query = [int(query), None]
    if query[0] is not None and query[1] is not None and query[0] > query[1]:
        raise ValueError(f"范围从{query[0]}到{query[1]}吗? 你在干什么?")
    rule_code = rules[:]
    tool.LOGGER = None
    logger = get_logger(log_lv=log_lv)
    get_random(seed, new=True)
    s = Summon(size=size, total=total, rules=rules, board=board_class,
               drop_r=drop_r, mask=mask_dye, dye=dye, vice_board=vice_board)

    rule_text = ""
    for rule in rules:
        rule_text += "[" + (rule.split(CONFIG['delimiter'])[0] if
                            CONFIG['delimiter'] in rule else rule) + "]"
    if rule_text == "":
        rule_text = "[V]"
    if dye:
        rule_text += f"[{dye}]"
    if mask_dye:
        rule_text += f"[{mask_dye}]"

    size_a = 0
    size_b = 0
    size_c = len(s.board.get_interactive_keys())
    for key in s.board.get_interactive_keys():
        bound = s.board.boundary(key)
        size_a = max(size_a, bound.x + 1)
        size_b = max(size_b, bound.y + 1)
    rule_text += f"{size[0]}x{size[1]}"
    if size_c > 1:
        rule_text += f"x{size_c}"
    total = s.total
    logger.info(f"total mines: {total}")

    attempt_index = 0

    while True:
        s = Summon(size=size, total=total, rules=rule_code[:], board=board_class,
                   drop_r=drop_r, mask=mask_dye, dye=dye, vice_board=vice_board)
        if unseed:
            s.unseed = True
        get_random(seed, new=True)
        a_time = time.time()
        if attempts != -1 and attempt_index >= attempts:
            break
        attempt_index += 1
        logger.info(f"尝试第{attempt_index}次minesweepervariants..", end="\r")
        game = GameSession(s, mode=PUZZLE, drop_r=drop_r)
        try:
            game.board = s.create_puzzle()
        except ModelGenerateError:
            continue
        except GenerateError:
            continue
        game.answer_board = get_board(board_class)(rules={}, code=s.answer_board_code)
        # game.board = get_board("0B")(code=b'')
        # game.answer_board = get_board("0B")(code=b'')
        _board = game.board.clone()
        game.logger.info("\n" + "=" * 30 + "\nanswer_board:\n" + game.answer_board.show_board())
        game.logger.info("\nboard:\n" + game.board.show_board())
        game.logger.info("board: " + str(game.board.encode()))
        game.logger.info("answer: " + str(game.answer_board.encode()))
        try:
            clue_freq = game.check_difficulty(diff=query if early_stop else None)
        except ModelGenerateError:
            continue
        if not clue_freq:
            logger.warn("生成失败")
            continue
        game.logger.info("\n" + "=" * 50 + "\n" + str(clue_freq))
        time_used = time.time() - a_time
        n_num = len([None for _ in _board("N")])
        board_str = _board.show_board()
        answer = game.answer_board.show_board()
        board_code = _board.encode()
        answer_code = game.answer_board.encode()

        if query[0] is not None and max(clue_freq.keys()) < query[0]:
            continue
        if query[1] is not None and max(clue_freq.keys()) > query[1]:
            continue

        if not os.path.exists(CONFIG["output_path"]):
            os.makedirs(CONFIG["output_path"])

        mask = 0
        for key in _board.get_board_keys():
            for _, obj in _board(key=key):
                if obj is None:
                    mask += 1
                mask <<= 1

        # 计算需要的字节长度
        byte_length = (mask.bit_length() + 7) // 8  # 计算所需字节数
        byte_length = max(byte_length, 1)  # 确保至少 1 字节

        mask = mask.to_bytes(byte_length, "big", signed=False)

        rule_code = [base64.urlsafe_b64encode(rule.encode("utf-8")).decode("utf-8") for rule in rule_code]

        with (open(os.path.join(CONFIG["output_path"], f"{file_name}.txt" if file_name else "demo.txt"), "a", encoding="utf-8") as f):
            f.write("\n" + ("=" * 100) + "\n\n生成时间" + logger.get_time() + "\n")
            f.write(f'线索表\n')
            if 0 in clue_freq:
                f.write("存在线索数为0的可推格 请注意辨别\n")
            if early_stop:
                f.write("(已启用-e参数 线索图不准确)\n")
            f.write(f'{clue_freq}\n')
            f.write(f"生成用时:{time_used}s\n")
            f.write(f"总雷数: {total}/{n_num}\n")
            f.write(f"种子: {get_seed()}\n")
            f.write("\n" + rule_text)
            f.write("\n" + board_str)
            f.write("\n" + answer)

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

        if image:
            draw_board(board=get_board(board_class)(code=board_code), cell_size=100, output=file_name + "demo",
                    bottom_text=(rule_text + "-" + str(max(clue_freq.keys())) +
                                    f"-R{'*' if drop_r else total}/{n_num}" +
                                    ("\n" if unseed else f"-{get_seed()}\n")))

            draw_board(board=get_board(board_class)(code=answer_code), output=file_name + "answer", cell_size=100,
                    bottom_text=(rule_text + "-" + str(max(clue_freq.keys())) +
                                    f"-R{total}/{n_num}" +
                                    ("\n" if unseed else f"-{get_seed()}\n")))

        return
