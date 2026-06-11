#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2026/05/24 14:03
# @Author  : Wu_RH
# @FileName: hint.py

import threading
import time

from minesweepervariants.json_object import json_loads, decompress
from minesweepervariants.position import Position
from minesweepervariants.config.config import DEFAULT_CONFIG
from minesweepervariants.impl.impl_obj import decode_board
from minesweepervariants.impl.summon import Summon
from minesweepervariants.impl.summon.game import GameSession, Mode, UMode, max_disjoint_lists
from minesweepervariants.size import Size
from minesweepervariants.utils import tool
from minesweepervariants.utils.image_create import draw_board
from minesweepervariants.utils.tool import get_logger

defaults = {}
defaults.update(DEFAULT_CONFIG)


def progress(global_map):
    def format_time(seconds):
        seconds = int(round(seconds))  # 四舍五入取整
        days, seconds = divmod(seconds, 86400)  # 1天=86400秒
        hours, seconds = divmod(seconds, 3600)  # 1小时=3600秒
        minutes, seconds = divmod(seconds, 60)  # 1分钟=60秒

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0 or days > 0:  # 如果有天，即使小时=0也要显示
            parts.append(f"{hours:02d}")
        if minutes > 0 or hours > 0 or days > 0:  # 如果有更高单位，分钟必须显示
            parts.append(f"{minutes:02d}")
        if (days + hours + minutes) > 0:
            parts.append(f"{seconds:02d}")  # 秒始终显示
        else:
            parts.append(f"{seconds}s")

        return ":".join(parts)

    start_time = time.time()
    _n_num = global_map['n_length']
    while True:
        n_length = global_map['n_length']
        elapsed = time.time() - start_time
        avg_time = (elapsed / (_n_num - n_length)) if (_n_num - n_length) > 0 else 0  # 平均每项耗时
        predicted_total = avg_time * _n_num  # 预计总时间
        print(
            f"线索: {str(global_map['clue_freq']).replace(' ', '')}  "
            f"进度: {_n_num - n_length}/{_n_num}  "
            f"用时:{format_time(elapsed)}"
            f"<{format_time(predicted_total)}"
            f"~{format_time(predicted_total - elapsed)}      ",
            end="\r", flush=True
        )
        if global_map['n_length'] == 0:
            break
        time.sleep(0.2)


def main(
    board_code: str,
    answer: str,
    rules: list[str],
    board_class: str,
    file_name: str,
    log_lv: str,
    no_image: bool,
    drop_r: bool,
    game_mode: str,
    total: int,
):
    DEFAULT_CONFIG["log_file_name"] = file_name
    tool.LOGGER = None
    logger = get_logger("hint", log_lv)
    if not board_code:
        raise ValueError("未输入待提示的题板")
    match game_mode:
        case "expert" | "EXPERT" | "专家":
            game_session_mode = Mode.EXPERT
        case "ultimate" | "ULTIMATE" | "终极模式":
            game_session_mode = Mode.ULTIMATE
        case "puzzle" | "PUZZLE" | "纸笔":
            game_session_mode = Mode.PUZZLE
        case _:
            raise ValueError(f"unknown game mode: {game_mode}")

    s = Summon(
        size=Size(0, 0), total=total, rules=rules[:],
        board=board_class, drop_r=drop_r, board_data=json_loads(decompress(board_code))
    )

    mask_board = s.board

    if answer:
        answer_board = decode_board(data=json_loads(decompress(answer)))
    else:
        answer_board = None

    if answer_board is None or None in [obj for _, obj in answer_board(mode="obj")]:
        from ortools.sat.python import cp_model

        from minesweepervariants.impl.summon.solver import Switch, get_solver
        from minesweepervariants.abs.rule import AbstractRule, AbstractValue
        from minesweepervariants.abs.Lrule import Rule0R

        # board = s.board.clone()
        answer_board = answer_board if answer_board else s.board.clone()
        for pos, obj in s.board(mode="obj"):
            if answer_board[pos] is None:
                if obj:
                    answer_board[pos] = obj
            elif answer_board[pos].json() != obj.json():
                raise ValueError(
                    f"传入部分答案题板与输入题板不符: POS[{pos}] "
                    f"({obj} != {answer_board[pos].json()})"
                )
        answer_board.clear_variable()
        model = answer_board.get_model()
        switch = Switch()
        # 2.获取所有规则约束
        for rule in (
                s.mines_rules.rules +
                [s.clue_rule, s.mines_clue_rule]
        ):
            if rule is None:
                continue
            if drop_r and isinstance(rule, Rule0R):
                continue
            rule: AbstractRule
            rule.create_constraints(answer_board, switch)

        for key in answer_board.get_board_keys():
            for pos, obj in answer_board(key=key, mode="obj"):
                if obj is None:
                    continue
                obj: AbstractValue
                obj.create_constraints(answer_board, switch)

        # 3.获取所有变量并赋值已解完的部分
        for key in answer_board.get_board_keys():
            for _, var in answer_board("C", mode="variable", key=key, special='raw'):
                model.add(var == 0)
                logger.trace(f"var: {var} == 0")
            for _, var in answer_board("F", mode="variable", key=key, special='raw'):
                model.add(var == 1)
                logger.trace(f"var: {var} == 1")

        model: cp_model.CpModel
        # model.AddAssumptions(switch.var_map.values())
        for switch_var in switch.get_all_vars():
            model.add(switch_var == 1)
        solver = get_solver(False)
        status = solver.solve(model)
        if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            logger.error("键入题板无解")
            raise ValueError("input board is not feasible")

        for key in answer_board.get_board_keys():
            for pos, var in answer_board(mode="var", key=key):
                if answer_board[pos] is not None:
                    continue
                if solver.Value(var) == 1:
                    answer_board[pos] = answer_board.get_config(key, "MINES")
                else:
                    answer_board[pos] = answer_board.get_config(key, "VALUE")

    logger.info(f"题板内容:\n{mask_board}")
    logger.info(f"答案题板:\n{answer_board}")

    s.answer_board = answer_board

    game = GameSession(s, mode=game_session_mode, drop_r=drop_r)
    game.mode = game_session_mode
    if game_session_mode == Mode.ULTIMATE:
        game.ultimate_mode = (
            UMode.ULTIMATE_A |
            UMode.ULTIMATE_F |
            UMode.ULTIMATE_S |
            UMode.ULTIMATE_R |
            UMode.ULTIMATE_P
        )
    game.board = s.board
    game.answer_board = s.answer_board

    deduced = game.deduced()
    undeduced = []
    for pos, obj in mask_board(mode="obj"):
        if obj is not None:
            continue
        if pos in deduced:
            continue
        if game_session_mode is Mode.PUZZLE:
            undeduced.append(pos)
    if undeduced:
        logger.warning(f"当前为纸笔[PUZZLE]模式 但是存在一些格子无法推出{undeduced} 游戏模式自动切为终极[ULTIMATE]")
        game.mode = Mode.ULTIMATE
        game.ultimate_mode = (
            UMode.ULTIMATE_A |
            UMode.ULTIMATE_F |
            UMode.ULTIMATE_S |
            UMode.ULTIMATE_R |
            UMode.ULTIMATE_P
        )

    clue_freq = {}
    hint_times = 0

    thread_list = []

    if not no_image:
        thread = threading.Thread(
            target=draw_board,
            kwargs={
                "board": game.board.clone(),
                "output": (file_name if file_name else "hint") + "_" + str(hint_times)
            }
        )
        thread.start()
        thread_list.append(thread)
    hint_times += 1
    n_num = len([None for key in mask_board.get_board_keys()
                 for _ in mask_board('N', key=key)])
    global_map = {"n_length": n_num, "clue_freq": clue_freq}
    progress_thread = threading.Thread(target=progress, args=(global_map, ), daemon=True)
    progress_thread.start()
    while game.deduced():
        global_map["n_length"] = len([None for key in mask_board.get_board_keys() for _ in mask_board('N', key=key)])
        hints = game.hint()
        [logger.debug(f"{i[0]} -> {i[1]}") for i in hints.items()]
        if not hints:
            logger.error("hint返回空 deduced仍然存在可推格 待检查规则/副板未框定")
            break
        minsize = min([len(k) for k in hints])
        hints: dict[tuple, list[Position]] = {
            because: deduceds
            for because, deduceds in hints.items()
            if len(because) == minsize
        }
        apply_hint = max_disjoint_lists(hints)
        grouped_hints = {}
        for apply in apply_hint:
            for b, t in hints.items():
                if t == apply:
                    grouped_hints[b] = apply
                    break
        pos_clues = [item for deduced in apply_hint for item in deduced]

        if minsize not in clue_freq:
            clue_freq[minsize] = 0
        clue_freq[minsize] += len(apply_hint)
        logger.info(f"clue freq now: {clue_freq}")

        for hint_because, hint_deduced in grouped_hints.items():
            logger.info(f"{hint_because} -> {hint_deduced}")

        if not no_image:
            for hint_because in grouped_hints:
                bottom_text = []
                for b in hint_because:
                    if isinstance(b, tuple):
                        bottom_text.append(b[0] + (f":{b[1]}" if b[1] else ""))
                bottom_text = "; ".join(bottom_text)
                thread = threading.Thread(
                    target=draw_board,
                    kwargs={
                        "board": game.board.clone(),
                        "bottom_text": bottom_text,
                        "output": (file_name if file_name else "hint") + "_" + str(hint_times),
                        "hint_because": [pos for pos in hint_because if isinstance(pos, Position)],
                        "hint_deduced": grouped_hints[hint_because]
                    }
                )
                thread.start()
                thread_list.append(thread)
                hint_times += 1
        logger.info(f"当前题板:\n{game.board}")

        for pos in pos_clues:
            imposs = game.answer_board.get_type(pos, special='raw')
            game.apply(pos, 0 if imposs == "C" else 1)

    if not no_image:
        thread = threading.Thread(
            target=draw_board,
            kwargs={
                "board": game.board.clone(),
                "output": (file_name if file_name else "hint") + "_" + str(hint_times)
            }
        )
        thread.start()
        thread_list.append(thread)
    hint_times += 1
    global_map["n_length"] = 0
    progress_thread.join()
    for thread in thread_list:
        thread.join()

    logger.info(f"最终题板:\n{game.board}")
    logger.info(f"线索图:{clue_freq}")

    return
