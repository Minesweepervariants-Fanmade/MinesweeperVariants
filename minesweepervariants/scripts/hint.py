#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2026/05/24 14:03
# @Author  : Wu_RH
# @FileName: hint.py

import argparse
import threading

from minesweepervariants.abs.board import Size
from minesweepervariants.config.config import DEFAULT_CONFIG
from minesweepervariants.impl.impl_obj import get_board, decode_board
from minesweepervariants.impl.summon import Summon
from minesweepervariants.impl.summon.game import GameSession, Mode, UMode
from minesweepervariants.utils.image_create import draw_board
from minesweepervariants.utils.tool import get_logger

defaults = {}
defaults.update(DEFAULT_CONFIG)


def main(
    board_code: str,
    answer: str,
    rules: list[str],
    board_class: str,
    log_path: str,
    output_path: str,
    file_name: str,
    log_lv: str,
    no_image: bool,
    drop_r: bool,
    game_mode: str,
):
    logger = get_logger("hint", log_lv)
    if not board_code:
        raise ValueError("未输入待提示的题板")
    s = Summon(
        size=Size(1, 1), total=-2, rules=rules[:],
        board=board_class, drop_r=drop_r
    )
    match game_mode:
        case "expert" | "EXPERT" | "专家":
            game_session_mode = Mode.EXPERT
        case "ultimate" | "ULTIMATE" | "终极模式":
            game_session_mode = Mode.ULTIMATE
        case "puzzle" | "PUZZLE" | "纸笔":
            game_session_mode = Mode.PUZZLE
        case _:
            raise ValueError(f"unknown game mode: {game_mode}")

    try:
        mask_board = decode_board(board_code)
    except:
        code = bytes.fromhex(board_code)
        mask_board = get_board(board_class)(rules=s.board.rules, code=code)

    s.board = mask_board

    if answer:
        try:
            answer_board = decode_board(answer)
        except:
            code = bytes.fromhex(answer)
            answer_board = get_board(board_class)(rules=s.board.rules, code=code)
    else:
        from ortools.sat.python import cp_model

        from minesweepervariants.impl.summon.solver import Switch, get_solver
        from minesweepervariants.abs.rule import AbstractRule, AbstractValue
        from minesweepervariants.abs.Lrule import Rule0R
        from minesweepervariants.impl.impl_obj import MINES_TAG, VALUE_QUESS

        board = s.board.clone()
        answer_board = s.board.clone()
        board.clear_variable()
        model = board.get_model()
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
            rule.create_constraints(board, switch)

        for key in board.get_board_keys():
            for pos, obj in board(key=key):
                if obj is None:
                    continue
                obj: AbstractValue
                obj.create_constraints(board, switch)

        # 3.获取所有变量并赋值已解完的部分
        for key in board.get_board_keys():
            for _, var in board("C", mode="variable", key=key, special='raw'):
                model.add(var == 0)
                logger.trace(f"var: {var} == 0")
            for _, var in board("F", mode="variable", key=key, special='raw'):
                model.add(var == 1)
                logger.trace(f"var: {var} == 1")

        model: cp_model.CpModel
        # model.AddAssumptions(switch.var_map.values())
        for switch_var in switch.get_all_vars():
            model.add(switch_var == 1)
        solver = get_solver(False)
        status = solver.solve(model)
        if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            raise ValueError("input board is not feasible")

        for pos, var in board(mode="var"):
            if answer_board[pos] is not None:
                continue
            if solver.Value(var) == 1:
                answer_board[pos] = MINES_TAG
            else:
                answer_board[pos] = VALUE_QUESS

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
    for pos, obj in mask_board():
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

    if not no_image:
        threading.Thread(
            target=draw_board,
            kwargs={
                "board": game.board.clone(),
                "output": file_name + "_" + str(hint_times)
            }
        ).start()
    hint_times += 1
    while game.deduced():
        hint = game.hint()
        hint = {b: hint[b] for b in hint if len(b) == min(len(key) for key in hint)}
        key = len(list(hint.keys())[0])
        clue_freq[key] = clue_freq.setdefault(key, 0) + len(hint)
        for hint_because in hint:
            print(hint_because, "->", hint[hint_because])
        print(clue_freq)

        botten_text = ""

        if not no_image:
            for hint_because in hint:
                # draw_board(
                #     board=game.board,
                #     bottom_text=botten_text,
                #     output=file_name + "_" + str(hint_times),
                #     hint_because=hint_because,
                #     hint_deduced=hint[hint_because]
                # )
                threading.Thread(
                    target=draw_board,
                    kwargs={
                        "board": game.board.clone(),
                        "bottom_text": botten_text,
                        "output": file_name + "_" + str(hint_times),
                        "hint_because": hint_because,
                        "hint_deduced": hint[hint_because]
                    }
                ).start()
                hint_times += 1

        for pos in set(sum(hint.values(), [])):
            imposs = game.answer_board.get_type(pos, special='raw')
            game.apply(pos, 0 if imposs == "C" else 1)

        print(game.board)

    if not no_image:
        threading.Thread(
            target=draw_board,
            kwargs={
                "board": game.board.clone(),
                "output": file_name + "_" + str(hint_times)
            }
        ).start()
    hint_times += 1

    return
