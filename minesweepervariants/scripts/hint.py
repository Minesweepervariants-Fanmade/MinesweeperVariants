#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2026/05/24 14:03
# @Author  : Wu_RH
# @FileName: hint.py

import threading
from typing import Set, Any, Dict, Tuple, List

from minesweepervariants.abs.board import AbstractPosition, MASTER_BOARD, decompress, json_loads, Size
from minesweepervariants.config.config import DEFAULT_CONFIG
from minesweepervariants.impl.impl_obj import decode_board
from minesweepervariants.impl.summon import Summon
from minesweepervariants.impl.summon.game import GameSession, Mode, UMode
from minesweepervariants.utils import tool
from minesweepervariants.utils.image_create import draw_board
from minesweepervariants.utils.tool import get_logger

defaults = {}
defaults.update(DEFAULT_CONFIG)


def max_disjoint_lists(data: Dict[Tuple, List[AbstractPosition]]) -> List[List[AbstractPosition]]:
    """
    从字典的列表中选出互不相交的子集，使并集元素数最大。
    返回选中的列表（保持原始列表顺序）。
    """
    items = list(data.items())  # [(key, list), ...]
    n = len(items)
    # 将列表转换为集合
    sets = [set(lst) for _, lst in items]
    # 元素到索引列表的映射，用于快速检查冲突
    elem_to_indices = {}
    for idx, s in enumerate(sets):
        for elem in s:
            elem_to_indices.setdefault(elem, []).append(idx)

    # 按集合大小降序排序，优先尝试大集合
    indices = list(range(n))
    indices.sort(key=lambda i: -len(sets[i]))
    # 重新排序 sets 和 items
    sorted_sets = [sets[i] for i in indices]
    sorted_items = [items[i] for i in indices]

    best_selection = []
    best_size = 0

    def dfs(start_idx: int, used: Set[Any], selected: List[int], cur_size: int):
        nonlocal best_selection, best_size
        # 剪枝：剩余集合即使全部不冲突且全选，最大可能大小
        if cur_size + max_possible(start_idx, used) <= best_size:
            return

        if cur_size > best_size:
            best_size = cur_size
            # 保存选中的原始列表（按原顺序？这里先按排序后的顺序，最后再恢复？）
            # 为了方便，保存 selected 列表的索引（排序后的）
            best_selection = selected[:]

        # 尝试从 start_idx 开始选下一个集合
        for i in range(start_idx, n):
            s = sorted_sets[i]
            # 如果有任何元素已被使用，则跳过
            if used.intersection(s):
                continue
            # 选择该集合
            used.update(s)
            selected.append(i)
            dfs(i + 1, used, selected, cur_size + len(s))
            # 回溯
            selected.pop()
            used.difference_update(s)

    def max_possible(start_idx: int, used: Set[Any]) -> int:
        """乐观估计：剩余未使用且不与已选冲突的集合大小和（简单贪心）"""
        # 简单上界：剩余所有集合的大小的和（忽略冲突，过于乐观但安全）
        # 更紧的上界：可以计算剩余元素总数，但需要谨慎
        total = 0
        for i in range(start_idx, n):
            s = sorted_sets[i]
            if not used.intersection(s):
                total += len(s)
        return total

    # 执行搜索
    dfs(0, set(), [], 0)

    # 将选中的索引（排序后的）转换回原始列表顺序
    result_lists = [sorted_items[i][1] for i in best_selection]  # 注意：sorted_items[i][1] 是原始列表
    return result_lists


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
        size=Size(0, 0), total=-2, rules=rules[:],
        board=board_class, drop_r=drop_r, board_data=json_loads(decompress(board_code))
    )

    mask_board = s.board

    if answer:
        answer_board = decode_board(data=json_loads(decompress(answer)))
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

        for key in board.get_board_keys():
            for pos, var in board(mode="var", key=key):
                if answer_board[pos] is not None:
                    continue
                if solver.Value(var) == 1:
                    answer_board[pos] = board.get_config(key, "MINES")
                else:
                    answer_board[pos] = board.get_config(key, "VALUE")

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

    thread_list = []

    if not no_image:
        thread = threading.Thread(
            target=draw_board,
            kwargs={
                "board": game.board.clone(),
                "output": file_name + "_" + str(hint_times)
            }
        )
        thread.start()
        thread_list.append(thread)
    hint_times += 1
    while game.deduced():
        hint = game.hint()
        hint = {b: hint[b] for b in hint if len(b) == min(len(key) for key in hint)}
        if not hint:
            logger.error("hint返回空 deduced仍然存在可推格 待检查规则/副板未框定")
            break
        key = len(list(hint.keys())[0])
        clue_freq[key] = clue_freq.setdefault(key, 0) + len(hint)
        for hint_because in hint:
            logger.info(f"{hint_because}->{hint[hint_because]}")
        logger.info(f"clue freq now: {clue_freq}")

        if not no_image:
            for hint_because in hint:
                bottom_text = '; '.join([b[0] + (f":{b[1]}" if b[1] else "") for b in hint_because if isinstance(b, tuple)])
                thread = threading.Thread(
                    target=draw_board,
                    kwargs={
                        "board": game.board.clone(),
                        "bottom_text": bottom_text,
                        "output": file_name + "_" + str(hint_times),
                        "hint_because": [pos for pos in hint_because if isinstance(pos, AbstractPosition)],
                        "hint_deduced": hint[hint_because]
                    }
                )
                thread.start()
                thread_list.append(thread)
                hint_times += 1

                for pos in hint[hint_because]:
                    imposs = game.answer_board.get_type(pos, special='raw')
                    game.apply(pos, 0 if imposs == "C" else 1)
                break
        logger.info(f"当前题板:\n{game.board}")

    if not no_image:
        thread = threading.Thread(
            target=draw_board,
            kwargs={
                "board": game.board.clone(),
                "output": file_name + "_" + str(hint_times)
            }
        )
        thread.start()
        thread_list.append(thread)
    hint_times += 1
    for thread in thread_list:
        thread.join()

    logger.info(f"最终题板:\n{game.board}")
    logger.info(f"线索图:{clue_freq}")

    return
