#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Interactive game CLI for MinesweeperVariants."""

from __future__ import annotations

import os
from typing import Optional

from ortools.sat.python import cp_model

from minesweepervariants.board import Board, Position
from minesweepervariants.abs.Lrule import Rule0R
from minesweepervariants.config.config import DEFAULT_CONFIG
from minesweepervariants.impl.impl_obj import decode_board
from minesweepervariants.impl.summon import Summon
from minesweepervariants.impl.summon.game import GameSession, Mode, UMode, max_disjoint_lists
from minesweepervariants.impl.summon.solver import Switch, get_solver
from minesweepervariants.utils.image_create import draw_board
from minesweepervariants.utils.tool import Logger, get_logger


def _save_image(
    board: Board,
    out_prefix: str,
    turn: int | None = None,
    hint_because: list[Position] | None = None,
    hint_deduced: list[Position] | None = None,
    logger: Logger | None = None,
) -> str:
    name = out_prefix if turn is None else f"{out_prefix}"
    draw_board(board, output=name, hint_because=hint_because or [], hint_deduced=hint_deduced or [])
    filepath = os.path.join(str(DEFAULT_CONFIG["output_path"]), f"{name}.png")
    (logger or get_logger("game")).info(f"Image saved to: {filepath}")
    return filepath


class _CoordParser:
    _instance: "_CoordParser | None" = None
    last_col: Optional[int]
    last_row: Optional[int]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.last_col = None
            cls._instance.last_row = None
        return cls._instance

    def parse(self, text: str, board: Board) -> Optional[Position]:
        text = text.strip()
        board_keys = board.get_board_keys()
        if not board_keys:
            return None
        key = board_keys[0]

        letters = "".join(ch for ch in text if ch.isalpha())
        digits = "".join(ch for ch in text if ch.isdigit())


        if letters:
            try:
                col = 0
                for ch in letters.upper():
                    col = col * 26 + (ord(ch) - ord("A") + 1)
                col -= 1
                self.last_col = col
            except Exception:
                return None

        if digits:
            try:
                row = int(digits) - 1
                self.last_row = row
            except Exception:
                return None

        if self.last_col is None or self.last_row is None:
            return None
        return board.get_pos(row=self.last_row, col=self.last_col, key=key)


_parse_coord = _CoordParser()


def _resolve_state(token: str) -> Optional[str]:
    token = token.lower()
    if token in ("c", "click"):
        return "c"
    if token in ("r", "right", "m", "mark", "f", "flag"):
        return "r"
    return None


def _click_pos(game: GameSession, pos: Position) -> tuple[bool, str]:
    if game.click(pos) is None:
        return False, "你踩雷了!"
    return True, ""


def _flag_pos(game: GameSession, pos: Position) -> tuple[bool, str]:
    if game.mark(pos) is None:
        return False, "你标记了一个错误的雷!"
    return True, ""


def _hint_once(game: GameSession) -> tuple[list[str], list[Position], list[Position], bool]:
    hints = game.hint()
    if not hints:
        return ["暂无提示"], [], [], False

    minsize = min(len(k) for k in hints)
    hints = {
        because: deduceds
        for because, deduceds in hints.items()
        if len(because) == minsize
    }
    apply_hint = max_disjoint_lists(hints)

    grouped_hints: dict[tuple[object, ...], list[Position]] = {}
    for apply in apply_hint:
        for because, deduceds in hints.items():
            if deduceds == apply:
                grouped_hints[because] = apply
                break

    try:
        first_key = next(iter(grouped_hints))
    except StopIteration:
        return ["已无可推格"], [], [], False
    hint_because = [p for p in first_key if isinstance(p, Position)] if isinstance(first_key, tuple) else []
    hint_deduced = grouped_hints[first_key]

    lines = ["Hints:"]
    lines.extend(f"- because: {k} -> deduced: {v}" for k, v in grouped_hints.items())
    return lines, hint_because, hint_deduced, True


def _process_token(
    token: str,
    state: str,
    game: GameSession,
) -> tuple[str, bool, list[str], list[Position], list[Position], bool, bool]:
    """Process one token.

    Returns: (new_state, consumed, messages, hint_because, hint_deduced, should_render, should_exit)
    """
    token = token.strip()
    if not token:
        return state, False, [], [], [], False, False

    lower = token.lower()

    new_state = _resolve_state(token)
    if new_state is not None:
        return new_state, False, [], [], [], False, False

    if lower in ("q", "quit", "exit"):
        return state, True, [], [], [], False, True

    if lower in ("h", "hint"):
        messages, hint_because, hint_deduced, should_render = _hint_once(game)
        return state, True, messages, hint_because, hint_deduced, should_render, False

    if lower in ("debug", "dbg"):
        try:
            # 触发调试断点，便于在调用处进入调试器
            breakpoint()
        except Exception:
            # 在不支持 breakpoint 的环境中忽略
            pass
        return state, True, ["debugger triggered"], [], [], False, False

    if any(ch.isalpha() and ch.islower() for ch in token):
        return state, True, ["未知命令"], [], [], False, False

    pos = _parse_coord.parse(token, game.board)
    if pos is None:
        return state, True, ["无法解析坐标"], [], [], False, False

    if state == "r":
        ok, message = _flag_pos(game, pos)
    else:
        ok, message = _click_pos(game, pos)

    if not ok:
        return state, True, [message], [], [], False, False

    return state, True, [], [], [], True, False


def _mode_text(game: GameSession, state: str) -> str:
    mode_text = f"mode={game.mode.name}"
    state_text = f" state={state}"
    ultimate_text = f" ultimate_mode={game.ultimate_mode}" if game.mode == Mode.ULTIMATE else ""
    drop_text = f" drop_r={game.drop_r}"
    deduced_cnt = len(game.deduced() or [])
    return f"[{mode_text}{state_text}{ultimate_text}{drop_text} deduced={deduced_cnt}]"


def _handle_input_line(
    line: str,
    state: str,
    game: GameSession,
) -> tuple[str, list[str], list[Position], list[Position], bool, bool]:
    messages: list[str] = []
    hint_because: list[Position] = []
    hint_deduced: list[Position] = []
    should_render = False
    should_exit = False

    for token in line.split():
        state, _consumed, token_messages, token_hint_because, token_hint_deduced, token_render, token_exit = _process_token(
            token,
            state,
            game,
        )
        messages.extend(token_messages)
        if token_hint_because or token_hint_deduced:
            hint_because = token_hint_because
            hint_deduced = token_hint_deduced
        should_render = should_render or token_render
        should_exit = should_exit or token_exit
        if should_exit:
            break

    return state, messages, hint_because, hint_deduced, should_render, should_exit


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
    logger = get_logger("game", log_lv)

    if not board_code:
        raise ValueError("未输入题板")

    match game_mode:
        case "expert" | "EXPERT" | "专家":
            game_session_mode = Mode.EXPERT
        case "ultimate" | "ULTIMATE" | "终极模式":
            game_session_mode = Mode.ULTIMATE
        case "puzzle" | "PUZZLE" | "纸笔":
            game_session_mode = Mode.PUZZLE
        case _:
            raise ValueError(f"unknown game mode: {game_mode}")

    summon = Summon(
        size=Size(0, 0),
        total=total,
        rules=rules[:],
        board=board_class,
        drop_r=drop_r,
        board_data=json_loads(decompress(board_code)),
    )

    answer_board = decode_board(data=json_loads(decompress(answer))) if answer else None

    if answer_board is None or answer_board.has("N"):
        answer_board = answer_board if answer_board else summon.board.clone()
        answer_board.clear_variable()
        model = answer_board.get_model()
        switch = Switch()

        for rule in summon.mines_rules.rules + [summon.clue_rule, summon.mines_clue_rule]:
            if rule is None:
                continue
            if drop_r and isinstance(rule, Rule0R):
                continue
            rule.create_constraints(answer_board, switch)

        for key in answer_board.get_board_keys():
            for pos, obj in answer_board(key=key):
                if obj is None:
                    continue
                obj.create_constraints(answer_board, switch)

        for key in answer_board.get_board_keys():
            for _, var in answer_board("C", mode="variable", key=key, special='raw'):
                model.add(var == 0)
            for _, var in answer_board("F", mode="variable", key=key, special='raw'):
                model.add(var == 1)

        for switch_var in switch.get_all_vars():
            model.add(switch_var == 1)

        solver = get_solver(False)
        status = solver.solve(model)
        if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            raise ValueError("input board is not feasible")

        for key in answer_board.get_board_keys():
                for pos, var in answer_board(mode="var", key=key):
                    if answer_board[pos] is not None:
                        continue
                    if solver.Value(var) == 1:
                        answer_board[pos] = answer_board.get_config(key, "MINES")  # type: ignore[arg-type]
                    else:
                        answer_board[pos] = answer_board.get_config(key, "VALUE")  # type: ignore[arg-type]

    summon.answer_board = answer_board

    game = GameSession(summon, mode=game_session_mode, drop_r=drop_r)
    game.mode = game_session_mode
    if game_session_mode == Mode.ULTIMATE:
        game.ultimate_mode = (
            UMode.ULTIMATE_A
            | UMode.ULTIMATE_F
            | UMode.ULTIMATE_S
            | UMode.ULTIMATE_R
            | UMode.ULTIMATE_P
        )
    game.board = summon.board
    game.answer_board = summon.answer_board

    turn = 0
    out = file_name if file_name else "game"
    state = "c"

    if not no_image:
        _save_image(game.board.clone(), out, logger=logger)

    while True:
        try:
            line = input(f"{_mode_text(game, state)} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出游戏")
            break

        if not line:
            continue

        state, messages, hint_because, hint_deduced, should_render, should_exit = _handle_input_line(line, state, game)

        for message in messages:
            print(message)

        if should_render and not no_image:
            turn += 1
            _save_image(
                game.board.clone(),
                out,
                turn=turn,
                hint_because=hint_because,
                hint_deduced=hint_deduced,
                logger=logger,
            )

        if should_exit:
            break

    return


if __name__ == "__main__":
    print("请通过主程序运行 game 子命令")