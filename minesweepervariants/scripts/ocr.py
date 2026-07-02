#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2026/06/30 16:41
# @Author  : Wu_RH
# @FileName: ocr.py
from typing import Optional

from minesweepervariants.abs.rule import AbstractValue
from minesweepervariants.board import Board
from minesweepervariants.config.config import DEFAULT_CONFIG
from minesweepervariants.impl.impl_obj import get_value
from minesweepervariants.impl.summon import Summon
from minesweepervariants.impl.summon.game import VALUE_TAG
from minesweepervariants.json_object import deep_wrap, json_dumps, compress
from minesweepervariants.position import Position
from minesweepervariants.size import Size
from minesweepervariants.utils import tool
from minesweepervariants.utils.impl_obj import VALUE_QUESS, MINES_TAG
from minesweepervariants.utils.ocr.ocr import ocr_board
from minesweepervariants.utils.tool import get_logger


def mapning_cell(
    texts: list[str],
    imgs: list[str],
    is_mines: bool,
) -> Optional[AbstractValue]:
    if "?" in texts:
        return VALUE_QUESS
    if "*" in texts:
        return VALUE_TAG
    if "Flag" in imgs:
        return MINES_TAG
    return None


def get_clue_value_code(
    pos: Position,
    clue_type: str,
    texts: list[str],
):
    for i in range(256):
        try:
            obj = get_value(pos, clue_type, deep_wrap({"code": [i]}))
        except:
            continue
        if str(obj) == texts[0]:
            return obj
    return None


def get_clue_value(
    pos: Position,
    clue_board: Board,
    texts: list[str],
    imgs: list[str],
    is_mines: bool,
):
    tmp_obj = clue_board[pos]
    if tmp_obj is None:
        return None
    tmp_obj_json = tmp_obj.json()
    if "code" in tmp_obj_json:
        return get_clue_value_code(
            pos, tmp_obj.id, texts
        )
    tmp_obj_data = tmp_obj_json.get("data", None)
    obj_data = {key: tmp_obj_json[key] for key in tmp_obj_json}
    if type(tmp_obj_data) is tuple:
        if type(texts[0]) is int:
            if not all(t.isdigit() for t in texts):
                return None
            obj_data["data"] = [int(t) for t in texts]
        else:
            obj_data["data"] = texts
    if type(tmp_obj_data) is int:
        if not texts[0].isdigit():
            return None
        obj_data["data"] = int(texts[0])
    else:
        obj_data["data"] = texts[0]
    clue_type = tmp_obj.id
    obj = get_value(pos, clue_type, deep_wrap(obj_data))
    return obj


def main(
    img_path: str,
    rules_id: list[str],
    log_lv: str,
    file_name: str,
):
    DEFAULT_CONFIG["log_file_name"] = file_name
    tool.LOGGER = None
    logger = get_logger(log_lv=log_lv)
    pos_cell = ocr_board(img_path)
    cell_data = pos_cell.get("cell_data", {})
    if not cell_data:
        raise ValueError("未找到有效网格")
    summon = Summon(
        size=pos_cell.get("size_data", Size(0, 0)),
        total=-2, rules=rules_id
    )
    mines_clue = summon.mines_clue_rule
    value_clue = summon.clue_rule
    board = summon.board
    board_tmp_mines = board.clone()
    board_tmp_value = board.clone()
    for pos, _ in board_tmp_mines():
        board_tmp_mines[pos] = MINES_TAG
    board_tmp_mines = mines_clue.fill(board_tmp_mines)
    board_tmp_value = value_clue.fill(board_tmp_value)

    for pos_key, pos_data in cell_data.items():
        pos = board.get_pos(pos_key[0], pos_key[1])
        map_value = mapning_cell(**pos_data)
        if map_value:
            board[pos] = map_value
            continue
        is_mines = pos_data.get("is_mines")
        clue_board = board_tmp_mines if is_mines else board_tmp_value
        try:
            board[pos] = get_clue_value(pos, clue_board, **pos_data)
        except:
            pass

    logger.info("\n" + str(board))
    logger.info(f"|[BOARD]: {compress(json_dumps(board.json()))}|")
