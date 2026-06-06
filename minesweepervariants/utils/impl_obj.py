#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2025/06/04 07:14
# @Author  : Wu_RH
# @FileName: impl_obj.py

from minesweepervariants.position import PositionTag
from ..utils import tool

from ..abs.Mrule import MinesTag, ValueCircle
from ..abs.Rrule import ValueQuess, ValueCross

POSITION_TAG = PositionTag()
MINES_TAG = MinesTag(POSITION_TAG)
VALUE_QUESS = ValueQuess(POSITION_TAG)
VALUE_CIRCLE = ValueCircle(POSITION_TAG)
VALUE_CROSS = ValueCross(POSITION_TAG)

TOTAL = -1

def serialize(value):
    from ..impl.summon.game import MinesAsterisk, ValueAsterisk
    if value == VALUE_QUESS:
        return "?"
    elif value == VALUE_CROSS:
        return "X"
    elif value == VALUE_CIRCLE:
        return "O"
    elif value == MINES_TAG:
        return "#"
    elif isinstance(value, ValueAsterisk):
        return "*"
    elif isinstance(value, MinesAsterisk):
        return "F"
    else:
        raise ValueError(f"Unknown value type: {type(value)}")

def decode(data: str):
    from ..impl.summon.game import MinesAsterisk, ValueAsterisk
    match data:
        case "?":
            return VALUE_QUESS
        case "X":
            return VALUE_CROSS
        case "O":
            return VALUE_CIRCLE
        case "#":
            return MINES_TAG
        case "*":
            return ValueAsterisk(POSITION_TAG)
        case "F":
            return MinesAsterisk(POSITION_TAG)
        case _:
            raise ValueError(f"Unknown data string: {data}")

def decode_singleton(clue_type: str):
    if clue_type == MINES_TAG.type().decode("ascii"):
        return MINES_TAG
    elif clue_type == VALUE_QUESS.type().decode("ascii"):
        return VALUE_QUESS
    elif clue_type == VALUE_CIRCLE.type().decode("ascii"):
        return VALUE_CIRCLE
    elif clue_type == VALUE_CROSS.type().decode("ascii"):
        return VALUE_CROSS
    else:
        return None

def set_total(total: int):
    global TOTAL
    TOTAL = total


def get_total() -> int:
    return TOTAL


def get_seed():
    return tool.SEED
