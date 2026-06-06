#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/06/03 18:38
# @Author  : Wu_RH
# @FileName: Mrule.py

# 雷线索由于未实装 等待版本大更新

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Mapping

from minesweepervariants.board import Board
from minesweepervariants.utils.image_template import get_col, get_image, get_text, get_dummy
from .rule import AbstractRule, AbstractValue
from ..utils.web_template import Number

if TYPE_CHECKING:
    from minesweepervariants.position import Position


class AbstractMinesClueRule(AbstractRule, ABC):
    """
    雷线索规则
    """

    @abstractmethod
    def fill(self, board: 'Board') -> 'Board':
        """
        将在左线放置完成后调用
        需要将题板内的所有雷值赋值为线索
        :param board: 题板
        :return: 题板
        """


class AbstractMinesValue(AbstractValue, ABC):
    pos: 'Position'

    @abstractmethod
    def __init__(self, pos: 'Position', *args, **kwargs) -> None:
        self.pos = pos

    def __repr__(self) -> str:
        """
        当前值在展示时候的显示字符串
        :return: 显示的字符串
        """
        return "F"

    def compose(self, board: 'Board') -> Mapping[str, object]:
        """
        返回一个可渲染对象列表
        默认使用__repr__
        """
        return get_col(
            get_dummy(height=0.175),
            get_text(self.__repr__(), color=("#FFFF00", "#FF7F00")),
            get_dummy(height=0.175),
        )

    def web_component(self, board: 'Board') -> Mapping[str, object]:
        """
        返回一个可渲染对象列表
        默认使用__repr__
        """
        if "compose" in type(self).__dict__:
            return self.compose(board)
        return Number(self.__repr__())

    def weaker(self, board: Board) -> AbstractValue:
        value = board.get_config(self.pos.board_key, "MINES")
        if isinstance(value, AbstractValue):
            return value
        return self

    def weaker_times(self) -> int:
        return 1


# --------实例类-------- #


class MinesTag(AbstractMinesValue):
    """
    雷标志类
    用于暂存表示为类
    """

    def __init__(self, pos: 'Position', code: bytes = b'') -> None:
        super().__init__(pos, code)

    def __repr__(self) -> str:
        return "雷"

    def compose(self, board: 'Board') -> Mapping[str, object]:
        return get_image(
            "flag",
            cover_pos_label=False,
        )

    @classmethod
    def type(cls) -> bytes:
        return b"F"

    def code(self) -> bytes:
        return b""

    def weaker(self, board: Board) -> AbstractValue:
        return self

    def weaker_times(self) -> int:
        return 0


class Rule0F(AbstractMinesClueRule):
    id = "_0F"
    name = "_0F"
    doc = ""
    author = ("", 0)
    tags = ["Untagged"]
    creation_time = ""

    def __init__(self, board: "Board | None" = None, data: str | None = None) -> None:
        super().__init__(board, data)
        self.drop = data is None

    def init_clear(self, board: 'Board') -> None:
        if not self.drop:
            return
        for key in board.get_board_keys():
            if not board.get_config(key, "interactive"):
                continue
            for pos, _ in board("F", key=key):
                board.set_value(pos, None)

    def fill(self, board: 'Board') -> 'Board':
        return board


class ValueCircle(AbstractMinesValue):
    def __init__(self, pos: 'Position', code: bytes = b'') -> None:
        super().__init__(pos, code)

    def __repr__(self) -> str:
        return "O"

    def web_component(self, board: 'Board') -> Mapping[str, object]:
        return get_image("circle", cover_pos_label=False)

    def compose(self, board: 'Board') -> Mapping[str, object]:
        return get_image("circle", cover_pos_label=False)

    def code(self) -> bytes:
        return b""

    @classmethod
    def type(cls) -> bytes:
        return b"O"

    def weaker(self, board: Board) -> AbstractValue:
        return self

    def weaker_times(self) -> int:
        return 0
