#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/06/03 18:38
# @Author  : Wu_RH
# @FileName: Mrule.py

# 雷线索由于未实装 等待版本大更新

from typing import TYPE_CHECKING, Dict

from minesweepervariants.abs.board import AbstractBoard
from .rule import AbstractRule, AbstractValue
from ..utils.image_create import get_image, get_col, get_dummy, get_text
from abc import abstractmethod, ABC

from ..utils.web_template import Number

if TYPE_CHECKING:
    from minesweepervariants.abs.board import AbstractPosition, AbstractBoard


class AbstractMinesClueRule(AbstractRule, ABC):
    """
    雷线索规则
    """

    @abstractmethod
    def fill(self, board: 'AbstractBoard') -> 'AbstractBoard':
        """
        将在左线放置完成后调用
        需要将题板内的所有雷值赋值为线索
        :param board: 题板
        :return: 题板
        """


class AbstractMinesValue(AbstractValue, ABC):
    pos: 'AbstractPosition'

    @abstractmethod
    def __init__(self, pos: 'AbstractPosition', code: bytes = None):
        self.pos = pos

    def __repr__(self):
        """
        当前值在展示时候的显示字符串
        :return: 显示的字符串
        """
        return "F"

    def compose(self, board: 'AbstractBoard') -> Dict:
        """
        返回一个可渲染对象列表
        默认使用__repr__
        """
        return get_col(
            get_dummy(height=0.175),
            get_text(self.__repr__(),
                     color=("#FFFF00", "#FF7F00")),
            get_dummy(height=0.175),
        )

    def web_component(self, board: 'AbstractBoard') -> Dict:
        """
        返回一个可渲染对象列表
        默认使用__repr__
        """
        if "compose" in type(self).__dict__:
            return self.compose(board)
        data = Number(self.__repr__())
        return data
    
    def weaker(self, board: AbstractBoard) -> AbstractValue:
        return board.get_config(self.pos.board_key, "MINES")
    
    def weaker_times(self) -> int:
        return 1


# --------实例类-------- #


class MinesTag(AbstractMinesValue):
    """
    雷标志类
    用于暂存表示为类
    """

    def __init__(self, pos: 'AbstractPosition', code: bytes = None):
        pass

    def __repr__(self):
        return "雷"

    def compose(self, board) -> Dict:
        return get_image(
            "flag",
            cover_pos_label=False
        )

    @classmethod
    def type(cls) -> bytes:
        return b"F"

    def code(self) -> bytes:
        return b""

    def weaker(self, board: AbstractBoard) -> AbstractValue:
        return self

    def weaker_times(self) -> int:
        return 0


class Rule0F(AbstractMinesClueRule):
    name = "_0F"

    def __init__(self, board: "AbstractBoard" = None, data=None) -> None:
        super().__init__(board, data)
        self.drop = data is None

    def init_clear(self, board: 'AbstractBoard'):
        if not self.drop:
            return
        for key in board.get_board_keys():
            if not board.get_config(key, "interactive"):
                continue
            for pos, _ in board("F", key=key):
                board.set_value(pos, None)

    def fill(self, board: 'AbstractBoard') -> 'AbstractBoard':
        return board


class ValueCircle(AbstractMinesValue):
    def __init__(self, pos: 'AbstractPosition', code: bytes = None):
        pass

    def __repr__(self):
        return "O"

    def web_component(self, board: 'AbstractBoard') -> Dict:
        return get_image("circle", cover_pos_label=False)

    def compose(self, board) -> Dict:
        return get_image("circle", cover_pos_label=False)

    def code(self) -> bytes:
        return b""

    @classmethod
    def type(cls) -> bytes:
        return b"O"

    def weaker(self, board: AbstractBoard) -> AbstractValue:
        return self

    def weaker_times(self) -> int:
        return 0
