#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2025/06/03 18:38
# @Author  : Wu_RH
# @FileName: Rrule.py

from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, List, Dict

from minesweepervariants.abs.board import AbstractBoard

from .rule import AbstractRule, AbstractValue
from ..utils.image_create import get_text, get_image, get_dummy, get_col
from ..utils.web_template import Number

if TYPE_CHECKING:
    from minesweepervariants.abs.board import AbstractBoard, AbstractPosition


class AbstractClueRule(AbstractRule):
    """
    数字线索规则
    """

    # 动态删线索模式能力标记。默认关闭，保持历史规则行为不变。
    dynamic_dig_enabled = False
    # 动态删线索是否启用“求解器最优化显示变量”路径，默认关闭。
    dynamic_dig_use_visibility_optimizer = False

    @abstractmethod
    def fill(self, board: 'AbstractBoard') -> 'AbstractBoard':
        """
        填充所有None为规则线索对象
        :param board: 题板
        :return: 题板
        """
        ...

    def dynamic_init_visibility(self, board: 'AbstractBoard', visibility_state: Dict[str, Dict[tuple[int, int], bool | None]]):
        """
        动态删线索模式初始化回调。
        visibility_state 采用 {key: {(x, y): Optional[bool]}}，
        其中 True=显示, False=隐藏, None=该格不参与动态显隐。
        """
        return None

    def dynamic_on_visibility_changed(
        self,
        board: 'AbstractBoard',
        visibility_state: Dict[str, Dict[tuple[int, int], bool | None]],
        changed_positions: List['AbstractPosition'],
    ):
        """
        动态删线索模式回调: 显隐状态变更后重建线索值/约束前置状态。
        默认空实现，老规则无需修改即可运行。
        """
        return None


class AbstractClueValue(AbstractValue, ABC):
    """
    线索格数字对象类
    """

    def __repr__(self) -> str:
        """
        当前值在展示时候的显示字符串
        :return: 显示的字符串
        """
        return "?"

    def compose(self, board) -> Dict:
        """
        返回一个可渲染对象列表
        默认使用__repr__
        """
        return get_col(
            get_dummy(height=0.3),
            get_text(self.__repr__()),
            get_dummy(height=0.3),
        )

    def web_component(self, board) -> Dict:
        """
        返回一个可渲染对象列表
        默认使用__repr__
        """
        if "compose" in type(self).__dict__:
            return self.compose(board)
        return Number(self.__repr__())

    def weaker(self, board: AbstractBoard) -> AbstractValue:
        return board.get_config(self.pos.board_key, "VALUE")

    def weaker_times(self) -> int:
        return 1


# --------实例类-------- #


class ValueQuess(AbstractClueValue):
    """
    问号类(线索非雷)
    """

    def __init__(self, pos: 'AbstractPosition', code: bytes = b''):
        super().__init__(pos)

    def __repr__(self):
        return "?"

    @classmethod
    def type(cls) -> bytes:
        return b"?"

    def code(self) -> bytes:
        return b""

    def high_light(self, board: 'AbstractBoard') -> List['AbstractPosition'] | None:
        return []

    def weaker(self, board: AbstractBoard) -> AbstractValue:
        return self


class ValueCross(AbstractClueValue):
    """
    副板的叉号
    """

    def __init__(self, pos: 'AbstractPosition', code: bytes = b''):
        super().__init__(pos)

    def __repr__(self):
        return "X"

    def web_component(self, board) -> Dict:
        return get_image("cross")

    def compose(self, board) -> Dict:
        return get_image("cross")

    @classmethod
    def type(cls) -> bytes:
        return b"X"

    def code(self) -> bytes:
        return b""

    def weaker(self, board: AbstractBoard) -> AbstractValue:
        return self
