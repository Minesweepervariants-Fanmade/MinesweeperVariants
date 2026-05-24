#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/06/03 18:38
# @Author  : Wu_RH
# @FileName: Mrule.py

# 雷线索由于未实装 等待版本大更新

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Mapping

from minesweepervariants.abs.board import AbstractBoard
from .rule import AbstractRule, AbstractValue
from ..utils.web_template import Number

if TYPE_CHECKING:
    from minesweepervariants.abs.board import AbstractPosition


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def _get_dummy(width: float = 0.01, height: float = 0.01) -> dict[str, object]:
    return {
        "type": "placeholder",
        "width": width,
        "height": height,
        "cover": True,
        "dominant": None,
    }


def _get_text(
    text: str,
    width: float | str = "auto",
    height: float | str = "auto",
    cover_pos_label: bool = True,
    color: tuple[str, str] = ("#FFFFFF", "#000000"),
    dominant_by_height: bool = True,
    style: str = "",
) -> dict[str, object]:
    dominant: str | None = "height" if dominant_by_height else "width"
    return {
        "type": "text",
        "text": text,
        "content": text,
        "color_black": _hex_to_rgb(color[0]),
        "color_white": _hex_to_rgb(color[1]),
        "width": width,
        "height": height,
        "font_size": 1,
        "cover": cover_pos_label,
        "dominant": dominant,
        "style": style,
    }


def _get_image(
    image_path: str,
    image_width: float | str = "auto",
    image_height: float | str = "auto",
    cover_pos_label: bool = True,
    dominant_by_height: bool = True,
    style: str = "",
) -> dict[str, object]:
    dominant: str | None = "height" if dominant_by_height else "width"
    return {
        "type": "image",
        "image": image_path,
        "height": image_height,
        "width": image_width,
        "cover": cover_pos_label,
        "dominant": dominant,
        "style": style,
    }


def _get_col(*args: dict[str, object], spacing: int = 0, dominant_by_height: bool = False) -> dict[str, object]:
    dominant: str | None = "height" if dominant_by_height else "width"
    for child in args:
        if child.get("dominant") is None:
            child["dominant"] = "width"
    width_values = [item["width"] for item in args if isinstance(item["width"], int)]
    width = max(width_values) if width_values else "auto"
    return {
        "type": "col",
        "children": args,
        "spacing": spacing,
        "cover": all(item["cover"] for item in args),
        "height": "auto",
        "width": width,
        "dominant": dominant,
    }


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
    def __init__(self, pos: 'AbstractPosition', code: bytes = b'') -> None:
        self.pos = pos

    def __repr__(self) -> str:
        """
        当前值在展示时候的显示字符串
        :return: 显示的字符串
        """
        return "F"

    def compose(self, board: 'AbstractBoard') -> Mapping[str, object]:
        """
        返回一个可渲染对象列表
        默认使用__repr__
        """
        return _get_col(
            _get_dummy(height=0.175),
            _get_text(self.__repr__(), color=("#FFFF00", "#FF7F00")),
            _get_dummy(height=0.175),
        )

    def web_component(self, board: 'AbstractBoard') -> Mapping[str, object]:
        """
        返回一个可渲染对象列表
        默认使用__repr__
        """
        if "compose" in type(self).__dict__:
            return self.compose(board)
        return Number(self.__repr__())

    def weaker(self, board: AbstractBoard) -> AbstractValue:
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

    def __init__(self, pos: 'AbstractPosition', code: bytes = b'') -> None:
        super().__init__(pos, code)

    def __repr__(self) -> str:
        return "雷"

    def compose(self, board: 'AbstractBoard') -> Mapping[str, object]:
        return _get_image(
            "flag",
            cover_pos_label=False,
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
    id = "_0F"
    name = "_0F"
    doc = ""
    author = ("", 0)
    tags = ["Untagged"]
    creation_time = ""

    def __init__(self, board: "AbstractBoard | None" = None, data: str | None = None) -> None:
        super().__init__(board, data)
        self.drop = data is None

    def init_clear(self, board: 'AbstractBoard') -> None:
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
    def __init__(self, pos: 'AbstractPosition', code: bytes = b'') -> None:
        super().__init__(pos, code)

    def __repr__(self) -> str:
        return "O"

    def web_component(self, board: 'AbstractBoard') -> Mapping[str, object]:
        return _get_image("circle", cover_pos_label=False)

    def compose(self, board: 'AbstractBoard') -> Mapping[str, object]:
        return _get_image("circle", cover_pos_label=False)

    def code(self) -> bytes:
        return b""

    @classmethod
    def type(cls) -> bytes:
        return b"O"

    def weaker(self, board: AbstractBoard) -> AbstractValue:
        return self

    def weaker_times(self) -> int:
        return 0
