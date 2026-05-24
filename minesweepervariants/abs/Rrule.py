#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2025/06/03 18:38
# @Author  : Wu_RH
# @FileName: Rrule.py

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Mapping

from json.decoder import JSONObject

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

    def dynamic_init_visibility(self, board: 'AbstractBoard', visibility_state: Dict[str, Dict[tuple[int, int], bool | None]]) -> None:
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
    ) -> None:
        """
        动态删线索模式回调: 显隐状态变更后重建线索值/约束前置状态。
        默认空实现，老规则无需修改即可运行。
        """
        return None


class AbstractClueValue(AbstractValue, ABC):
    """
    线索格数字对象类
    """

    def from_json(self, data: JSONObject) -> None:
        if data['type'] =="old_style":
            self.__init__(self.pos, code=data.get('code', b''))
        else:
            raise ValueError(f"Unsupported clue value type: {data['type']}")


    def __repr__(self) -> str:
        """
        当前值在展示时候的显示字符串
        :return: 显示的字符串
        """
        return "?"

    def compose(self, board: 'AbstractBoard') -> Mapping[str, object]:
        """
        返回一个可渲染对象列表
        默认使用__repr__
        """
        return _get_col(
            _get_dummy(height=0.3),
            _get_text(self.__repr__()),
            _get_dummy(height=0.3),
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
        value = board.get_config(self.pos.board_key, "VALUE")
        if isinstance(value, AbstractValue):
            return value
        return self

    def weaker_times(self) -> int:
        return 1

    def json(self) -> JSONObject:
        from base64 import b64encode
        return {"type": "old_style", "code": b64encode(self.code()).decode()}


# --------实例类-------- #


class ValueQuess(AbstractClueValue):
    """
    问号类(线索非雷)
    """

    def __init__(self, pos: 'AbstractPosition', code: bytes = b'') -> None:
        super().__init__(pos)

    def __repr__(self) -> str:
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

    def weaker_times(self) -> int:
        return 0


class ValueCross(AbstractClueValue):
    """
    副板的叉号
    """

    def __init__(self, pos: 'AbstractPosition', code: bytes = b'') -> None:
        super().__init__(pos)

    def __repr__(self) -> str:
        return "X"

    def web_component(self, board: 'AbstractBoard') -> Mapping[str, object]:
        return _get_image("cross")

    def compose(self, board: 'AbstractBoard') -> Mapping[str, object]:
        return _get_image("cross")

    @classmethod
    def type(cls) -> bytes:
        return b"X"

    def code(self) -> bytes:
        return b""

    def weaker(self, board: AbstractBoard) -> AbstractValue:
        return self

    def weaker_times(self) -> int:
        return 0
