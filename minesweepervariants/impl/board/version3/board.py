#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time    : 2025/06/03 01:58
# @Author  : Wu_RH
# @FileName: board.py

from base64 import b64decode
from minesweepervariants.abs.board import ImmutableDict, JSONObject
from typing import List, Literal, Optional, TypedDict, Union, Tuple, Any, Generator, overload, override
import heapq

import gc
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import IntVar




from ....abs.rule import AbstractValue
from ....utils.impl_obj import VALUE_QUESS, MINES_TAG
from ....utils.impl_obj import POSITION_TAG, VALUE_CROSS, VALUE_CIRCLE
from ....utils.tool import get_logger, get_random
from ....abs.board import AbstractBoard, AbstractPosition, MASTER_BOARD, JSONObject, Size, ImmutableDict, get_with_valid, jsonify, valid
from ....abs.Rrule import AbstractClueValue
from ....abs.Mrule import AbstractMinesValue


def alpha(col: int) -> str:
    alpha_map = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if col < 26:
        return alpha_map[col]
    return alpha_map[col // 26 - 1] + alpha_map[col % 26]


def encode_int_7bit(n: int) -> bytes:
    # 编码主体：每7位 -> 1字节（bit6~bit0，bit7=0）
    if n == 0:
        return b'\x00'
    payload = []

    while n > 0:
        payload.append(n & 0x7f)
        n >>= 7

    return bytes(payload)


def decode_bytes_7bit(data: bytes) -> int:
    if len(data) == 0:
        return 0

    result = 0
    for i in data[::-1]:
        result <<= 7
        result |= i

    return result


class Position(AbstractPosition):
    def __repr__(self):
        return (f"{self.board_key+':' if self.board_key != MASTER_BOARD else ''}"
                f"{alpha(self.col)}{self.row+1}")

    def _up(self, n: int = 1):
        self.row -= n

    def _down(self, n: int = 1):
        self.row += n

    def _left(self, n: int = 1):
        self.col -= n

    def _right(self, n: int = 1):
        self.col += n

    def _deviation(self, pos: 'AbstractPosition'):
        self.row += pos.row
        self.col += pos.col

    # def _north(self, n: int = 1):
    #     if self.row % 2 == 1:
    #         self.col -= n
    #         self.row -= n
    #     else:
    #         self.row -= n

    # def _east(self, n: int = 1):
    #     if self.row % 2 == 1:
    #         self.col -= n
    #         self.row += n
    #     else:
    #         self.row += n

    # def _west(self, n: int = 1):
    #     if self.row % 2 == 1:
    #         self.row -= n
    #     else:
    #         self.col += n
    #         self.row -= n

    # def _south(self, n: int = 1):
    #     if self.row % 2 == 1:
    #         self.row += n
    #     else:
    #         self.col += n
    #         self.row += n

    def in_bounds(self, bound_pos: 'AbstractPosition') -> bool:
        if bound_pos.board_key != self.board_key:
            return False
        return (0 <= self.col <= bound_pos.col and
                0 <= self.row <= bound_pos.row)

    def neighbors(self, *args: int):
        """
        按照欧几里得距离从小到大逐层扩散，筛选范围由距离平方控制（不包含当前位置）。

        调用方式（类似 range）：
            neighbors(end_layer)
                返回所有欧几里得距离 ≤ √end_layer 的位置（从第 1 层开始）。
            neighbors(start_layer, end_layer)
                返回所有欧几里得距离 ∈ [√start_layer, √end_layer] 的位置。

        :param args: 一个或两个整数
            - 若提供一个参数 end_layer，视为从 √1 到 √end_layer。
            - 若提供两个参数 start_layer 和 end_layer，视为从 √start_layer 到 √end_layer。
            - 参数非法（数量不为 1 或 2，或值非法）时返回空列表。

        :return: 位置列表，按距离从近到远排序。
        """

        # 解析参数
        if len(args) == 1:
            low, high = 1, args[0]
        elif len(args) == 2:
            low, high = args
        else:
            return []

        # 处理无效参数
        if high < low:
            return []

        x0, y0 = self.col, self.row
        directions = [(dx, dy) for dx in (-1, 0, 1)
                      for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]

        heap = []  # 最小堆存储 (距离平方, x, y)
        visited = {(x0, y0)}
        result = []

        # 处理包含自身的情况 (距离平方=0)
        if low <= 0 <= high:
            result.append(self.clone())

        # 初始化邻居
        for dx, dy in directions:
            nx, ny = x0 + dx, y0 + dy
            d_sq = (nx - x0) ** 2 + (ny - y0) ** 2
            if d_sq <= high:
                heapq.heappush(heap, (d_sq, nx, ny))
                visited.add((nx, ny))

        # 遍历所有可达位置
        while heap:
            d_sq, x, y = heapq.heappop(heap)

            # 检查是否在目标范围内
            if low <= d_sq <= high:
                result.append(Position(x, y, self.board_key))

            # 扩展新位置
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (nx, ny) in visited:
                    continue

                visited.add((nx, ny))
                new_d_sq = (nx - x0) ** 2 + (ny - y0) ** 2

                # 仅考虑距离平方未超过上限的位置
                if new_d_sq <= high:
                    heapq.heappush(heap, (new_d_sq, nx, ny))

        return result

    # def hex_neighbors(self, *args: int):
    #     """
    #     按照六边形网格距离从小到大逐层扩散，筛选范围由层数控制（不包含当前位置）。

    #     调用方式（类似 range）：
    #         hex_neighbors(end_layer)
    #             返回距离 ≤ end_layer 的位置（从第 1 层开始）。
    #         hex_neighbors(start_layer, end_layer)
    #             返回距离 ∈ [start_layer, end_layer] 的位置。

    #     :param args: 一个或两个整数
    #         - 若提供一个参数 end_layer，视为从 1 到 end_layer。
    #         - 若提供两个参数 start_layer 和 end_layer，视为从 start_layer 到 end_layer。
    #         - 参数非法（数量不为 1 或 2，或值非法）时返回空列表。

    #     :return: 位置列表，按距离从近到远排序。
    #     """
    #     # 参数处理
    #     if len(args) == 1:
    #         low, high = 1, args[0]
    #     elif len(args) == 2:
    #         low, high = args
    #     else:
    #         return []

    #     # 处理无效参数
    #     if high < low or low < 1:
    #         return []

    #     result = []
    #     visited = {(self.col, self.row)}
    #     current_layer = [self]  # 当前层的位置列表

    #     for distance in range(1, high + 1):
    #         next_layer = []
    #         for pos in current_layer:
    #             # 获取六个邻接方向
    #             hex_adjacent = [
    #                 pos.up(),
    #                 pos.down(),
    #                 pos.north(),
    #                 pos.east(),
    #                 pos.west(),
    #                 pos.south(),
    #             ]
    #             for neighbor in hex_adjacent:
    #                 key = (neighbor.col, neighbor.row)
    #                 if key not in visited:
    #                     visited.add(key)
    #                     next_layer.append(neighbor)
    #                     # 如果距离在目标范围内，添加到结果
    #                     if low <= distance <= high:
    #                         result.append(neighbor)

    #         if not next_layer:
    #             break
    #         current_layer = next_layer

    #     return result

class Matrix[T]:
    def __init__(self, size: Size, default: T = None):
        self.size = size
        self._data: list[list[T]] = [[default for _ in range(self.size.cols)] for _ in range(self.size.rows)]

    def _check(self, pos: AbstractPosition):
        if pos.row < 0 or pos.col < 0:
            raise IndexError("position out of bounds")
        rows = len(self._data)
        cols = len(self._data[0]) if rows else 0
        if pos.row >= rows or pos.col >= cols:
            raise IndexError("position out of bounds")

    def get(self, pos: AbstractPosition) -> T:
        self._check(pos)
        return self._data[pos.row][pos.col]

    def set(self, pos: AbstractPosition, value: T) -> None:
        self._check(pos)
        self._data[pos.row][pos.col] = value

    def __getitem__(self, pos: AbstractPosition) -> T:
        return self.get(pos)

    def __setitem__(self, pos: AbstractPosition, value: T) -> None:
        self.set(pos, value)

class Config(TypedDict):
    size: Size
    VALUE: AbstractValue
    MINES: AbstractValue
    mask: List[Position]
    labels: Union[List[str], dict[Position, str]]
    row_col: bool
    interactive: bool
    by_mini: bool
    pos_label: bool

class BoardData(TypedDict):
    config: Config
    variable: Matrix[Optional[IntVar]]
    obj: Matrix[Optional[AbstractValue]]
    type: Matrix[Optional[str]]
    dye: Matrix[Optional[bool]]
    type_special: dict[str, Any]
    variable_special: dict[str, Any]



class Board(AbstractBoard):
    """
    通过实现
    """
    name = "Board2"
    version = 0

    def __init__(self, *, rules=None, size: Optional[Size] = None, default_special: str = 'raw', data: JSONObject = None):
        # traceback.print_stack()
        if rules is None:
            rules = {}
        self._model = None
        self.board_data: dict[str, BoardData] = dict()
        self.default_special = default_special
        self.rules = rules

        if rules:
            for _rules in rules.values():
                for rule in _rules:
                    rule.onboard_init(self)
        if data:
            self.from_json(data)
        else:
            if size is None:
                raise ValueError("board size Undefined")
            self.generate_board(MASTER_BOARD, size=size)
            self.board_data[MASTER_BOARD]["config"]["row_col"] = True
            self.board_data[MASTER_BOARD]["config"]["interactive"] = True
            self.board_data[MASTER_BOARD]["config"]["VALUE"] = VALUE_QUESS
            self.board_data[MASTER_BOARD]["config"]["MINES"] = MINES_TAG

    @overload
    def __call__(
        self, target: str,
        mode: Literal['object', 'obj'],
        key: Optional[str] = None,

        *args: object, **kwargs: object
    ) -> Generator[
        Tuple[
            'Position',
             'AbstractClueValue | AbstractMinesValue | None'
            ],
        None, None
    ]: ...

    @overload
    def __call__(
        self, target: str,
        mode: Literal['variable', 'var'],
        key: Optional[str] = None,
        *args: object, **kwargs: object
    ) -> Generator[
        Tuple[
            'Position', Union[
            'AbstractClueValue',
            'AbstractMinesValue',
            None
            ]],
        None, None
    ]: ...

    @overload
    def __call__(
        self, target: str,
        mode: Literal['dye'],
        key: Optional[str] = None,
        *args: object, **kwargs: object
    ) -> Generator[
        Tuple['Position', bool],
        None, None
    ]: ...

    @overload
    def __call__(
        self, target: str,
        mode: Literal['none'],
        key: Optional[str] = None,
        *args: object, **kwargs: object
    ) -> Generator[
        Tuple['Position', None],
        None, None
    ]: ...

    def __call__(
            self, target: str = "always",
            mode: Literal['object', 'obj', 'type', 'variable', 'var', 'dye', 'none'] = "object",
            key: Optional[str] = None,
            *args: object, **kwargs: object
    ) -> Generator[
        Tuple[
            'Position',
            Union[
            'AbstractClueValue',
            'AbstractMinesValue',
            str, IntVar, bool, None
            ]],
        None, None
    ]:
        """
        被调用时循环返回目标值

        @:param
            target (str): 遍历目标类型。可选值：
                - "C": 线索 (Clue)
                - "F": 雷 (Mines)
                - "N": 未定义或未翻开
                - "always": 默认，遍历所有

            mode (str): 返回的目标类型, 可选值:
                - "object":     存储在board内的实例对象
                - "type":       对象的类型('C', 'F', 'N')
                - "variable":   变量对象
                - "dye":        染色bool

        @:return
            当前位置与选择的值。
        """
        if key is None:
            for key in self.get_interactive_keys():
                for i in self(target=target, mode=mode, key=key, *args, **kwargs):
                    yield i
        else:
            size = self.board_data[key]["config"]["size"]
            for row_idx in range(size.rows):
                for col_idx in range(size.cols):
                    pos = Position(col_idx, row_idx, key)
                    if (_config := self.get_config(pos.board_key, "mask")) and pos in _config:
                        continue
                    pos_type = self.get_type(pos, special='raw')

                    # 检查是否符合目标类型
                    if target == "always" or pos_type in target:
                        if mode == "object":
                            yield pos, self.get_value(pos, *args, **kwargs)
                        elif mode == "obj":
                            yield pos, self.get_value(pos, *args, **kwargs)
                        elif mode == "type":
                            yield pos, self.get_type(pos, *args, **kwargs)
                        elif mode == "var":
                            yield pos, self.get_variable(pos, *args, **kwargs)
                        elif mode == "variable":
                            yield pos, self.get_variable(pos, *args, **kwargs)
                        elif mode == "dye":
                            yield pos, self.get_dyed(pos, *args, **kwargs)
                        elif mode == "none":
                            yield pos, None

    def has(self, target: str, key: Optional[str] = None) -> bool:
        if key not in self.get_board_keys() + [None]:
            return False
        for pos, type_obj in self(mode="type", key=key):
            if type_obj == target:
                return True
        return False

    def get_model(self):
        if self._model is None:
            self._model = cp_model.CpModel()
            random = get_random()
            for _key in self.board_data:
                _size: Size = self.board_data[_key]["config"]["size"]
                variables: Matrix[Optional[IntVar]] = Matrix(_size)
                positions = [Position(col, row, _key) for row in range(_size.rows) for col in range(_size.cols)]
                random.shuffle(positions)
                for pos in positions:
                    variables[pos] = \
                        self._model.NewBoolVar(f"var({self.get_pos(pos.row, pos.col, pos.board_key)})")

                get_logger().trace(f"构建新变量:{variables}")
                self.board_data[_key]["variable"] = variables
        return self._model


    def generate_board(
            self, board_key: str,
            size: Optional[Size] = None,
            labels: list[str] = [],
            true_tag: "AbstractValue" = VALUE_CROSS,
            false_tag: "AbstractValue" = VALUE_CIRCLE,
    ) -> None:
        """
        创建一个新的题板
        :param board_key: 题板名称
        :param size: 题板的尺寸 (与code二选一)
        :param code: 题板的代码 (与size二选一)
        :param true_tag: 题板默认非雷对象
        :param false_tag:题板默认雷对象
        """
        if board_key in self.board_data:
            return


        if size is None:
            raise ValueError("size Undefined")

        config: Config = {
            "size": size,
            "VALUE": true_tag,
            "MINES": false_tag,
            "mask": [],
            "labels": labels,
            "row_col": False,
            "interactive": False,
            "by_mini": False,
            "pos_label": False
        }
        data: BoardData = {
            "config": config,
            "variable": Matrix(size, None),
            "obj": Matrix(size, None),
            "type": Matrix(size, "N"),
            "dye": Matrix(size, False),
            "type_special": {},
            "variable_special": {}
        }
        self.board_data[board_key] = data
        self.labels = labels

    def json(self) -> JSONObject:
        boards: dict[str, JSONObject] = {}
        for board_key in self.board_data:
            cfg = self.board_data[board_key]["config"]
            size = cfg["size"]
            value = cfg["VALUE"]
            mines = cfg["MINES"]
            labels = cfg.get("labels")
            if isinstance(labels, dict):
                labels = ImmutableDict(labels)
            if isinstance(labels, list):
                labels = tuple(labels)
            flags = {name: bool(cfg.get(name, False)) for name in self.CONFIG_FLAGS}

            # mask as list of booleans (row-major: cols x rows)
            mask = []
            for col_idx in range(size.cols):
                for row_idx in range(size.rows):
                    pos = self.get_pos(row_idx, col_idx, board_key)
                    mask.append(pos is None)

            # cells list with position and object info
            cells: list[JSONObject] = []
            for pos, obj in self(key=board_key, mode="object"):
                cell: dict[str, int | str | bool | None | ImmutableDict[str, int | str | bool | None]] = {
                    "col": pos.col,
                    "row": pos.row,
                    "board_key": pos.board_key,
                    "dyed": bool(self.get_dyed(pos))
                }

                obj: Optional[AbstractClueValue | AbstractMinesValue]
                if obj is None:
                    cell["type"] = None
                    cell["data"] = None
                else:
                    cell["type"] = obj.type().decode("ascii")
                    cell["data"] = ImmutableDict(obj.json())
                cells.append(ImmutableDict(cell))

            boards[board_key] = ImmutableDict({
                "size": ImmutableDict({"cols": size.cols, "rows": size.rows}),
                "flags": ImmutableDict(flags),
                "mask": tuple(mask),
                "value": {"type": value.type().decode("ascii"),"data": value.json()},
                "mines": {"type": mines.type().decode("ascii"),"data": mines.json()},
                "labels": labels,
                "cells": tuple(cells)
            })

        return jsonify({"boards": boards})


    def from_json(self, data: JSONObject) -> None:
        """Load board state from json produced by json()"""

        from minesweepervariants.impl.impl_obj import get_value
        from minesweepervariants.impl.impl_obj import get_value_type

        boards = get_with_valid(data, "boards", ImmutableDict[str, JSONObject])
        # clear existing
        self.board_data = {}
        for board_key, cfg in boards.items():
            size_obj = get_with_valid(cfg, "size", ImmutableDict[str, JSONObject])
            cols = get_with_valid(size_obj, "cols", int)
            rows = get_with_valid(size_obj, "rows", int)
            size = Size(cols, rows)
            labels = get_with_valid(cfg, "labels", tuple)
            labels_list: list[str] = []
            for label in labels:
                assert valid(label, str)
                labels_list.append(label)
            self.generate_board(board_key, size=size, labels=labels_list)

            # flags
            flags = get_with_valid(cfg, "flags", ImmutableDict[str, JSONObject])
            for name, val in flags.items():
                self.set_config(board_key, name, bool(val))

            # mask
            mask_list = get_with_valid(cfg, "mask", tuple)
            for col_idx in range(size.cols):
                for row_idx in range(size.rows):
                    idx = col_idx * size.rows + row_idx
                    if idx < len(mask_list) and mask_list[idx]:
                        pos = self.get_pos(row_idx, col_idx, board_key)
                        if pos is not None:
                            self.set_mask(pos)

            # value and mines templates

            v = get_with_valid(cfg, "value", ImmutableDict[str, JSONObject])
            m = get_with_valid(cfg, "mines", ImmutableDict[str, JSONObject])
            if v:
                type_ = get_with_valid(v, "type", str)
                data_ = get_with_valid(v, "data", ImmutableDict[str, JSONObject])
                self.board_data[board_key]["config"]["VALUE"] = get_value(POSITION_TAG, type_, data_)
            if m:
                type_ = get_with_valid(m, "type", str)
                data_ = get_with_valid(m, "data", ImmutableDict[str, JSONObject])
                self.board_data[board_key]["config"]["MINES"] = get_value(POSITION_TAG, type_, data_)


            # cells
            for cell in get_with_valid(cfg, "cells", tuple[JSONObject, ...]):
                col = get_with_valid(cell, "col", int)
                row = get_with_valid(cell, "row", int)
                pos = self.get_pos(row, col, board_key)
                if pos is None:
                    continue
                if get_with_valid(cell, "dyed", bool):
                    self.set_dyed(pos, True)
                assert valid(cell, ImmutableDict[str, JSONObject])

                if 'type' not in cell or cell['type'] is None:
                    self.set_value(pos, None)
                    continue
                obj_data: ImmutableDict[str, JSONObject] = get_with_valid(cell, "data", ImmutableDict[str, JSONObject])
                type_ = get_with_valid(cell, "type", str)
                value_obj = get_value(pos, type_, obj_data)
                if isinstance(value_obj, (AbstractMinesValue, AbstractClueValue)):
                    self.set_value(pos, value_obj)


    def boundary(self, key=MASTER_BOARD) -> "Position":
        if key not in self.get_board_keys():
            return Position(-1, -1, key)
        size = self.board_data[key]["config"]["size"]
        return Position(size.cols - 1, size.rows - 1, key)

    def is_valid(self, pos: 'AbstractPosition') -> bool:
        if pos in self.get_config(pos.board_key, "mask"):
            return False
        return super().is_valid(pos)

    @staticmethod
    def type_value(value) -> str:
        # 查看value的类型
        if value is None:
            return "N"
        elif isinstance(value, AbstractMinesValue):
            return "F"
        elif isinstance(value, AbstractClueValue):
            return "C"
        get_logger().error(f"unknown type: value{value}, type{type(value)}")
        raise ValueError(f"unknown type: {value}, type{type(value)}")

    def register_type_special(self, name: str, func):
        for key in self.board_data:
            if "type_special" not in self.board_data[key]:
                self.board_data[key]["type_special"] = dict()

            self.board_data[key]["type_special"][name] = func

    def get_type(self, pos: 'Position', special: str = '', *args, **kwargs) -> str:
        special = special or self.default_special

        key = pos.board_key

        if self.is_valid(pos):
            if special == 'raw':
                return self.board_data[key]["type"][pos]

            if "type_special" not in self.board_data[key] or \
                    special not in self.board_data[key]["type_special"]:
                raise ValueError(f"unknown special type: {special}")

            return self.board_data[key]["type_special"][special](self, pos, *args, **kwargs)

        return ""

    def get_value(self, pos: 'Position', *args, **kwargs) -> Union['AbstractClueValue', 'AbstractMinesValue', None]:
        key = pos.board_key
        if self.is_valid(pos):
            return self.board_data[key]["obj"][pos]
        return None

    def set_value(self, pos: 'Position', value):
        key = pos.board_key
        if self.is_valid(pos):
            self.board_data[key]["obj"][pos] = value
            self.board_data[key]["type"][pos] = self.type_value(value)

    def get_dyed(self, pos: 'Position', *args, **kwargs) -> bool | None:
        key = pos.board_key
        if self.is_valid(pos):
            return self.board_data[key]["dye"][pos]

    def set_dyed(self, pos: 'Position', dyed: bool):
        key = pos.board_key
        if self.is_valid(pos):
            self.board_data[key]["dye"][pos] = dyed

    def get_variable(self, pos: 'Position', special: str = '', *args, **kwargs) -> IntVar | None:
        special = special or self.default_special
        # if special != 'raw':
        #     s = "".join(traceback.format_stack())
        #     if "V.py" not in s and "3I" not in s:
        #         print(s)
        #         print(f'-{special}------------------------------------------------------------------------------')
        #         ...
        # if special == 'raw':
        #     s = "".join(traceback.format_stack())
        #     if "V.py" in s or "3I" in s:
        #         print(s)
        #         print(f'-raw------------------------------------------------------------------------------')
        #         pass

        key = pos.board_key
        self.get_model()
        if self.is_valid(pos):
            if special == 'raw':
                return self.board_data[key]["variable"][pos]

            if "variable_special" not in self.board_data[key]:
                self.board_data[key]["variable_special"] = dict()

            if special not in self.board_data[key]["variable_special"]:
                self.board_data[key]["variable_special"][special] = dict()

            if (pos.col, pos.row) not in self.board_data[key]["variable_special"][special]:
                self.board_data[key]["variable_special"][special][(pos.col, pos.row)] = \
                    self._model.NewIntVar(-999, 999, f"var_{special}({self.get_pos(pos.row, pos.col, key)})")
            return self.board_data[key]["variable_special"][special][(pos.col, pos.row)]

    def clear_variable(self):
        for key in self.board_data.keys():
            if "variable" in self.board_data[key]:
                del self.board_data[key]["variable"]
            if "variable_special" in self.board_data[key]:
                del self.board_data[key]["variable_special"]
        self._model = None
        gc.collect()

    def get_config(self, board_key, config_name):
        if board_key not in self.board_data:
            return None
        return self.board_data[board_key]["config"][config_name]

    def set_config(self, board_key, config_name, value: bool):
        if board_key not in self.board_data:
            return None
        self.board_data[board_key]["config"][config_name] = value

    def set_mask(self, pos):
        if not self.is_valid(pos):
            return
        self.get_config(pos.board_key, "mask").append(pos)

    def get_row_pos(self, pos: 'Position') -> List["Position"]:
        bound = self.boundary(pos.board_key)
        _pos = pos.clone()
        pos_list = [_pos]
        while True:
            _pos = _pos.left()
            if not _pos.in_bounds(bound):
                break
            pos_list.append(_pos)
        _pos = pos.clone()
        pos_list = pos_list[::-1]
        while True:
            _pos = _pos.right()
            if not _pos.in_bounds(bound):
                break
            pos_list.append(_pos)
        return pos_list

    def get_col_pos(self, pos: 'Position') -> List["Position"]:
        bound = self.boundary(pos.board_key)
        _pos = pos.clone()
        pos_list = [_pos]
        while True:
            _pos = _pos.up()
            if not _pos.in_bounds(bound):
                break
            pos_list.append(_pos)
        _pos = pos.clone()
        pos_list = pos_list[::-1]
        while True:
            _pos = _pos.down()
            if not _pos.in_bounds(bound):
                break
            pos_list.append(_pos)
        return pos_list


    def get_pos(self, row: int, col: int, key=MASTER_BOARD) -> Union['Position', None]:
        size = self.board_data[key]["config"]["size"]
        if -size.cols < col < size.cols and -size.rows < row < size.rows:
            col = col if col >= 0 else size.cols + col
            row = row if row >= 0 else size.rows + row
            pos = Position(col, row, key)
            if self.is_valid(pos):
                return pos
        return None

    def get_pos_box(self, pos1: "AbstractPosition", pos2: "AbstractPosition") -> List["AbstractPosition"]:
        if pos1.board_key != pos2.board_key:
            return []
        if not (self.in_bounds(pos1) and self.in_bounds(pos2)):
            return []
        c_min, c_max = sorted([pos1.col, pos2.col])
        r_min, r_max = sorted([pos1.row, pos2.row])

        result = []
        for row in range(r_min, r_max + 1):
            for col in range(c_min, c_max + 1):
                result.append(self.get_pos(row, col, key=pos1.board_key))
        return result

    def batch(self, positions: List['Position'],
              mode: str, drop_none: bool = False, *args, **kwargs) -> List[Any]:
        result = []
        for pos in positions:
            if drop_none and not self.in_bounds(pos):
                continue
            if mode == "object":
                result.append(self.get_value(pos, *args, **kwargs))
            elif mode == "obj":
                result.append(self.get_value(pos, *args, **kwargs))
            elif mode == "variable":
                result.append(self.get_variable(pos, *args, **kwargs))
            elif mode == "var":
                result.append(self.get_variable(pos, *args, **kwargs))
            elif mode == "type":
                result.append(self.get_type(pos, *args, **kwargs))
            elif mode == "dye":
                result.append(self.get_dyed(pos, *args, **kwargs))
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        return result

    def clear_board(self):
        for key in self.board_data:
            data = self.board_data[key]
            size = data["config"]["size"]
            data["obj"] = Matrix(size, None)
            data["type"] = Matrix(size, "N")
            self.clear_variable()

    def get_board_keys(self) -> list[str]:
        return list(self.board_data.keys())

    def show_board(self, show_tag: bool = False):
        r = ""
        for key in self.board_data:
            size = self.board_data[key]["config"]["size"]
            if len(self.board_data.keys()) > 1:
                r += key + "\n"
            for row_idx in range(size.rows):
                for col_idx in range(size.cols):
                    pos = self.get_pos(row_idx, col_idx, key)
                    if pos is None:
                        r += "\t\t" if show_tag else "\t"
                        continue
                    value = self[pos]
                    if value is None:
                        r += "______" if show_tag else "___"
                    else:
                        r += str(value) + ("_" + value.type().decode() if show_tag else "")
                    r += "\t"
                r += "\n"
            r += "\n\n"
        return r[:-2]

    def show_board_discord(self, answer_board=None, hide_clues=False):
        """
        展示题板，使用Discord剧透格式（||spoiler||）包裹未挖出的格子，
        并使用Discord emoji和格式化字符显示特殊值。

        参数:
            answer_board: 答案题板，用于在未挖出格子处显示正确答案（用剧透包裹）
            hide_clues: 是否隐藏未挖出的线索（仅显示问号和雷），默认False

        - 已挖出的格子：直接显示其值，使用emoji和格式化
        - 未挖出的格子(None)：
          - 当hide_clues=False时：显示答案题板中对应位置的值，用||xxx||包裹（剧透）
          - 当hide_clues=True时：如果是线索数字则用||___||隐藏，问号和雷则用||xxx||显示
        - 单数字线索(0-9)：使用数字emoji (0️⃣-9️⃣)
        - 雷标签(F)：:flag:
        - 问号(?)：❓
        """
        digit_emojis = {
            '0': '0️⃣', '1': '1️⃣', '2': '2️⃣', '3': '3️⃣', '4': '4️⃣',
            '5': '5️⃣', '6': '6️⃣', '7': '7️⃣', '8': '8️⃣', '9': '9️⃣'
        }

        def format_value(value, is_spoiler=False):
            """格式化单个值以供Discord显示"""
            if value is None:
                content = "__"
            else:
                value_str = str(value)

                # 处理雷标签
                if value is MINES_TAG or value_str == "雷" or value_str == "F":
                    content = ":flag:"
                # 处理问号
                elif value is VALUE_QUESS or value_str == "?":
                    content = "❓"
                # 处理单数字
                elif value_str.isdigit() and len(value_str) == 1:
                    content = digit_emojis[value_str]
                else:
                    content = value_str

            # 如果是剧透（未挖出的格子），包裹在||...||中
            if is_spoiler:
                return f"||{content}||"
            else:
                return content

        r = ""
        for key in self.board_data:
            size = self.board_data[key]["config"]["size"]
            if len(self.board_data.keys()) > 1:
                r += key + "\n"
            for row in range(size.rows):
                for col in range(size.cols):
                    pos = self.get_pos(row, col, key)
                    if pos is None:
                        continue

                    current_value = self[pos]

                    # 如果当前值为None（未挖出），尝试从答案题板获取值
                    if current_value is None and answer_board is not None:
                        answer_value = answer_board[pos]

                        # 判断是否需要隐藏线索数字
                        hide_as_unknown = False
                        if hide_clues and answer_value is not None:
                            answer_str = str(answer_value)
                            # 如果是线索数字（不是问号、雷），则隐藏为问号或雷
                            if answer_str.isdigit() and len(answer_str) == 1:
                                hide_as_unknown = True

                        if hide_as_unknown:
                            # 雷线索隐藏为flag，数字线索隐藏为问号
                            if answer_value is MINES_TAG or str(answer_value) in ("雷", "F"):
                                formatted = "||:flag:||"
                            else:
                                formatted = "||❓||"
                        else:
                            formatted = format_value(answer_value, is_spoiler=True)
                    else:
                        # 已挖出的格子直接显示
                        formatted = format_value(current_value, is_spoiler=False)

                    r += formatted
                r += "\n"
            r += "\n"
        return r.rstrip()

    def pos_label(self, pos: 'AbstractPosition') -> str:
        labels = self.get_config(pos.board_key, "labels")
        if type(labels) is dict:
            if pos in labels:
                return labels[pos]
            else:
                return ""
        txt = chr(64 + pos.col // 26) if pos.col > 25 else ''
        txt += chr(pos.col % 26 + 65)
        txt += '='
        txt += labels[pos.row] if pos.row < len(labels) else str(pos.row)
        return txt
