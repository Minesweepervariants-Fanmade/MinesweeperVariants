#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time    : 2025/06/03 01:58
# @Author  : Wu_RH
# @FileName: board.py

from typing import (
    Generator,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
    Callable,
)
import heapq

import gc
from ortools.sat.python.cp_model import CpModel, IntVar

from minesweepervariants.abs.Lrule import AbstractMinesRule

from ....abs.rule import AbstractValue, AbstractRule
from ....utils.impl_obj import VALUE_QUESS, MINES_TAG
from ....utils.impl_obj import POSITION_TAG, VALUE_CROSS, VALUE_CIRCLE
from ....utils.tool import get_logger, get_random
from ....abs.board import AbstractBoard, AbstractPosition, MASTER_BOARD, RulesDict, Size
from ....abs.Rrule import AbstractClueRule, AbstractClueValue
from ....abs.Mrule import AbstractMinesClueRule, AbstractMinesValue

def get_value(pos: object | None = None, code: bytes | None = None):
    from minesweepervariants.impl.impl_obj import get_value

    return get_value(pos, code)


def alpha(col: int) -> str:
    alpha_map = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if col < 26:
        return alpha_map[col]
    return alpha_map[col // 26 - 1] + alpha_map[col % 26]


def encode_int_7bit(n: int) -> bytes:
    # 编码主体：每7位 -> 1字节（bit6~bit0，bit7=0）
    if n == 0:
        return b"\x00"
    payload = []

    while n > 0:
        payload.append(n & 0x7F)
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
        return (
            f"{self.board_key + ':' if self.board_key != MASTER_BOARD else ''}"
            f"{alpha(self.col)}{self.row + 1}"
        )

    def _up(self, n: int = 1):
        self.row -= n

    def _down(self, n: int = 1):
        self.row += n

    def _left(self, n: int = 1):
        self.col -= n

    def _right(self, n: int = 1):
        self.col += n

    def _deviation(self, pos: AbstractPosition) -> None:
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

    def in_bounds(self, bound_pos: AbstractPosition) -> bool:
        if bound_pos.board_key != self.board_key:
            return False
        return 0 <= self.col <= bound_pos.col and 0 <= self.row <= bound_pos.row

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
        directions = [
            (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)
        ]

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
        self._data: list[list[T]] = [
            [default for _ in range(self.size.cols)] for _ in range(self.size.rows)
        ]

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
    labels: object
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
    type_special: dict[str, Callable[..., str]]
    variable_special: dict[str, dict[Tuple[int, int], IntVar]]



class Board(AbstractBoard):
    """
    通过实现
    """

    CONFIG_FLAGS: List[str] = ["row_col", "interactive", "by_mini", "pos_label"]

    name = "Board2"
    version = 0

    _model: CpModel | None
    board_data: dict[str, BoardData]
    default_special: str

    def __init__(
        self,
        *,
        rules: RulesDict | None,
        size: Optional[Size] = None,
        code: Optional[bytes] = None,
        default_special: str = "raw",
    ):
        self._model = None
        self.board_data= dict()
        self.default_special = default_special
        self.rules = {"clue_rules": [], "mines_rules": [], "mines_clue_rules": []}

        if rules is not None:
            self.rules.update(rules)

        for l_rules in self.rules["mines_rules"]:
            l_rules.onboard_init(self)

        for m_rules in self.rules["mines_clue_rules"]:
            m_rules.onboard_init(self)

        for r_rules in self.rules["clue_rules"]:
            r_rules.onboard_init(self)

        if code is None:
            if size is None:
                raise ValueError("board size Undefined")
            self.generate_board(MASTER_BOARD, size=size)
            self.board_data[MASTER_BOARD]["config"]["row_col"] = True
            self.board_data[MASTER_BOARD]["config"]["interactive"] = True
            self.board_data[MASTER_BOARD]["config"]["VALUE"] = VALUE_QUESS
            self.board_data[MASTER_BOARD]["config"]["MINES"] = MINES_TAG
            return
        for chunks in code.split(b"\xff\xff"):
            board_key, chunks = chunks.split(b"\xff", 1)
            board_key = board_key.decode("ascii")
            self.generate_board(board_key, code=chunks)

    def __call__(
        self,
        target: str | None = "always",
        mode: str = "object",
        key: Optional[str] = None,
        *args: object,
        **kwargs: object,
    ) -> Generator[Tuple["Position", object], None, None]:
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
                for i in self(target=target, mode=mode, key=key):
                    yield i
        else:
            size = self.board_data[key]["config"]["size"]
            for row_idx in range(size.rows):
                for col_idx in range(size.cols):
                    pos = Position(col_idx, row_idx, key)
                    _config = self.get_config(pos.board_key, "mask")
                    if isinstance(_config, list) and pos in _config:
                        continue
                    pos_type = self.get_type(pos, special="raw")

                    # 检查是否符合目标类型
                    if target is None or target == "always" or pos_type in target:
                        if mode == "object":
                            yield pos, self.get_value(pos)
                        elif mode == "obj":
                            yield pos, self.get_value(pos)
                        elif mode == "type":
                            yield pos, self.get_type(pos)
                        elif mode == "var":
                            yield pos, self.get_variable(pos)
                        elif mode == "variable":
                            yield pos, self.get_variable(pos)
                        elif mode == "dye":
                            yield pos, self.get_dyed(pos)
                        elif mode == "none":
                            yield pos, None

    def has(self, target: str, key: Optional[str] = None) -> bool:
        if key not in self.get_board_keys() + [None]:
            return False
        for _pos, type_obj in self(mode="type", key=key):
            if type_obj == target:
                return True
        return False

    def get_model(self) -> CpModel:
        if self._model is None:
            self._model = CpModel()
        model = self._model
        random = get_random()
        for _key in self.board_data:
            _size: Size = self.board_data[_key]["config"]["size"]
            variables: Matrix[Optional[IntVar]] = Matrix(_size)
            positions = [
                Position(col, row, _key)
                for row in range(_size.rows)
                for col in range(_size.cols)
            ]
            random.shuffle(positions)
            for pos in positions:
                variables[pos] = model.new_bool_var(
                    f"var({self.get_pos(pos.row, pos.col, pos.board_key)})"
                )

            get_logger().trace(f"构建新变量:{variables}")
            self.board_data[_key]["variable"] = variables
        return model

    def generate_board(
        self,
        board_key: str,
        size: Optional[Size] = None,
        labels: object | None = None,
        code: Optional[bytes] = None,
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
        flag_byte = 0
        mask = 0
        labels_obj: object = [] if labels is None else labels
        if code is not None:
            encoded_parts = code.split(b"\xff", 5)
            config_bytes = encoded_parts[0]
            mask = encoded_parts[1]
            ture_code = encoded_parts[2]
            false_code = encoded_parts[3]
            labels_code = encoded_parts[4]
            payload_parts = encoded_parts[5:]
            code = b"".join(payload_parts)
            size_cols, size_rows, flag_byte = config_bytes
            size = Size(cols=size_cols, rows=size_rows)
            mask = decode_bytes_7bit(mask)
            if len(labels_code) == 1 and labels_code[0] == 0:
                labels_obj = []
            elif labels_code[0] == 1:
                labels_obj = labels_code[1:].decode("ascii").split(";")
            elif labels_code[0] == 2:
                labels_dict: dict[Position, str] = {}
                for label in labels_code[1:].split(b"\xfe"):
                    if not label:
                        continue
                    pos = Position(
                        label[0], label[1], label[2 : label.index(253)].decode("ascii")
                    )
                    labels_dict[pos] = label[label.index(253) + 1 :].decode("ascii")
                labels_obj = labels_dict
            true_tag_obj = get_value(pos=POSITION_TAG, code=ture_code)
            false_tag_obj = get_value(pos=POSITION_TAG, code=false_code)
            if true_tag_obj is None or false_tag_obj is None:
                raise ValueError("invalid board tags in encoded data")
            true_tag = true_tag_obj
            false_tag = false_tag_obj
        if size is None:
            raise ValueError("size Undefined")

        board_config: Config = {
            "size": size,
            "VALUE": true_tag,
            "MINES": false_tag,
            "mask": [],
            "labels": labels_obj,
            "row_col": False,
            "interactive": False,
            "by_mini": False,
            "pos_label": False,
        }
        data: BoardData = {
            "config": board_config,
            "variable": Matrix(size, None),
            "obj": Matrix(size, None),
            "type": Matrix(size, "N"),
            "dye": Matrix(size, False),
            "type_special": {},
            "variable_special": {},
        }
        self.board_data[board_key] = data
        self.labels = labels_obj

        if code is None:
            return
        positions = [pos for pos, _ in self(key=board_key, mode="none")]
        for pos in positions[::-1]:
            if mask & 1:
                self.set_mask(pos)
            mask >>= 1

        data["config"]["by_mini"] = bool(flag_byte & (1 << 3))
        data["config"]["pos_label"] = bool(flag_byte & (1 << 2))
        data["config"]["interactive"] = bool(flag_byte & (1 << 1))
        data["config"]["row_col"] = bool(flag_byte & 1)

        codes = code.split(b"\xff")

        code_queue: List[bytes] = []

        for part in codes:
            if not part:
                continue
            if part[0] == 0:
                count = int(part[1])
                code_queue.extend([b"_"] * count)
            else:
                code_queue.append(part)

        for _pos, _ in self(key=board_key):
            code = code_queue.pop(0)
            if code[0] == 35:
                self.set_dyed(_pos, True)
                code = code[1:]
            if code == b"_":
                continue
            value = get_value(_pos, code)
            if value is not None:
                self.set_value(_pos, value)
                continue
            raise ValueError(f"unknown type{code}")

        for rule in self.rules["mines_rules"]:
            rule.onboard_init(self)

        for rule in self.rules["mines_clue_rules"]:
            rule.onboard_init(self)

        for rule in self.rules["clue_rules"]:
            rule.onboard_init(self)

    def encode(self) -> bytes:
        """
        字节头: 尺寸
        无需换行符 初始化自动排序
        '_'表示None
        :return: 字节码
        """
        board_bytes = bytearray()
        for board_key in self.board_data:
            size = self.board_data[board_key]["config"]["size"]
            value = self.board_data[board_key]["config"]["VALUE"]
            mines = self.board_data[board_key]["config"]["MINES"]
            labels = self.board_data[board_key]["config"]["labels"]
            flags = 0
            mask = 0
            for name in self.CONFIG_FLAGS:
                flags = (flags << 1) | int(
                    self.board_data[board_key]["config"].get(name, False)
                )
            for col_idx in range(size.cols):
                for row_idx in range(size.rows):
                    pos = self.get_pos(row_idx, col_idx, board_key)
                    mask <<= 1
                    if pos is None:
                        mask |= 1
            board_bytes.extend(board_key.encode("ascii") + b"\xff")
            board_bytes.extend(bytes([size.cols, size.rows, flags, 255]))
            board_bytes.extend(encode_int_7bit(mask) + bytes([255]))
            board_bytes.extend(value.type() + b"|" + value.code())
            board_bytes.extend(bytes([255]))
            board_bytes.extend(mines.type() + b"|" + mines.code())
            board_bytes.extend(bytes([255]))
            if isinstance(labels, dict):
                if any(isinstance(i, AbstractPosition) for i in labels.keys()):
                    raise ValueError("invalid label key")
                board_bytes.extend([2])
                for pos, str_value in labels.items():
                    board_bytes.extend([pos.col, pos.row])
                    board_bytes.extend(pos.board_key.encode("ascii"))
                    board_bytes.extend([253])
                    board_bytes.extend(str_value.encode("ascii"))
                    board_bytes.extend([254])
                if labels:
                    board_bytes.pop(-1)
            elif isinstance(labels, list):
                board_bytes.extend(
                    b"\x01" + (";".join(label for label in labels)).encode("ascii")
                )
            else:
                board_bytes.extend(b"\x00")
            # key | sizex | sizey | config
            for pos, obj in self(key=board_key):
                board_bytes.extend(b"\xff")
                if self.get_dyed(pos):
                    board_bytes.extend(b"#")
                if obj is None:
                    board_bytes.extend(b"_")
                else:
                    if isinstance(obj, AbstractValue):
                        code = obj.code()
                        if b"\xff" in code:
                            get_logger().error(f"{obj.type().decode()}中编码出现\\xff")
                            raise ValueError(r"code contains forbidden byte: \xff")
                        board_bytes.extend(obj.type() + b"|" + code)
                    else:
                        board_bytes.extend(b"_")
            board_bytes.extend(b"\xff\xff")
        # 只用split(b"\xff_")切分
        parts: list[bytearray] = board_bytes.split(b"\xff_")

        # 处理连续 \xff_ 的次数
        encoded_bytes = bytearray()
        i = 0
        while i < len(parts):
            if i > 0:
                # 统计连续 \xff_ 的次数
                count = 1
                # 看后续parts里是否以空字节开头来判断是否连续（split后的空串）
                # 但这里由于只分割 \xff_，连续情况只能靠检查下一个part是否空
                while i + count < len(parts) and len(parts[i + count - 1]) == 0:
                    count += 1
                # 输出 \xff + 数字（表示连续多少个 \xff_）
                while count > 254:
                    encoded_bytes.extend(b"\xff\x00" + bytes([254]))
                    count -= 254
                if count > 0:
                    encoded_bytes.extend(b"\xff\x00" + bytes([count]))
                i += count - 1
            encoded_bytes.extend(parts[i])
            i += 1
        return bytes(encoded_bytes[:-2])

    def boundary(self, key: str = MASTER_BOARD) -> "Position":
        if key not in self.get_board_keys():
            return Position(-1, -1, key)
        size = self.board_data[key]["config"]["size"]
        return Position(size.cols - 1, size.rows - 1, key)

    def is_valid(self, pos: "AbstractPosition") -> bool:
        mask = self.get_config(pos.board_key, "mask")
        if isinstance(mask, list) and pos in mask:
            return False
        return super().is_valid(pos)

    @staticmethod
    def type_value(value: object) -> str:
        # 查看value的类型
        if value is None:
            return "N"
        elif isinstance(value, AbstractMinesValue):
            return "F"
        elif isinstance(value, AbstractClueValue):
            return "C"
        get_logger().error(f"unknown type: value{value}, type{type(value)}")
        raise ValueError(f"unknown type: {value}, type{type(value)}")

    def register_type_special(self, name: str, func: Callable[..., str]) -> None:
        for key in self.board_data:
            if "type_special" not in self.board_data[key]:
                self.board_data[key]["type_special"] = dict()

            self.board_data[key]["type_special"][name] = func

    def get_type(
        self,
        pos: "AbstractPosition",
        special: str = "",
        *args: object,
        **kwargs: object,
    ) -> str:
        special = special or self.default_special

        key = pos.board_key

        if self.is_valid(pos):
            if special == "raw":
                val = self.board_data[key]["type"][pos]
                return val if val is not None else ""

            if (
                "type_special" not in self.board_data[key]
                or special not in self.board_data[key]["type_special"]
            ):
                raise ValueError(f"unknown special type: {special}")

            callback = self.board_data[key]["type_special"][special]
            if callable(callback):
                return callback(self, pos, *args, **kwargs)
            raise TypeError(f"unknown special type handler: {special}")

        return ""

    def get_value(
        self, pos: "AbstractPosition", *args: object, **kwargs: object
    ) -> Union["AbstractClueValue", "AbstractMinesValue", None]:
        key = pos.board_key
        if self.is_valid(pos):
            value = self.board_data[key]["obj"][pos]
            if isinstance(value, (AbstractClueValue, AbstractMinesValue)):
                return value
            return None
        return None

    def set_value(
        self,
        pos: "AbstractPosition",
        value: Union["AbstractClueValue", "AbstractMinesValue", None],
    ) -> None:
        key = pos.board_key
        if self.is_valid(pos):
            self.board_data[key]["obj"][pos] = value
            self.board_data[key]["type"][pos] = self.type_value(value)

    def get_dyed(
        self, pos: "AbstractPosition", *args: object, **kwargs: object
    ) -> bool | None:
        key = pos.board_key
        if self.is_valid(pos):
            return self.board_data[key]["dye"][pos]
        return False

    def set_dyed(self, pos: "AbstractPosition", dyed: bool) -> None:
        key = pos.board_key
        if self.is_valid(pos):
            self.board_data[key]["dye"][pos] = dyed

    def get_variable(
        self,
        pos: "AbstractPosition",
        special: str = "",
        *args: object,
        **kwargs: object,
    ) -> IntVar | None:
        special = special or self.default_special

        key = pos.board_key
        model = self.get_model()
        if self.is_valid(pos):
            if special == "raw":
                return self.board_data[key]["variable"][pos]

            if "variable_special" not in self.board_data[key]:
                self.board_data[key]["variable_special"] = dict()

            if special not in self.board_data[key]["variable_special"]:
                self.board_data[key]["variable_special"][special] = dict()

            if (pos.col, pos.row) not in self.board_data[key]["variable_special"][
                special
            ]:
                self.board_data[key]["variable_special"][special][
                    (pos.col, pos.row)
                ] = model.new_int_var(
                    -999, 999, f"var_{special}({self.get_pos(pos.row, pos.col, key)})"
                )
            return self.board_data[key]["variable_special"][special][(pos.col, pos.row)]
        raise IndexError("position out of bounds")

    def clear_variable(self) -> None:
        for key in self.board_data.keys():
            size = self.board_data[key]["config"]["size"]
            self.board_data[key]["variable"] = Matrix(size, None)
            self.board_data[key]["variable_special"] = {}
        self._model = None
        gc.collect()

    def get_config(self, board_key: str, config_name: str) -> object:
        if board_key not in self.board_data:
            return None
        return self.board_data[board_key]["config"][config_name]

    def set_config(self, board_key: str, config_name: str, value: object) -> None:
        if board_key not in self.board_data:
            return None
        self.board_data[board_key]["config"][config_name] = value

    def set_mask(self, pos: "AbstractPosition") -> None:
        if not self.is_valid(pos):
            return
        mask = self.get_config(pos.board_key, "mask")
        if isinstance(mask, list):
            mask.append(pos)

    def get_row_pos(self, pos: "AbstractPosition") -> List["AbstractPosition"]:
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

    def get_col_pos(self, pos: "AbstractPosition") -> List["AbstractPosition"]:
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

    def get_pos(
        self, row: int, col: int, key: str = MASTER_BOARD
    ) -> Union["Position", None]:
        size = self.board_data[key]["config"]["size"]
        if -size.cols < col < size.cols and -size.rows < row < size.rows:
            col = col if col >= 0 else size.cols + col
            row = row if row >= 0 else size.rows + row
            pos = Position(col, row, key)
            if self.is_valid(pos):
                return pos
        return None

    def get_pos_box(
        self, pos1: "AbstractPosition", pos2: "AbstractPosition"
    ) -> List["AbstractPosition"]:
        if pos1.board_key != pos2.board_key:
            return []
        if not (self.in_bounds(pos1) and self.in_bounds(pos2)):
            return []
        c_min, c_max = sorted([pos1.col, pos2.col])
        r_min, r_max = sorted([pos1.row, pos2.row])

        result = []
        for row in range(r_min, r_max + 1):
            for col in range(c_min, c_max + 1):
                pos = self.get_pos(row, col, key=pos1.board_key)
                if pos is not None:
                    result.append(pos)
        return result

    def batch(
        self,
        positions: List["AbstractPosition"],
        mode: str,
        drop_none: bool = False,
        *args: object,
        **kwargs: object,
    ) -> List[object]:
        result = []
        for pos in positions:
            if drop_none and not self.in_bounds(pos):
                continue
            if mode == "object":
                result.append(self.get_value(pos))
            elif mode == "obj":
                result.append(self.get_value(pos))
            elif mode == "variable":
                result.append(self.get_variable(pos))
            elif mode == "var":
                result.append(self.get_variable(pos))
            elif mode == "type":
                result.append(self.get_type(pos))
            elif mode == "dye":
                result.append(self.get_dyed(pos))
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        return result

    def clear_board(self) -> None:
        for key in self.board_data:
            data = self.board_data[key]
            size = data["config"]["size"]
            data["obj"] = Matrix(size, None)
            data["type"] = Matrix(size, "N")
            self.clear_variable()

    def get_board_keys(self) -> list[str]:
        return list(self.board_data.keys())

    def show_board(self, show_tag: bool = False) -> str:
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
                        r += str(value) + (
                            "_" + value.type().decode() if show_tag else ""
                        )
                    r += "\t"
                r += "\n"
            r += "\n\n"
        return r[:-2]

    def show_board_discord(
        self, answer_board: AbstractBoard | None = None, hide_clues: bool = False
    ) -> str:
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
            "0": "0️⃣",
            "1": "1️⃣",
            "2": "2️⃣",
            "3": "3️⃣",
            "4": "4️⃣",
            "5": "5️⃣",
            "6": "6️⃣",
            "7": "7️⃣",
            "8": "8️⃣",
            "9": "9️⃣",
        }

        def format_value(value: object | None, is_spoiler: bool = False) -> str:
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
                            if answer_value is MINES_TAG or str(answer_value) in (
                                "雷",
                                "F",
                            ):
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

    def pos_label(self, pos: "AbstractPosition") -> str:
        labels = self.get_config(pos.board_key, "labels")
        if isinstance(labels, dict):
            if pos in labels:
                return labels[pos]
            else:
                return ""
        if not isinstance(labels, list):
            return ""
        txt = chr(64 + pos.col // 26) if pos.col > 25 else ""
        txt += chr(pos.col % 26 + 65)
        txt += "="
        txt += labels[pos.row] if pos.row < len(labels) else str(pos.row)
        return txt
