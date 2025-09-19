#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time    : 2025/06/03 01:58
# @Author  : Wu_RH
# @FileName: board.py

import traceback
from typing import List, Union, Tuple, Any, Generator, TYPE_CHECKING
import heapq

import gc
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import IntVar

from ....abs.rule import AbstractValue
from ....utils.impl_obj import VALUE_QUESS, MINES_TAG
from ....utils.impl_obj import POSITION_TAG, VALUE_CROSS, VALUE_CIRCLE
from ....utils.tool import get_logger
from ....abs.board import AbstractBoard, AbstractPosition, MASTER_BOARD
from ....abs.Rrule import AbstractClueValue
from ....abs.Mrule import AbstractMinesValue

if TYPE_CHECKING:
    ...


def get_value(pos=None, code=None):
    from minesweepervariants.impl.impl_obj import get_value
    return get_value(pos, code)


def alpha(n: int) -> str:
    alpha_map = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if n < 26:
        return alpha_map[n]
    return alpha_map[n // 26 - 1] + alpha_map[n % 26]


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


class Vector(AbstractPosition):
    def __init__(self, x: int, y: int):
        super().__init__(x, y, "")

    def __repr__(self):
        return f"Vector({self.y}, {self.x})"


class Position(AbstractPosition):
    def __repr__(self):
        return (f"{self.board_key+':' if self.board_key != MASTER_BOARD else ''}"
                f"{alpha(self.y)}{self.x+1}")


class Board(AbstractBoard):
    """
    通过实现
    """
    name = "Board2"
    version = 0

    def __init__(self, size: tuple = None, code: bytes = None):
        # traceback.print_stack()
        self._model = None
        self.board_data = dict()

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
            self, target: Union[str, None] = "always",
            mode: str = "object",
            key: str = None,
            *args, **kwargs
    ) -> Generator[
        Tuple[
            'Position',
            Union[
            'AbstractClueValue',
            'AbstractMinesValue',
            str, IntVar, bool, None
            ]],
        Any, None
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
            for posx in range(size[0]):
                for posy in range(size[1]):
                    pos = Position(posx, posy, key)
                    if pos in self.get_config(pos.board_key, "mask"):
                        continue
                    pos_type = self.get_type(pos)

                    # 检查是否符合目标类型
                    if target == "always" or pos_type in target:
                        if mode == "object":
                            yield pos, self.get_value(pos, *args, **kwargs)
                        elif mode == "obj":
                            yield pos, self.get_value(pos, *args, **kwargs)
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

    def has(self, target: str, key: str = None) -> bool:
        if key not in self.get_board_keys() + [None]:
            return False
        for pos, type_obj in self(mode="type", key=key):
            if type_obj == target:
                return True
        return False

    def get_model(self):
        if self._model is None:
            self._model = cp_model.CpModel()
            for _key in self.board_data:
                _size = self.board_data[_key]["config"]["size"]
                if "variable" in self.board_data[_key]:
                    del self.board_data[_key]["variable"]
                self.board_data[_key]["variable"] = \
                    [[self._model.NewBoolVar(f"var({self.get_pos(x, y, _key)})")
                      for y in range(_size[1])]
                     for x in range(_size[0])]
                get_logger().trace(f"构建新变量:{self.board_data[_key]['variable']}")
        return self._model

    def generate_board(
            self, board_key: str,
            size: tuple = (),
            labels=None,
            code: bytes = None,
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
        if labels is None:
            labels = []
        if board_key in self.board_data:
            return
        flag_byte = 0
        mask = 0
        if code is not None:
            config, mask, ture_code, false_code, labels_code, *code\
                = code.split(b"\xff", 5)
            code = b''.join(code)
            *size, flag_byte = config
            size = (size[0], size[1])
            mask = decode_bytes_7bit(mask)
            if len(labels_code) == 1 and labels_code[0] == 0:
                labels = []
            else:
                labels = labels_code.decode("ascii").split(";")
            true_tag = get_value(pos=POSITION_TAG, code=ture_code)
            false_tag = get_value(pos=POSITION_TAG, code=false_code)
        data = dict()
        self.board_data[board_key] = data
        self.labels = labels
        # 配置初始化
        data["config"] = {
            "size": size,
            "VALUE": true_tag,
            "MINES": false_tag,
            "mask": [],
            "labels": labels,
        }
        for key in self.CONFIG_FLAGS:
            data["config"].update({key: False})

        data["obj"] = [[None for _ in range(size[0])] for _ in range(size[1])]
        data["type"] = [["N" for _ in range(size[0])] for _ in range(size[1])]
        data["dye"] = [[False for _ in range(size[0])] for _ in range(size[1])]

        if code is None:
            return
        positions = [pos for pos, _ in self(key=board_key, mode="none")]
        for pos in positions[::-1]:
            if mask & 1:
                self.set_mask(pos)
            mask >>= 1

        for i, key in enumerate(self.CONFIG_FLAGS[::-1]):
            data["config"].update({key: bool(flag_byte & (1 << i))})

        codes = code.split(b"\xff")

        code_queue = []

        for part in codes:
            if not part:
                continue
            if part[0] == 0:
                count = int(part[1])
                code_queue.extend([b'_'] * count)
            else:
                code_queue.append(part)

        for pos, _ in self(key=board_key):
            code = code_queue.pop(0)
            if code[0] == 35:
                self.set_dyed(pos, True)
                code = code[1:]
            if code == b'_':
                continue
            value = get_value(pos, code)
            if value is not None:
                self.set_value(pos, value)
                continue
            raise ValueError(f"unknown type{code}")

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
                flags = (flags << 1) | int(self.board_data[board_key]["config"].get(name, False))
            for posx in range(size[0]):
                for posy in range(size[1]):
                    pos = self.get_pos(posx, posy, board_key)
                    mask <<= 1
                    if pos is None:
                        mask |= 1
            board_bytes.extend(board_key.encode("ascii") + b"\xff")
            board_bytes.extend(bytes([size[0], size[1], flags, 255]))
            board_bytes.extend(encode_int_7bit(mask) + bytes([255]))
            board_bytes.extend(value.type() + b"|" + value.code())
            board_bytes.extend(bytes([255]))
            board_bytes.extend(mines.type() + b"|" + mines.code())
            board_bytes.extend(bytes([255]))
            board_bytes.extend((";".join(label for label in labels) if len(labels) > 0 else "\x00").encode("ascii"))
            # key | sizex | sizey | config
            for pos, obj in self(key=board_key):
                board_bytes.extend(b"\xff")
                if self.get_dyed(pos):
                    board_bytes.extend(b"#")
                if obj is None:
                    board_bytes.extend(b"_")
                else:
                    code = obj.code()
                    if b"\xff" in code:
                        get_logger().error(f"{obj.type().decode()}中编码出现\\xff")
                        raise ValueError(r"code contains forbidden byte: \xff")
                    board_bytes.extend(obj.type() + b"|" + code)
            board_bytes.extend(b"\xff\xff")
        # 只用split(b"\xff_")切分
        parts = board_bytes.split(b"\xff_")

        # 处理连续 \xff_ 的次数
        encoded_bytes = bytearray()
        i = 0
        while i < len(parts):
            if i > 0:
                # 统计连续 \xff_ 的次数
                count = 1
                # 看后续parts里是否以空字节开头来判断是否连续（split后的空串）
                # 但这里由于只分割 \xff_，连续情况只能靠检查下一个part是否空
                while i + count < len(parts) and parts[i + count - 1] == b'':
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
        return encoded_bytes[:-2]

    def boundary(self, key=MASTER_BOARD) -> "Position":
        if key not in self.get_board_keys():
            return Position(-1, -1, key)
        size = self.board_data[key]["config"]["size"]
        return Position(size[0] - 1, size[1] - 1, key)

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

    def get_type(self, pos: 'Position') -> str:

        key = pos.board_key

        if self.is_valid(pos):
            return self.board_data[key]["type"][pos.y][pos.x]

        return ""

    def get_value(self, pos: 'Position', *args, **kwargs) -> Union['AbstractClueValue', 'AbstractMinesValue', None]:
        key = pos.board_key
        if self.is_valid(pos):
            return self.board_data[key]["obj"][pos.y][pos.x]
        return None

    def set_value(self, pos: 'Position', value):
        key = pos.board_key
        if self.is_valid(pos):
            self.board_data[key]["obj"][pos.y][pos.x] = value
            self.board_data[key]["type"][pos.y][pos.x] = self.type_value(value)

    def get_dyed(self, pos: 'Position') -> bool | None:
        key = pos.board_key
        if self.is_valid(pos):
            return self.board_data[key]["dye"][pos.y][pos.x]

    def set_dyed(self, pos: 'Position', dyed: bool):
        key = pos.board_key
        if self.is_valid(pos):
            self.board_data[key]["dye"][pos.y][pos.x] = dyed

    def get_variable(self, pos: 'Position') -> IntVar | None:
        key = pos.board_key
        self.get_model()
        if self.is_valid(pos):
            return self.board_data[key]["variable"][pos.x][pos.y]
        return None

    def clear_variable(self):
        for key in self.board_data.keys():
            if "variable" in self.board_data[key]:
                del self.board_data[key]["variable"]
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
            _pos = self.left(_pos)
            if not _pos.in_bounds(bound):
                break
            pos_list.append(_pos)
        _pos = pos.clone()
        pos_list = pos_list[::-1]
        while True:
            _pos = self.right(_pos)
            if not _pos.in_bounds(bound):
                break
            pos_list.append(_pos)
        return pos_list

    def get_col_pos(self, pos: 'Position') -> List["Position"]:
        bound = self.boundary(pos.board_key)
        _pos = pos.clone()
        pos_list = [_pos]
        while True:
            _pos = self.up(_pos)
            if not _pos.in_bounds(bound):
                break
            pos_list.append(_pos)
        _pos = pos.clone()
        pos_list = pos_list[::-1]
        while True:
            _pos = self.down(_pos)
            if not _pos.in_bounds(bound):
                break
            pos_list.append(_pos)
        return pos_list

    def get_pos(self, x: int, y: int, key=MASTER_BOARD) -> Union['Position', None]:
        size = self.board_data[key]["config"]["size"]
        if -size[0] < x < size[0] and -size[1] < y < size[1]:
            x = x if x >= 0 else size[0] + x
            y = y if y >= 0 else size[1] + y
            pos = Position(x, y, key)
            if self.is_valid(pos):
                return pos
        return None

    def up(self, pos: 'Position', n: int = 1):
        pos.x -= n
        return pos

    def down(self, pos: 'Position', n: int = 1):
        pos.x += n
        return pos

    def left(self, pos: 'Position', n: int = 1):
        pos.y -= n
        return pos

    def right(self, pos: 'Position', n: int = 1):
        pos.y += n
        return pos

    def deviation(self, pos: 'Position', diff_pos: 'Position'):
        pos = pos.clone()
        pos.x += diff_pos.x
        pos.y += diff_pos.y
        return pos

    def difference(self, pos1: 'AbstractPosition', pos2: 'AbstractPosition') -> 'AbstractPosition':
        vector = Vector(0, 0)
        vector.x = pos1.x - pos2.x
        vector.y = pos1.y - pos2.y
        return vector

    def neighbors(self, root_pos, *args: int) -> list['Position']:
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

        x0, y0 = root_pos.x, root_pos.y
        directions = [(dx, dy) for dx in (-1, 0, 1)
                      for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]

        heap = []  # 最小堆存储 (距离平方, x, y)
        visited = {(x0, y0)}
        result = []

        # 处理包含自身的情况 (距离平方=0)
        if low <= 0 <= high:
            result.append(root_pos.clone())

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
                result.append(Position(x, y, root_pos.board_key))

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

    def get_pos_box(self, pos1: "AbstractPosition", pos2: "AbstractPosition") -> List["AbstractPosition"]:
        if pos1.board_key != pos2.board_key:
            return []
        if not (self.in_bounds(pos1) and self.in_bounds(pos2)):
            return []
        x_min, x_max = sorted([pos1.x, pos2.x])
        y_min, y_max = sorted([pos1.y, pos2.y])

        result = []
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                result.append(self.get_pos(x, y, key=pos1.board_key))
        return result

    def batch(self, positions: List['Position'],
              mode: str, drop_none: bool = True, *args, **kwargs) -> List[Any]:
        result = []
        for pos in positions:
            if drop_none and not self.in_bounds(pos):
                continue
            if mode == "object":
                result.append(self.get_value(pos, *args, **kwargs))
            elif mode == "obj":
                result.append(self.get_value(pos, *args, **kwargs))
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

    def clear_board(self):
        for key in self.board_data:
            data = self.board_data[key]
            size = data["config"]["size"]
            data["obj"] = [[None for _ in range(size[0])] for _ in range(size[1])]
            data["type"] = [["N" for _ in range(size[0])] for _ in range(size[1])]
            self.clear_variable()

    def get_board_keys(self) -> list[str]:
        return list(self.board_data.keys())

    def show_board(self, show_tag: bool = False):
        r = ""
        for key in self.board_data:
            size = self.board_data[key]["config"]["size"]
            if len(self.board_data.keys()) > 1:
                r += key + "\n"
            for i in range(size[0]):
                for j in range(size[1]):
                    pos = self.get_pos(i, j, key)
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

    def pos_label(self, pos: 'AbstractPosition') -> str:
        labels = self.get_config(pos.board_key, "labels")
        txt = chr(64 + pos.y // 26) if pos.y > 25 else ''
        txt += chr(pos.y % 26 + 65)
        txt += '='
        txt += labels[pos.x] if pos.x < len(labels) else str(pos.x)
        return txt
