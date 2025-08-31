#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2025/06/03 01:58
# @Author  : Wu_RH
# @FileName: board.py

from typing import List, Union, Tuple, Any, Generator, TYPE_CHECKING
import heapq

from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import IntVar

from ....utils.impl_obj import VALUE_QUESS, MINES_TAG
from ....utils.tool import get_logger
from ....abs.board import AbstractBoard, AbstractPosition
from ....abs.board import MASTER_BOARD
from ....abs.Rrule import AbstractClueValue
from ....abs.Mrule import AbstractMinesValue

if TYPE_CHECKING:
    ...


def get_value(pos=None, code=None):
    from minesweepervariants.impl.impl_obj import get_value
    return get_value(pos, code)


class Position(AbstractPosition):

    def __repr__(self):
        if self.y > 14:
            return ""
        return f"{'ABCDEFGHIJKLMNO'[self.y]}{self.x + 1}"

    def _up(self, n: int = 1):
        self.x -= 1

    def _down(self, n: int = 1):
        self.x += 1

    def _left(self, n: int = 1):
        self.y -= 1

    def _right(self, n: int = 1):
        self.y += 1

    def _deviation(self, pos: 'AbstractPosition'):
        self.x += pos.x
        self.y += pos.y

    def in_bounds(self, bound_pos: 'AbstractPosition') -> bool:
        return (0 <= self.x <= bound_pos.x and
                0 <= self.y <= bound_pos.y)

    def neighbors(self, *args: int) -> list['Position']:
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

        x0, y0 = self.x, self.y
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


class Board(AbstractBoard):
    name = "Board0"
    version = 0

    __board: list[list[AbstractClueValue | AbstractClueValue | None]]
    __type_board: list[list[str]]
    __dye_board: list[list[bool]]
    variables: list[list[IntVar]] | None = None
    config = {}

    def __init__(self, size: tuple[int, int] = (), code: bytes = None):
        self.model = None
        if code is not None:
            self.size = int((code.split(b"\xff", 1)[0]).decode())
        else:
            self.size = size[0]
        self.__board = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.__type_board = [["N" for _ in range(self.size)] for _ in range(self.size)]
        self.__dye_board = [[False for _ in range(self.size)] for _ in range(self.size)]
        self.config.update({key: False for key in self.CONFIG_FLAGS})
        self.config["row_col"] = True
        self.config["VALUE"] = VALUE_QUESS
        self.config["MINES"] = MINES_TAG
        self.config["interactive"] = True
        if code is None:
            return
        codes = code.split(b"\xff")[1:]
        for pos, _ in self():
            code = codes.pop(0)
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

    def __call__(
            self, target: Union[str, None] = "always",
            mode: str = "object",
            key: str = MASTER_BOARD
    ) -> Generator[
        Tuple[
            'AbstractPosition',
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
            key = MASTER_BOARD
        if key != MASTER_BOARD:
            return
        for posx in range(self.size):
            for posy in range(self.size):
                pos = Position(posx, posy, key)
                pos_type = self.get_type(pos)

                # 检查是否符合目标类型
                if target == "always" or pos_type in target:
                    if mode in "object":
                        yield pos, self.get_value(pos)
                    elif mode == "type":
                        yield pos, pos_type
                    elif mode in "variable":
                        yield pos, self.get_variable(pos)
                    elif mode == "dye":
                        yield pos, self.get_dyed(pos)

    def get_model(self):
        if self.model is None:
            self.model = cp_model.CpModel()
        return self.model

    def has(self, target: str, key=None) -> bool:
        for line in self.__type_board:
            for type_obj in line:
                if type_obj in target:
                    return True
        return False

    def generate_board(self, board_key: str, size: tuple = (), labels: list[str] = [], code: bytes = None) -> None:
        get_logger().error("请使用其他版本的题板 该题板不支持副板")
        raise ValueError("ERROR BOARD")

    def boundary(self, key=MASTER_BOARD):
        if key != MASTER_BOARD:
            return None
        return Position(len(self.__board) - 1, len(self.__board[0]) - 1, MASTER_BOARD)

    def encode(self) -> bytes:
        """
        字节头: 尺寸
        无需换行符 初始化自动排序
        '_'表示None
        :return: 字节码
        """
        board_bytes = bytearray()
        board_bytes.extend(f"{self.size}".encode())
        for pos, obj in self():
            board_bytes.extend(b"\xff")
            if self.get_dyed(pos):
                board_bytes.extend(b"#")
            if obj is None:
                board_bytes.extend(b"_")
            else:
                board_bytes.extend(obj.type() + b"|" + obj.code())
        return board_bytes

    def get_type(self, pos: 'AbstractPosition') -> str:
        if 0 <= pos.x < self.size and 0 <= pos.y < self.size:
            return self.__type_board[pos.x][pos.y]
        return ""

    @staticmethod
    def type_value(value) -> str:
        # 查看value的类型
        if isinstance(value, AbstractMinesValue):
            return "F"
        if isinstance(value, AbstractClueValue):
            return "C"
        return "N"

    def get_value(self, pos: 'AbstractPosition') -> Union['AbstractClueValue', 'AbstractMinesValue', None]:
        if 0 <= pos.x < self.size and 0 <= pos.y < self.size:
            return self.__board[pos.x][pos.y]
        return None

    def set_value(self, pos: 'AbstractPosition', value):
        if 0 <= pos.x < self.size and 0 <= pos.y < self.size:
            self.__board[pos.x][pos.y] = value
            self.__type_board[pos.x][pos.y] = self.type_value(value)

    def get_dyed(self, pos: 'AbstractPosition') -> bool | None:
        if 0 <= pos.x < self.size and 0 <= pos.y < self.size:
            return self.__dye_board[pos.x][pos.y]

    def set_dyed(self, pos: 'AbstractPosition', dyed: bool):
        if 0 <= pos.x < self.size and 0 <= pos.y < self.size:
            self.__dye_board[pos.x][pos.y] = dyed

    def get_variable(self, pos: 'AbstractPosition') -> IntVar | None:
        if self.model is None:
            model = self.get_model()
            self.variables = \
                [[model.NewBoolVar(f"{x}.{y}")
                  for y in range(self.size)]
                 for x in range(self.size)]
        if 0 <= pos.x < self.size and 0 <= pos.y < self.size:
            return self.variables[pos.x][pos.y]

    def clear_variable(self):
        if self.variables is None:
            return
        for index_a in range(self.size - 1, -1, -1):
            for index_b in range(self.size - 1, -1, -1):
                del self.variables[index_a][index_b]
            del self.variables[index_a]
        self.variables = None
        self.model = None

    def get_row_pos(self, pos: 'AbstractPosition') -> List["AbstractPosition"]:
        _pos = pos.clone()
        pos_list = [_pos]
        while True:
            _pos = _pos.left()
            if not self.in_bounds(_pos):
                break
            pos_list.append(_pos)
        _pos = pos.clone()
        pos_list = pos_list[::-1]
        while True:
            _pos = _pos.right()
            if not self.in_bounds(_pos):
                break
            pos_list.append(_pos)
        return pos_list

    def get_col_pos(self, pos: 'AbstractPosition') -> List["AbstractPosition"]:
        _pos = pos.clone()
        pos_list = [_pos]
        while True:
            _pos = _pos.up()
            if not self.in_bounds(_pos):
                break
            pos_list.append(_pos)
        _pos = pos.clone()
        pos_list = pos_list[::-1]
        while True:
            _pos = _pos.down()
            if not self.in_bounds(_pos):
                break
            pos_list.append(_pos)
        return pos_list

    def get_pos(self, x: int, y: int, key=MASTER_BOARD) -> Union['AbstractPosition', None]:
        if -self.size < x < self.size and -self.size < y < self.size:
            x = x if x >= 0 else self.size + x
            y = y if y >= 0 else self.size + y
            return Position(x, y, MASTER_BOARD)
        return None

    def get_pos_box(self, pos1: "AbstractPosition", pos2: "AbstractPosition") -> List["AbstractPosition"]:
        if not (self.in_bounds(pos1) and self.in_bounds(pos2)):
            return []
        x_min, x_max = sorted([pos1.x, pos2.x])
        y_min, y_max = sorted([pos1.y, pos2.y])

        result = []
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                result.append(self.get_pos(x, y))
        return result

    def batch(self, positions: List['AbstractPosition'],
              mode: str, drop_none: bool = False) -> List[Any]:
        result = []
        for pos in positions:
            if drop_none and not self.in_bounds(pos):
                continue
            if mode == "object":
                result.append(self.get_value(pos))
            elif mode == "variable":
                result.append(self.get_variable(pos))
            elif mode == "type":
                result.append(self.get_type(pos))
            elif mode == "dye":
                result.append(self.get_dyed(pos))
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        return result

    def clear_board(self):
        self.__board = [[None for _ in range(self.size)] for _ in range(self.size)]
        self.__type_board = [["N" for _ in range(self.size)] for _ in range(self.size)]
        self.variables = None

    def show_board(self, show_tag: bool = False):
        r = ""
        for i in range(self.size):
            for j in range(self.size):
                value = self.__board[i][j]
                if value is None:
                    r += "_____" if show_tag else "___"
                else:
                    r += str(value) + ("_" + value.type().decode() if show_tag else "")
                r += "\t"
            r += "\n"
        return r

    def set_config(self, board_key: str, config_name: str, value: bool):
        if board_key != MASTER_BOARD:
            return
        self.config[config_name] = value

    def get_config(self, board_key: str, config_name: str):
        if board_key != MASTER_BOARD:
            return
        return self.config[config_name]

    def get_board_keys(self) -> list[str]:
        return [MASTER_BOARD]
    
    def pos_label(self, pos: 'AbstractPosition'):
        get_logger().error("请使用其他版本的题板 该题板不支持副板")
        raise ValueError("ERROR BOARD")
