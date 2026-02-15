#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2025/06/02 23:52
# @Author  : Wu_RH
# @FileName: board.py

from abc import ABC, abstractmethod
from typing import List, Tuple, Union, TYPE_CHECKING, Generator, Any
from dataclasses import dataclass

from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import IntVar

from ..impl.board.dye import get_dye

if TYPE_CHECKING:
    from minesweepervariants.abs.Rrule import AbstractClueValue
    from minesweepervariants.abs.Mrule import AbstractMinesValue

MASTER_BOARD = "1"


@dataclass(order=True)
class AbstractPosition(ABC):
    x: int
    y: int
    board_key: str

    def __init__(self, x: int, y: int, board_key: str):
        self.x = x
        self.y = y
        self.board_key = board_key

    def __eq__(self, other):
        return (
            isinstance(other, self.__class__) and
            self.x == other.x and
            self.y == other.y and
            self.board_key == other.board_key
        )

    def __hash__(self):
        return hash((self.x, self.y, self.board_key))

    def __repr__(self):
        return f"([{self.board_key}]{self.x}, {self.y})"

    def clone(self):
        """
        复制并返回一个相同的位置
        :return: 另一个相同的位置对象
        """
        return self.__class__(self.x, self.y, self.board_key)

    @abstractmethod
    def _up(self, n: int = 1):
        """
        将自己向上移动n格
        :param n: 向上n格
        """

    @abstractmethod
    def _down(self, n: int = 1):
        """
        将自己向下移动n格
        :param n: 向下n格
        """

    @abstractmethod
    def _left(self, n: int = 1):
        """
        将自己向左移动n格
        :param n: 向左n格
        """

    @abstractmethod
    def _right(self, n: int = 1):
        """
        将自己向右移动n格
        :param n: 向右n格
        """

    @abstractmethod
    def _deviation(self, pos: 'AbstractPosition'):
        """
        对于输入位置进行偏移并赋值给自身
        :param pos: 相对量
        :return:
        """

    @abstractmethod
    def _north(self, n: int = 1):
        """
        将自己向北(六边形方向)移动n格
        :param n: 向北n格
        """

    @abstractmethod
    def _east(self, n: int = 1):
        """
        将自己向东(六边形方向)移动n格
        :param n: 向东n格
        """

    @abstractmethod
    def _west(self, n: int = 1):
        """
        将自己向西(六边形方向)移动n格
        :param n: 向西n格
        """

    @abstractmethod
    def _south(self, n: int = 1):
        """
        将自己向南(六边形方向)移动n格
        :param n: 向南n格
        """

    @abstractmethod
    def neighbors(self, *args: int) -> list['AbstractPosition']:
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

    @abstractmethod
    def in_bounds(self, bound_pos: 'AbstractPosition') -> bool:
        """
        判断是否在该表示范围边界的点的范围内
        :param bound_pos:边界点
        :return: True 在边界内 False 不在边界内
        """

    def deviation(self, pos: 'AbstractPosition') -> 'AbstractPosition':
        """
        对于输入位置进行偏移
        :param pos: 相对量
        :return: 偏移完成后的另外一个值
        """
        _pos = self.clone()
        _pos._deviation(pos)
        return _pos

    def up(self, n: int = 1) -> 'AbstractPosition':
        """
        返回一个向上n格的位置对象
        :param n: 向上n格
        :return: 结果位置
        """
        _pos = self.clone()
        _pos._up(n)
        return _pos

    def down(self, n: int = 1) -> 'AbstractPosition':
        """
        返回一个向下n格的位置对象
        :param n: 向上n格
        :return: 结果位置
        """
        _pos = self.clone()
        _pos._down(n)
        return _pos

    def left(self, n: int = 1) -> 'AbstractPosition':
        """
        返回一个向左n格的位置对象
        :param n: 向上n格
        :return: 结果位置
        """
        _pos = self.clone()
        _pos._left(n)
        return _pos

    def right(self, n: int = 1) -> 'AbstractPosition':
        """
        返回一个向右n格的位置对象
        :param n: 向上n格
        :return: 结果位置
        """
        _pos = self.clone()
        _pos._right(n)
        return _pos

    def shift(self, x: int = 0, y: int = 0):
        return self.up(x).right(y)

    def north(self, n: int = 1) -> 'AbstractPosition':
        """
        返回一个向北(六边形方向)移动n格的位置对象
        :param n: 向北n格
        :return: 结果位置
        """
        _pos = self.clone()
        _pos._north(n)
        return _pos

    def east(self, n: int = 1) -> 'AbstractPosition':
        """
        返回一个向东(六边形方向)移动n格的位置对象
        :param n: 向东n格
        :return: 结果位置
        """
        _pos = self.clone()
        _pos._east(n)
        return _pos

    def west(self, n: int = 1) -> 'AbstractPosition':
        """
        返回一个向西(六边形方向)移动n格的位置对象
        :param n: 向西n格
        :return: 结果位置
        """
        _pos = self.clone()
        _pos._west(n)
        return _pos

    def south(self, n: int = 1) -> 'AbstractPosition':
        """
        返回一个向南(六边形方向)移动n格的位置对象
        :param n: 向南n格
        :return: 结果位置
        """
        _pos = self.clone()
        _pos._south(n)
        return _pos

    @abstractmethod
    def hex_neighbors(self, *args: int) -> list['AbstractPosition']:
        """
        按照六边形网格距离从小到大逐层扩散，筛选范围由层数控制（不包含当前位置）。

        调用方式（类似 range）：
            hex_neighbors(end_layer)
                返回距离 ≤ end_layer 的位置（从第 1 层开始）。
            hex_neighbors(start_layer, end_layer)
                返回距离 ∈ [start_layer, end_layer] 的位置。

        :param args: 一个或两个整数
            - 若提供一个参数 end_layer，视为从 1 到 end_layer。
            - 若提供两个参数 start_layer 和 end_layer，视为从 start_layer 到 end_layer。
            - 参数非法（数量不为 1 或 2，或值非法）时返回空列表。

        :return: 位置列表，按距离从近到远排序。
        """


class AbstractBoard(ABC):
    version = -1
    name = ""

    default_special = 'raw'

    rules: dict = None

    # 设置选项名列表
    CONFIG_FLAGS: list[str] = [
        "by_mini",      # 题板是否附带类角标
        "pos_label",    # 题板是否有X=N标志
        "row_col",      # 题板是否启用行列号
        "interactive"   # 允许在该题板上放置雷和删除线索
    ]

    @abstractmethod
    def __init__(self, *, rules, size, code, default_special):
        """
        :param size: 题板尺寸
        :param code: 题板代码
        """
        ...

    def __repr__(self):
        return self.show_board()

    @abstractmethod
    def __call__(
            self, target: Union[str, None] = "always",
            mode: str = "object",
            key: str | None = MASTER_BOARD,
            *args, **kwargs
    ) -> Generator[Tuple['AbstractPosition', Any], Any, None]:
        """
        被调用时循环返回目标值
        :param target: 遍历目标类型 可选参数: C:线索, F:雷, N:未定义|未翻开
        :param mode: 选择返回类型   可选参数: object/obj: 实例对象, type: 对象的类型, variable/var: cp_model变量, dye: 染色情况
        :param key: 选择哪块题板    默认使用主题版 如果传入None则遍历全部题板
        :return: 位置坐标与对应的值
        """

    def __getitem__(self, pos):
        return self.get_value(pos)

    def __setitem__(self, pos, value):
        return self.set_value(pos, value)

    def __contains__(self, item):
        return self.has(target=item, key=None)

    def __eq__(self, other: 'AbstractBoard'):
        if not isinstance(other, AbstractBoard):
            return False
        if self.get_board_keys() != other.get_board_keys():
            return False
        if self.get_interactive_keys() != other.get_interactive_keys():
            return False
        for key in self.get_board_keys():
            for pos, obj1 in self(key=key, special='raw'):
                obj2 = other[pos]
                if obj1 is None and obj2 is None:
                    continue
                if obj1 is None or obj2 is None:
                    return False
                if obj1.code() != obj2.code():
                    return False
                if obj1.type() != obj2.type():
                    return False
        return True

    def dyed(self, name: str):
        get_dye(name).dye(self)

    def has(self, target: str, key: str = None):
        """
        判断指定题板中是否包含目标字符串对应的元素
        target: 指定目标类型字符串
        key: 题板标识 指定在哪个题板内搜索
        """

    def clone(self) -> 'AbstractBoard':
        """
        克隆一个题板对象
        实际为编码后初始化生成
        :return: 克隆后的对象
        """
        new_board = self.__class__(code=self.encode(), default_special=self.default_special, rules=self.rules)
        if hasattr(self, "_get_rule_instance"):
            new_board._bound_get_rule_instance(self._get_rule_instance)

        return new_board

    def get_model(self) -> cp_model.CpModel:
        """获取cp_model"""

    def get_board_keys(self) -> list[str]:
        """返回当前所有题板的名称"""

    def get_interactive_keys(self) -> list[str]:
        """返回所有与主板同等级的题板索引"""
        return [k for k in self.get_board_keys()
                if self.get_config(k, "interactive")]

    def _bound_get_rule_instance(self, get_rule_instance):
        """
        绑定get_rule_instance方法
        :param get_rule_instance: 方法
        """
        self._get_rule_instance = get_rule_instance
        self.get_rule_instance = self._get_rule_instance.__get__(self)

    def get_rule_instance(self, rule_name: str, data: str|None = None, add: bool = True) -> "AbstractRule | None":
        """
        返回指定名称的规则对象
        :param rule_name: 规则名称
        :return: 规则对象
        :add: 如果没有则添加
        """
        raise RuntimeError("Method get_rule_instance is not bound")

    @abstractmethod
    def generate_board(self, board_key: str, size: tuple = (), labels: list[str] = [], code: bytes = None) -> None:
        """
        创建一块副板 board_key为名称 size为尺寸 labels 为 X=N 的 N 可能取值
        """

    @abstractmethod
    def encode(self) -> bytes:
        """
        编码成字节代码
        可在初始化时导入
        :return: 编码后的字节数据
        """

    @abstractmethod
    def boundary(self, key=MASTER_BOARD) -> 'AbstractPosition':
        """
        返回选择题板的边界极限位置
        :return: 极限位置对象
        """

    def is_valid(self, pos: 'AbstractPosition') -> bool:
        """
        检测对象是否在borad的范围内
        :param pos: 输入位置
        :return: 是否在范围内
        """
        return pos.in_bounds(self.boundary(pos.board_key))

    def in_bounds(self, pos: 'AbstractPosition') -> bool:
        """
        检测对象是否在borad的范围内
        :param pos: 输入位置
        :return: 是否在范围内
        """
        return self.is_valid(pos)

    def set_mask(self, pos):
        """
        挖去题板的指定位置
        """

    @staticmethod
    @abstractmethod
    def type_value(value) -> str:
        """
        对象的类型
        返回 F:雷, C:线索, N:未赋值
        :param value: 对象值
        :return: 类型字符串
        """

    @abstractmethod
    def register_type_special(self, name: str, func):
        """
        注册一个类型特殊处理函数
        :param name: 特殊名称
        :param func: 函数
        """

    @abstractmethod
    def get_type(self, pos: 'AbstractPosition', special: str='') -> str:
        """
        位置的类型
        返回 F:雷, C:线索, N:未赋值
        若未翻开则返回N, 题板外则返回空字符串
        :param pos: 位置
        :return: 位置类型字符串
        """

    def used_type(self) -> bool:
        """
        返回在此之前的过程中是否使用过get_type()接口
        调用该接口后状态将会重置
        """

    @abstractmethod
    def get_value(self, pos: 'AbstractPosition')\
            -> Union['AbstractClueValue', 'AbstractMinesValue', None]:
        """
        获取位置里的对象
        若在题板外则返回None
        :param pos: 位置
        :return: 位置上的对象或None
        """

    @abstractmethod
    def set_value(self, pos: 'AbstractPosition', value):
        """
        将位置设置为指定对象
        :param pos: 位置
        :param value: 设置的对象值
        """

    @abstractmethod
    def clear_board(self):
        """
        清空所有的数据
        """

    @abstractmethod
    def set_dyed(self, pos: 'AbstractPosition', dyed: bool):
        """
        设置位置为指定染色
        :param pos: 位置
        :param dyed: 是否染色
        """

    @abstractmethod
    def get_dyed(self, pos: 'AbstractPosition') -> bool:
        """
        返回某个格子是否被染色
        :param pos: 位置
        :return: 是否染色
        """

    @abstractmethod
    def get_config(self, board_key: str, config_name: str):
        """
        返回某个题板的设置
        """

    @abstractmethod
    def set_config(self, board_key: str, config_name: str, value: any):
        """
        设置某个题板的设置
        """

    def set_default_special(self, special: str = 'raw'):
        """ 设置默认变量类型(只能设置一次)"""
        if special == 'raw' or self.default_special == 'raw':
            self.default_special = special
        else:
            raise ValueError("default_special was already set")

    @abstractmethod
    def get_variable(self, pos: 'AbstractPosition', special: str = '') -> IntVar:
        """
        返回指定坐标的布尔变量
        :param pos: 位置
        :return: 变量
        """

    @abstractmethod
    def clear_variable(self):
        """
        清空当前题板的所有变量 将其设为None
        """

    @abstractmethod
    def get_row_pos(self, pos: 'AbstractPosition') -> List["AbstractPosition"]:
        """
        返回输入坐标值的该行的所有坐标对象并打包为列表
        :param pos: 输入位置
        :return: 该行的所有坐标对象
        """

    @abstractmethod
    def get_col_pos(self, pos: 'AbstractPosition') -> List["AbstractPosition"]:
        """
        返回输入坐标值的该的所有坐标对象并打包为列表
        :param pos: 输入位置
        :return: 该列的所有坐标对象
        """

    @abstractmethod
    def get_pos(self, x, y, key=MASTER_BOARD) -> 'AbstractPosition':
        """
        返回位置实体
        创建时需要遵守board实现的位置规则
        :return: 位置
        """

    @abstractmethod
    def get_pos_box(self, pos1: "AbstractPosition", pos2: "AbstractPosition") -> List["AbstractPosition"]:
        """
        使用输入的两个坐标作为对角点来生成一个矩形
        随后返回矩形框内的所有位置对象的列表
        对角点顺序不限
        :param pos1: 对角点1
        :param pos2: 对角点2
        :return: 矩形框内的所有位置
        """

    @abstractmethod
    def batch(self, positions: List['AbstractPosition'], mode: str, drop_none: bool = False, *args, **kwargs) -> List[Any]:
        """
        批量获取指定位置上的信息。
        :param positions: 位置列表
        :param mode: 模式字符串，表示要获取的类型:
            - "object": 返回原始对象
            - "type": 返回位置的类型
            - "variable"/"var": 返回 OR-Tools 中与该位置关联的变量
            - "dye": 返回染色情况
        :param drop_none:
            返回时是否丢弃none
        :return:
            与 positions 一一对应的列表，包含所请求的对象
        """

    @abstractmethod
    def show_board(self, show_tag: bool = False):
        """
        展示可视化调整的界面，如可选展示线索类型
        :param show_tag: 是否展示标签
        """

    @abstractmethod
    def pos_label(self, pos: 'AbstractPosition') -> str:
        """
        返回位置的标签
        :param pos: 位置
        :return: 标签字符串
        """

    def serialize(self):
        from ..impl.impl_obj import encode_board
        return encode_board(self.encode())

    @classmethod
    def from_str(cls, data: str):
        from ..impl.impl_obj import decode_board
        return decode_board(data)



# --------实例类-------- #


class PositionTag(AbstractPosition):
    def __init__(self):
        super().__init__(0, 0, MASTER_BOARD)

    def neighbors(self, *args: int) -> list['AbstractPosition']:
        return []

    def in_bounds(self, bound_pos: 'AbstractPosition') -> bool:
        return False

    def _deviation(self, pos: 'AbstractPosition'):
        pass

    def _up(self, n: int = 1):
        pass

    def _down(self, n: int = 1):
        pass

    def _left(self, n: int = 1):
        pass

    def _right(self, n: int = 1):
        pass

    def _north(self, n: int = 1):
        pass

    def _east(self, n: int = 1):
        pass

    def _west(self, n: int = 1):
        pass

    def _south(self, n: int = 1):
        pass

    def hex_neighbors(self, *args: int) -> list['AbstractPosition']:
        return []
