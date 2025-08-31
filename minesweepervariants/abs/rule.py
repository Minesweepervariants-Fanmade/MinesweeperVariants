#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2025/06/07 13:39
# @Author  : Wu_RH
# @FileName: rule.py

from abc import ABC, abstractmethod
from typing import List, Union, TYPE_CHECKING, Dict, Tuple, Optional

if TYPE_CHECKING:
    from minesweepervariants.abs.board import AbstractBoard, AbstractPosition
    from minesweepervariants.impl.summon.solver import Switch


class AbstractRule(ABC):
    # 规则名称
    name: Union[Tuple[str], List[str], str] = [""]

    def __init__(self, board: "AbstractBoard" = None, data=None) -> None:
        self.__data = data

    def create_constraints(self, board: 'AbstractBoard', switch: 'Switch'):
        """
        基于当前线索对象向 CP-SAT 模型添加约束。
        此方法根据当前线索的位置与规则，分析题板上的变量布局，并在模型中添加等价的逻辑约束表达式。
        所有变量必须来源于 board.get_variable(pos) 返回的变量。
        model 可以通过 board.get_model() 获取。

        :param board: 输入的题板对象
        :param switch: 接收当前规则，返回一个布尔变量，作为该线索激活开关；约束只在该变量为 True 时生效
        """

    def suggest_total(self, info: dict):
        """
        :param info:
            `info (dict)`：上下文信息字典，包含以下关键字段：
                * `size (dict[str, tuple[int, int]])` 其键为题板的字符串索引 值为size元组
                * `interactive (list[str])`：题板交互权，列表内为题板索引，所有键均为允许求解器主动交互。
                * `hard_fns (list[Callable[[CpModel, IntVar], None]])`：硬约束函数列表。
                    * 规则通过定义函数的形式添加硬约束（如调用 `model.Add(...)`），
                    * 需要将该函数追加到此列表，生成器后续会统一调用执行，确保所有硬约束生效。
                    * 函数签名应为 `(model: CpModel, total: IntVar) -> None`，不返回值。
                * `soft_fn (Callable[[int, int], None])`：软约束函数。
                    * 签名为 `(target_value: int, priority: int)`，用于表示软约束的目标值和优先级。
                    * 规则调用此函数以注册软约束，具体添加到模型的逻辑由生成器统一处理。
                    * 规则只需传入期望的目标值与优先级，无需关心底层实现和返回值。
        规则在生成阶段调用，向`info`添加硬约束，并通过调用 `info` 根键的软约束函数实现软约束。
        """

    def init_board(self, board: 'AbstractBoard'):
        """
        用于生成answer.png 需要将题板填充至无空
        """

    def init_clear(self, board: 'AbstractBoard'):
        """
        在题板生成阶段调用，用于删除题板上必须被清除的线索或对象。
        例如纸笔题目中，某些规则可能要求特定位置不能出现雷或线索。
        """

    def combine(self, rules: List[Tuple['AbstractRule', Optional[str]]]):
        """
        尝试在规则层面进行特判合并。

        当多条规则同时生效时，单独逐条建立约束可能会导致效率低下。
        本方法会接收当前所有已启用的规则，并允许具体规则实现自行检查、
        判断是否存在可以进行联合优化的情况（如剪枝、约束合并、特解处理等）。

        :param rules: 已启用的规则列表，每项为 (规则对象, 规则的参数(无参为None))。
        """

    def get_name(self):
        if type(self.name) is str:
            name = self.name[:]
        elif type(self.name) in [tuple, list]:
            name = self.name[0][:]
        else:
            name = ""
        if self.__data is None:
            return name
        return name + ":" + self.__data


class AbstractValue(ABC):
    @abstractmethod
    def __init__(self, pos: 'AbstractPosition', code: bytes = b''):
        """
        获取code并初始化 输入值为code函数的返回值
        :param code: 实例对象代码
        """
        self.pos = pos

    def __repr__(self):
        ...

    def compose(self, board: 'AbstractBoard') -> Dict:
        """
        返回一个可渲染对象列表
        默认使用__repr__
        """
        ...

    def web_component(self, board: 'AbstractBoard') -> Dict:
        """
        返回一个可渲染对象列表
        默认使用__repr__
        """
        ...

    def invalid(self, board: 'AbstractBoard'):
        if self.high_light(board) is not None:
            for _pos in self.high_light(board):
                if board.get_type(_pos) == "N":
                    return False
        else:
            return False
        return True

    @classmethod
    @abstractmethod
    def type(cls) -> bytes:
        """
        返回当前规则的类型 必须所有规则返回是不同的
        如0V返回0V
        :return:
        """
        ...

    def tag(self, board) -> bytes:
        """
        返回标签
        默认使用type
        """
        return type(self).type()

    def code(self) -> bytes:
        """
        返回为当前对象的格式化值 返回为str
        返回值会被初始化的时候使用
        返回值不可包含空格
        :return:
        """
        return b''

    def create_constraints(self, board: 'AbstractBoard', switch: 'Switch'):
        """
        基于当前线索对象向 CP-SAT 模型添加约束。
        此方法根据当前线索的位置与规则，分析题板上的变量布局，并在模型中添加等价的逻辑约束表达式。
        所有变量必须来源于 board.get_variable(pos) 返回的变量。
        model 可以通过 board.get_model() 获取。

        :param board: 输入的题板对象
        :param switch: get接收当前线索对象与位置，返回一个布尔变量，作为该线索激活开关；约束只在该变量为 True 时生效
        """
        ...

    def high_light(self, board: 'AbstractBoard') -> List['AbstractPosition'] | None:
        """
        输入一个题板 随后返回所有应该显示的高光位置(web)
        :param board: 题板
        :return: 位置列表
        """
        return None

    def deduce_cells(self, board: 'AbstractBoard') -> Union[bool, None]:
        """
        快速检查当前题板并修改可以直接得出结论的地方
        :param board: 输入题板
        :return: 是否修改了 True 修改 False 未修改  None:未实现该方法
        """
        return None
