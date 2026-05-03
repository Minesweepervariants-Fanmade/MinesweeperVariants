#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2025/06/03 18:38
# @Author  : Wu_RH
# @FileName: Lrule.py

from typing import TYPE_CHECKING, Optional

from ..utils.impl_obj import get_total
from ..utils.tool import get_logger
from .rule import AbstractRule

if TYPE_CHECKING:
    from minesweepervariants.abs.board import AbstractBoard


class AbstractMinesRule(AbstractRule):
    """
    雷布局规则
    """


# --------实例类-------- #


class MinesRules:
    """
    雷布局规则组
    """
    def __init__(self, rules: list['AbstractMinesRule'] = None):
        """
        雷布局规则组初始化
        :param rules:
        """
        if rules is None:
            rules = []
        self.rules = rules
        self.logger = get_logger()

    def append(self, rule: 'AbstractMinesRule'):
        """
        将规则加入组
        :param rule:
        :return:
        """
        self.rules.append(rule)


class Rule0R(AbstractMinesRule):
    name = "0R"
    """
    总雷数规则
    """

    def __init__(self, board: "AbstractBoard" = None, data=None) -> None:
        super().__init__(board, data)
        self.data: Optional[str] = data

    def create_constraints(self, board: 'AbstractBoard', switch):
        model = board.get_model()
        s = switch.get(model, self)
        if self.data == "2" and get_total() == -1:
            return
        all_variable = [board.get_variable(pos, special='raw') for pos, _ in board()]
        model.Add(sum(all_variable) == get_total()).OnlyEnforceIf(s)
        get_logger().trace(f"[R]: model add {all_variable} == {get_total()}")

    def suggest_total(self, info: dict):
        ub = 0
        for key in info["interactive"]:
            total = info["total"][key]
            ub += total
        info["soft_fn"](ub * 0.4, -1)
