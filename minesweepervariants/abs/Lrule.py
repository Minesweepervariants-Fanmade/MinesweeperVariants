#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2025/06/03 18:38
# @Author  : Wu_RH
# @FileName: Lrule.py

from typing import TYPE_CHECKING

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
    subrules = [[True, "R"]]
    """
    总雷数规则
    """
    def create_constraints(self, board: 'AbstractBoard', switch):
        if not self.subrules[0][0]:
            return
        model = board.get_model()
        all_variable = [board.get_variable(pos, special='raw') for pos, _ in board()]
        model.Add(sum(all_variable) == get_total()).OnlyEnforceIf(switch.get(model, self))
        get_logger().trace(f"[R]: model add {all_variable} == {get_total()}")

    def suggest_total(self, info: dict):
        ub = 0
        for key in info["interactive"]:
            total = info["total"][key]
            ub += total
        info["soft_fn"](ub * 0.4, -1)
