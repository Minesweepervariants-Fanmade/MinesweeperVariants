#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2025/06/03 18:38
# @Author  : Wu_RH
# @FileName: Lrule.py

from __future__ import annotations
from ortools.sat.python.cp_model import CpModel


from typing import TYPE_CHECKING, Optional, Protocol, TypeGuard

from ..utils.tool import get_logger
from .rule import AbstractRule

if TYPE_CHECKING:
    from minesweepervariants.board import Board
    from minesweepervariants.impl.summon.solver import Switch


class SoftFn(Protocol):
    def __call__(self, value: float, priority: int) -> None: ...

def _is_str_int_dict(value: object) -> TypeGuard[dict[str, int]]:
    if not isinstance(value, dict):
        return False
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, int):
            return False
    return True


def _is_str_list(value: object) -> TypeGuard[list[str]]:
    if not isinstance(value, list):
        return False
    for item in value:
        if not isinstance(item, str):
            return False
    return True


def _is_soft_fn(value: object) -> TypeGuard[SoftFn]:
    return callable(value)

class AbstractMinesRule(AbstractRule):
    """
    雷布局规则
    """


# --------实例类-------- #


class MinesRules:
    """
    雷布局规则组
    """
    def __init__(self, rules: list['AbstractMinesRule'] | None = None):
        """
        雷布局规则组初始化
        :param rules:
        """
        self.rules = [] if rules is None else rules
        self.logger = get_logger()

    def append(self, rule: 'AbstractMinesRule') -> None:
        """
        将规则加入组
        :param rule:
        :return:
        """
        self.rules.append(rule)


class Rule0R(AbstractMinesRule):
    """
    总雷数规则
    """
    id = "0R"
    name = "0R"
    doc = "Total number of mines is given"
    author = ("", 0)
    tags = ["Untagged"]
    creation_time = ""

    total: int

    def __init__(self, board: "Board | None" = None, data: str | None = None) -> None:
        super().__init__(board, data)
        if data is None:
            raise ValueError("Data for Rule0R cannot be None")
        self.total = int(data)

    def create_constraints(self, board: 'Board', switch: "Switch") -> None:
        model: CpModel = board.get_model()
        model_obj: object = model
        s = switch.get(model, self)
        if self.total == -2:
            return
        all_variable = [board.get_variable(pos, special='raw') for pos, _ in board()]
        constraint = model_obj.add(sum(all_variable) == self.total)
        constraint.OnlyEnforceIf(s)

    def suggest_total(self, info: dict[str, object]) -> None:
        ub: int = 0
        totals_obj = info["total"]
        interactive_obj = info["interactive"]
        soft_fn_obj = info["soft_fn"]
        if not _is_str_int_dict(totals_obj) or not _is_str_list(interactive_obj):
            return
        if not _is_soft_fn(soft_fn_obj):
            return
        totals: dict[str, int] = totals_obj
        for key in interactive_obj:
            ub += totals[key]
        soft_fn_obj(ub * 0.4, -1)
