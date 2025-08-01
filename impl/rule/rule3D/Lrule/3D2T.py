#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/06/15 11:59
# @Author  : Wu_RH
# @FileName: 3D2T.py

"""
[3D2T]无三连: (1)雷不能在横竖上下向构成三连, (2)非雷不能在横竖上下向构成三连
"""

from .. import Abstract3DMinesRule
from abs.board import AbstractBoard
from utils.solver import get_model


class Rule2T(Abstract3DMinesRule):
    name = ["3D2T", "无三连"]
    doc = "(1)雷不能在横竖上下向构成三连, (2)非雷不能在横竖上下向构成三连"
    subrules = [
        [True, "[3D2T]雷无三连"],
        [True, "[3D2T]非雷无三连"]
    ]

    def create_constraints(self, board: 'AbstractBoard'):
        model = get_model()

        for pos, _ in board():
            for positions in [
                [pos, pos.left(), pos.left().left()],
                [pos, pos.down(), pos.down().down()],
                [pos, self.up(board, pos), self.up(board, pos, 2)],
            ]:
                if None in positions:
                    continue
                var_list = board.batch(positions, mode="variable")
                if True in [None is i for i in var_list]:
                    continue
                if self.subrules[1][0]:
                    model.Add(sum(var_list) != 0)
                if self.subrules[0][0]:
                    model.Add(sum(var_list) != 3)

    def check(self, board: 'AbstractBoard') -> bool:
        pass

    @classmethod
    def method_choose(cls) -> int:
        return 1

    def suggest_total(self, info: dict):
        ub = 0
        for key in info["interactive"]:
            size = info["size"][key]
            ub += size[0] * size[1]
        info["soft_fn"](ub * 0.5, 0)


