#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minesweepervariants.board import Board


class AbstractDye(ABC):
    name: str
    fullname: str
    doc: str = ""

    def __init__(self, args: str) -> None:
        self.args = args

    @abstractmethod
    def dye(self, board: "Board") -> None:
        """染色函数"""
