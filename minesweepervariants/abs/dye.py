#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minesweepervariants.abs.board import AbstractBoard


class AbstractDye(ABC):
    name: str
    fullname: str
    doc: str = ""

    def __init__(self, args: str) -> None:
        self.args = args

    @abstractmethod
    def dye(self, board: "AbstractBoard") -> None:
        """染色函数"""
