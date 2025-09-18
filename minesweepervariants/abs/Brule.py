from abc import ABC, abstractmethod

from .rule import AbstractRule
from .board import AbstractBoard


class AbstractBoardRule(AbstractRule, ABC):
    """
    题板规则
    """
    @classmethod
    @abstractmethod
    def get_board(cls) -> type['AbstractProxyBoard']:
        """
        返回题板类型
        """

    def apply(self, board: AbstractBoard) -> AbstractBoard:
        """
        应用并返回代理题板
        """
        return AbstractProxyBoard(board)

class AbstractProxyBoard(AbstractBoard, ABC):
    def __init__(self, board: AbstractBoard):
        if hasattr(board, "core"):
            self.core = board.core
        else:
            self.core = board

    def __getattr__(self, name):
        return getattr(self.core, name)

# --------实例类-------- #

class Board14MV(AbstractProxyBoard): 
    ...


class Rule14MV(AbstractBoardRule):
    @classmethod
    def get_board(cls):
        return Board14MV
