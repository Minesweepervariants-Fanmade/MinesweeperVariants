

from dataclasses import dataclass
import heapq
from typing import Optional, Self
from warnings import deprecated

__all__ = ["Position", "PositionTag", "alpha"]

def alpha(col: int) -> str:
    alpha_map = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if col < 26:
        return alpha_map[col]
    return alpha_map[col // 26 - 1] + alpha_map[col % 26]

@dataclass(order=True)
class Position:
    row: int
    col: int

    board_key: str

    def __init__(self, col: int, row: int, board_key: str) -> None:
        self.col = col
        self.row = row
        self.board_key = board_key

    @property
    @deprecated("x is deprecated, use row instead")
    def x(self) -> int:
        return self.row

    @x.setter
    def x(self, value: int) -> None:
        self.row = value

    @property
    @deprecated("y is deprecated, use col instead")
    def y(self) -> int:
        return self.col

    @y.setter
    def y(self, value: int) -> None:
        self.col = value

    def __eq__(self, other: object) -> bool:
        return (
                isinstance(other, self.__class__) and
                self.col == other.col and
                self.row == other.row and
                self.board_key == other.board_key
        )

    def __hash__(self) -> int:
        return hash((self.col, self.row, self.board_key))

    def clone(self) -> Self:
        """
        复制并返回一个相同的位置
        :return: 另一个相同的位置对象
        """
        return self.__class__(self.col, self.row, self.board_key)

    def deviation(self, pos: Self) -> Self:
        """
        对于输入位置进行偏移
        :param pos: 相对量
        :return: 偏移完成后的另外一个值
        """
        _pos = self.clone()
        _pos._deviation(pos)
        return _pos

    def up(self, n: int = 1) -> Self:
        """
        返回一个向上n格的位置对象
        :param n: 向上n格
        :return: 结果位置
        """
        _pos = self.clone()
        _pos._up(n)
        return _pos

    def down(self, n: int = 1) -> Self:
        """
        返回一个向下n格的位置对象
        :param n: 向上n格
        :return: 结果位置
        """
        _pos = self.clone()
        _pos._down(n)
        return _pos

    def left(self, n: int = 1) -> Self:
        """
        返回一个向左n格的位置对象
        :param n: 向上n格
        :return: 结果位置
        """
        _pos = self.clone()
        _pos._left(n)
        return _pos

    def right(self, n: int = 1) -> Self:
        """
        返回一个向右n格的位置对象
        :param n: 向上n格
        :return: 结果位置
        """
        _pos = self.clone()
        _pos._right(n)
        return _pos

    def shift(self, col: int = 0, row: int = 0) -> Self:
        return self.up(col).right(row)

    def __repr__(self):
        from minesweepervariants.board import MASTER_BOARD_KEY
        return (f"{self.board_key+':' if self.board_key != MASTER_BOARD_KEY else ''}"
                f"{alpha(self.col)}{self.row+1}")

    def _up(self, n: int = 1):
        self.row -= n

    def _down(self, n: int = 1):
        self.row += n

    def _left(self, n: int = 1):
        self.col -= n

    def _right(self, n: int = 1):
        self.col += n

    def _deviation(self, pos: Self):
        self.row += pos.row
        self.col += pos.col

    def in_bounds(self, bound_pos: Self) -> bool:
        if bound_pos.board_key != self.board_key:
            return False
        return (0 <= self.col <= bound_pos.col and
                0 <= self.row <= bound_pos.row)

    def neighbors(self, *args: int) -> list[Self]:
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

        x0, y0 = self.col, self.row
        directions = [(dx, dy) for dx in (-1, 0, 1)
                      for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]

        heap: list[tuple[int, int, int]] = []  # 最小堆存储 (距离平方, x, y)
        visited = {(x0, y0)}
        result: list[Self] = []

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
                result.append(self.__class__(x, y, self.board_key))

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

    @classmethod
    def parse(cls, s: str, board_key: Optional[str] = None) -> Optional[Self]:
        from minesweepervariants.board import MASTER_BOARD_KEY
        try:
            if ':' in s:
                bk, pos_str = s.split(':', 1)

                if bk != MASTER_BOARD_KEY:
                    board_key = bk
            else:
                pos_str = s

            col_str = ''.join(filter(str.isalpha, pos_str)).upper()
            row_str = ''.join(filter(str.isdigit, pos_str))

            col = 0
            for char in col_str:
                col = col * 26 + (ord(char) - ord('A') + 1)
            col -= 1

            row = int(row_str) - 1

            return cls(col, row, board_key or MASTER_BOARD_KEY)
        except Exception:
            return None


class PositionTag(Position):
    def __init__(self) -> None:
        from minesweepervariants.board import MASTER_BOARD_KEY
        super().__init__(0, 0, MASTER_BOARD_KEY)

    def neighbors(self, *args: int) -> list[Position]:
        return []

    def in_bounds(self, bound_pos: Self) -> bool:
        return False

    def _deviation(self, pos: Self) -> None:
        pass

    def _up(self, n: int = 1) -> None:
        pass

    def _down(self, n: int = 1) -> None:
        pass

    def _left(self, n: int = 1) -> None:
        pass

    def _right(self, n: int = 1) -> None:
        pass
