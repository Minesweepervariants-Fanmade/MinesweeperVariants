from dataclasses import dataclass
from typing import Self, overload
from abs.boardv2 import AbstractBoard, AbstractPosition, Board
from minesweepervariants.utils.dump import Serializeable

@dataclass(frozen=True)
class Coord:
    x: int
    y: int
    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def _at(self, x: int, y: int) -> Self | None:
        return self.__class__(x, y)

    def up(self, distance=1) -> Self | None:
        return self._at(self.x, self.y - distance)

    def down(self, distance=1) -> Self | None:
        return self._at(self.x, self.y + distance)

    def left(self, distance=1) -> Self | None:
        return self._at(self.x - distance, self.y)

    def right(self, distance=1) -> Self | None:
        return self._at(self.x + distance, self.y)

    def offset(self, coord: 'Coord') -> Self | None:
        return self._at(self.x + coord.x, self.y + coord.y)

    def __add__(self, other: 'Coord') -> Self | None:
        return self.offset(other)

    def __sub__(self, other: 'Coord') -> Self | None:
        return self.offset(Coord(-other.x, -other.y))

    def __mul__(self, factor: int) -> Self | None:
        return self._at(self.x * factor, self.y * factor)

    def __rmul__(self, factor: int) -> Self | None:
        return self.__mul__(factor)

    def __floordiv__(self, factor: int) -> Self | None:
        return self._at(self.x // factor, self.y // factor)

    def euler_distance2(self, other: 'Coord') -> int:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def euler_distance(self, other: 'Coord') -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def manhattan_distance(self, other: 'Coord') -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def chebyshev_distance(self, other: 'Coord') -> int:
        return max(abs(self.x - other.x), abs(self.y - other.y))

    def neighbors4(self) -> list[Self]:
        return [c
                for c in [self.up(), self.down(), self.left(), self.right()]
                if c is not None]

    def neighbors8(self) -> list[Self]:
        return [r
                for c in [Coord(-1, -1), Coord(-1, 0), Coord(-1, 1),
                          Coord( 0, -1),               Coord( 0, 1),
                          Coord( 1, -1), Coord( 1, 0), Coord( 1, 1)]
                if (r:=self + c) is not None]

class SquarePosition(AbstractPosition['SquareBoard'], Coord):
    def __init__(self, board: 'SquareBoard', coord: Coord):
        Coord.__init__(self, coord.x, coord.y)
        AbstractPosition.__init__(self, board, label=str(coord))

    def to_coord(self) -> Coord:
        return Coord(self.x, self.y)

    def _at(self, x: int, y: int) -> 'SquarePosition | None':
        return self.board.at(x, y)

class SquareBoard(AbstractBoard['SquarePosition']):
    Position = SquarePosition

    @overload
    def at(self, x: int, y: int, /) -> 'SquarePosition | None': ...

    @overload
    def at(self, coord: Coord, /) -> 'SquarePosition | None': ...

    def at(self, arg1, arg2 = None) -> 'SquarePosition | None':
        if arg2 is None:
            coord: Coord = arg1
            coord_x: int = coord.x
            coord_y: int = coord.y
        else:
            x: int = arg1
            y: int = arg2
            coord_x, coord_y = x, y

        if 0 <= coord_x < self.width and 0 <= coord_y < self.height:
            # positions_map 可优化
            return next((p for p in self.positions if p.x == coord_x and p.y == coord_y), None)
        return None

    def __init__(self, width: int, height: int, label="SquareBoard"):
        super().__init__(label)
        self.width = width
        self.height = height

        for x in range(width):
            for y in range(height):
                pos = self.Position(self, Coord(x, y))
                self.positions.add(pos)

        for pos in self.positions:
            x, y = pos.x, pos.y
            neighbors: list[SquarePosition] = pos.neighbors4()
            self.graph[pos] = neighbors


    @classmethod
    def postload(cls, board: AbstractBoard) -> 'SquareBoard':
        # TODO: 将AbstractBoard转化为SquareBoard，假设数据格式正确
        board.__class__ = cls
        return board  # type: ignore


if __name__ == "__main__":
    board = SquareBoard(3, 3, "")
    d = board.dump()
    l = Board.load(d)
    print(d)
    print(l.dump())