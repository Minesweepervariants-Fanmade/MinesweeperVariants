from collections.abc import MutableSet
from typing import Callable, Generator, Iterable, Optional, Self, TypeIs
from .position import Position

__all__ = ["PositionSet"]

def _Some[T](value: Optional[T]) -> TypeIs[T]:
    return value is not None

def _Must[T](value: Optional[T]) -> T:
    if value is None:
        raise ValueError("Expected a value, but got None")
    return value

class PositionSet(MutableSet[Position]):
    _positions: set[Position]

    def __init__(self, positions: Optional[Iterable[Position]] = None):
        self._positions = set(positions) if _Some(positions) else set()

    @classmethod
    def parse(cls, data: list[str]) -> Self:
        return cls(_Must(Position.parse(s)) for s in data)


    def map[T](self, func: Callable[[Position], T]) -> Generator[T, None, None]:
        yield from map(func, self._positions)


    def filter(self, func: Callable[[Position], bool]) -> Self:
        return self.__class__(filter(func, self._positions))

    def in_bounds(self, bound_pos: Position) -> Self:
        return self.filter(lambda pos: pos.in_bounds(bound_pos))

    def clone(self) -> Self:
        return self.__class__(self._positions)


    def deviation(self, pos: Position) -> Self:
        return self.__class__(p.deviation(pos) for p in self._positions)

    def up(self, n: int = 1) -> Self:
        return self.__class__(p.up(n) for p in self._positions)

    def down(self, n: int = 1) -> Self:
        return self.__class__(p.down(n) for p in self._positions)

    def left(self, n: int = 1) -> Self:
        return self.__class__(p.left(n) for p in self._positions)

    def right(self, n: int = 1) -> Self:
        return self.__class__(p.right(n) for p in self._positions)

    def shift(self, col: int = 0, row: int = 0) -> Self:
        return self.deviation(Position(col, row))

    def neighbors(self, *args: int) -> Self:
        result = self.__class__()
        for pos in self._positions:
            result._positions.update(pos.neighbors(*args))
        return result

    def to_board(self, key: str | None):
        for pos in self._positions:
            pos.to_board(key)


    def add(self, pos: Position) -> None:
        self._positions.add(pos)

    def discard(self, value: Position, /) -> None:
        return self._positions.discard(value)

    def __contains__(self, x: object, /) -> bool:
        return self._positions.__contains__(x)

    def __iter__(self):
        return iter(self._positions)

    def __len__(self):
        return len(self._positions)

    def __repr__(self):
        return str(self._positions)