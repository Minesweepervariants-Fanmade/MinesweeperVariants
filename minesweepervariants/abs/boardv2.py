from abc import ABC
from typing import Generic, Self, TypeVar
T = TypeVar("T", bound="AbstractBoard", covariant=True)

class AbstractPosition(Generic[T], ABC):
    class NeibhorAccessor:
        def __init__(self, pos: 'AbstractPosition[T]'):
            self.pos = pos
            self.visited = {pos}
            self.map = {0: {pos}}

        def __getitem__(self, distance: 'int | slice[int | None]' = 1) -> set['AbstractPosition[T]']:
            if isinstance(distance, int):
                max_visited = max(self.map.keys(), default=0)
                if distance <= max_visited:
                    return self.map[distance]
                else:
                    current = self.map[max_visited]
                    next_set: set[AbstractPosition] = set()
                    for pos in current:
                        next_set.update(self.pos.board.graph.get(pos, []))
                    next_set -= self.visited
                    self.visited.update(next_set)
                    current = next_set
                    if not current:
                        return set()
                    self.map[max_visited + 1] = current
                    return self.__getitem__(distance)
            elif isinstance(distance, slice):
                start = distance.start or 1
                stop = distance.stop or 1000
                step = distance.step or 1

                result: set[AbstractPosition] = set()
                for d in range(start, stop, step):
                    positions = self.__getitem__(d)
                    if not positions:
                        break
                    result.update(positions)
                return result

    def __init__(self, board: T, label="unknown"):
        self.label = label
        self.board = board
        self.neighbor = self.NeibhorAccessor(self)

    def is_near(self, other: Self):
        """self -> other"""
        return self.board.is_neighbor(self, other)

    def is_nearby(self, other: Self):
        """other -> self"""
        return self.board.is_neighbor(other, self)

    def __str__(self) -> str:
        return f"{self.board}[{self.label}]"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.board}, {self.label})"

class AbstractBoard[T: AbstractPosition['AbstractBoard']](ABC):
    Position: type[T]

    positions: set[T]
    # type PosLabel_T = int
    # positions_map: dict[PosLabel_T, T]
    graph: dict[T, list[T]]

    def __init__(self, label="unknown"):
        self.label = label
        self.positions = set()
        self.graph = {}
        self._nei_cache = {}

    def create_position(self, label="unknown") -> T:
        pos = self.Position(self, label)
        self.positions.add(pos)
        return pos

    def add_edge(self, pos1: T, pos2: T):
        if pos1 not in self.positions:
            raise ValueError(f"{pos1}不在{self}中")
        if pos2 not in self.positions:
            raise ValueError(f"{pos2}不在{self}中")
        self.graph.setdefault(pos1, []).append(pos2)

    def is_neighbor(self, posA: T, posB: T):
        """A -> B"""
        return posB in self.graph.get(posA, [])

    def at(self, *args, **kwargs) -> T | None:
        raise NotImplementedError

    def __str__(self) -> str:
        return f"{self.label}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.label})"

class Position(AbstractPosition['Board']):
    pass

class Board(AbstractBoard['Position']):
    Position = Position