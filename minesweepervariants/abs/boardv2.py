from abc import ABC
from typing import Generic, Self, TypeVar, overload
from ..utils.dump import Serializeable, dump

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
        self.variables= {}

    def is_near(self, other: Self):
        """self -> other"""
        return self.board.is_neighbor(self, other)

    def is_nearby(self, other: Self):
        """other -> self"""
        return self.board.is_neighbor(other, self)

    def __str__(self) -> str:
        return f"{self.label}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.board}, {self.label})"

    def __hash__(self) -> int:
        return hash((self.board.label, self.label))

    def dump(self) -> Serializeable:
        return str(self)

    def get_var(self, key: str):
        return self.variables.get(key)


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

    @overload
    def at(self, label: str, /) -> T | None: ...

    @overload
    def at(self, *args, **kwargs) -> T | None: ...

    def at(self, *args, **kwargs) -> T | None:
        label = args[0] if args else kwargs.get("label")
        return next((p for p in self.positions if p.label == label), None)

    def __str__(self) -> str:
        return f"{self.label}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.label})"

    def dump(self) -> Serializeable:
        return {
            "type": self.__class__.__name__,
            "meta": {"label": self.label},
            "data": {"positions": dump(self.positions), "graph": {str(k): dump(v) for k, v in self.graph.items()}} # type: ignore
        }

    @classmethod
    def load(cls, data: Serializeable) -> Self:
        """加载 Board 对象，支持多态路由到正确的子类"""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data)}")

        type_name = data.get("type")

        # 验证和还原基础信息
        meta = data.get("meta", {})
        if not isinstance(meta, dict):
            raise ValueError(f"Expected meta to be dict, got {type(meta)}")
        label = meta.get("label", "unknown")
        if not isinstance(label, str):
            raise ValueError(f"Expected label to be str, got {type(label)}")

        # 创建对象
        board = Board(label)

        # 还原 graph
        data = data.get("data", {})
        if not isinstance(data, dict):
            raise ValueError(f"Expected data to be dict, got {type(data)}")
        positions = data.get("positions", [])

        if not isinstance(positions, list):
            raise ValueError(f"Expected positions to be list, got {type(positions)}")

        for pos_label in positions:
            if not isinstance(pos_label, str):
                raise ValueError(f"Expected position labels to be str, got {type(pos_label)}")
            board.create_position(pos_label)

        graph = data.get("graph", {})
        if not isinstance(graph, dict):
            raise ValueError(f"Expected graph to be dict, got {type(graph)}")

        for k, v in graph.items():
            if not isinstance(k, str):
                raise ValueError(f"Expected graph keys to be str, got {type(k)}")
            if not isinstance(v, list):
                raise ValueError(f"Expected graph values to be list, got {type(v)}")
            pos1 = board.at(k)
            if pos1 is None:
                raise ValueError(f"Position {k} not found in board {board}")
            for neighbor_label in v:
                if not isinstance(neighbor_label, str):
                    raise ValueError(f"Expected neighbor labels to be str, got {type(neighbor_label)}")
                pos2 = board.at(neighbor_label)
                if pos2 is None:
                    raise ValueError(f"Neighbor position {neighbor_label} not found in board {board}")
                board.add_edge(pos1, pos2)

        if type_name == cls.__name__:
            return cls.postload(board)

        # 找到对应的子类并调用其 postload
        for subclass in AbstractBoard.__subclasses__():
            if subclass.__name__ == type_name:
                return subclass.postload(board) # type: ignore

        raise ValueError(f"Unknown board type: {type_name}, expected {cls.__name__}")

    @classmethod
    def postload(cls, board: 'AbstractBoard') -> Self:
        if isinstance(board, cls):
            return board
        raise ValueError(f"Expected board of type {cls.__name__}, got {type(board).__name__}")


class Position(AbstractPosition['Board']):
    pass

class Board(AbstractBoard['Position']):
    Position = Position


if __name__ == "__main__":
    b = Board("test")
    p1 = b.create_position("A")
    p2 = b.create_position("B")
    p3 = b.create_position("C")
    b.add_edge(p1, p2)
    b.add_edge(p2, p3)
    print(p1.neighbor[1])