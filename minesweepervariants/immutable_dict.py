
from typing import Iterator, Mapping, TypeVar, overload

__all__ = ["ImmutableDict"]

K = TypeVar("K")
V = TypeVar("V", covariant=True)

class ImmutableDict(Mapping[K, V]):
    _data: dict[K, V]
    @overload
    def __init__(self, _mapping: Mapping[K, V]) -> None: ...

    @overload
    def __init__(self, *args: object, **kwargs: object) -> None: ...

    def __init__(self, *args: object, **kwargs: object) -> None:
        self._data = dict[K, V](*args, **kwargs)

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return str(self._data)

    def get_data(self) -> dict[K, V]:
        return self._data