from typing import Protocol

type Serializeable = bool | None | int | float | str | tuple[Serializeable, ...] | set[Serializeable] | list[Serializeable] | dict[str, Serializeable]

class DumpableObject(Protocol):
    def dump(self) -> 'Serializeable':
        ...

Dumpable = DumpableObject | Serializeable

def dump(obj: Dumpable, _visited: set[int] | None = None) -> Serializeable:
    if _visited is None:
        _visited = set()

    obj_id = id(obj)
    if obj_id in _visited:
        raise ValueError(f"Circular reference detected for object {obj}")

    if obj is None:
        return None
    elif isinstance(obj, bool | int | float | str):
        return obj
    elif isinstance(obj, tuple | set | list):
        _visited.add(obj_id)
        try:
            return [dump(i, _visited) for i in obj]
        finally:
            _visited.discard(obj_id)
    elif isinstance(obj, dict):
        _visited.add(obj_id)
        try:
            return {str(k): dump(v, _visited) for k, v in obj.items()}
        finally:
            _visited.discard(obj_id)
    elif hasattr(obj, "dump") and callable(getattr(obj, "dump")):
        _visited.add(obj_id)
        try:
            obj_dump = getattr(obj, "dump")
            result = obj_dump()
            return dump(result, _visited)
        finally:
            _visited.discard(obj_id)
    else:
        raise ValueError(f"Not serializable: {obj}")