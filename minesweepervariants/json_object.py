from typing import Mapping, Protocol, Sequence, TypeIs, get_origin, overload, runtime_checkable
from minesweepervariants.immutable_dict import ImmutableDict

__all__ = ["JSONObject", "JSONString", "JSONDirectlySerializable", "SerializeAble", "JSONLikeType", "json_dumps", "json_loads", "compress", "decompress", "valid", "assert_", "get_with_valid", "jsonify"]

type JSONObject = ImmutableDict[str, JSONObject] | tuple[JSONObject, ...] | str | int | float | bool | None
type JSONString = str

type JSONDirectlySerializable = dict[str, JSONDirectlySerializable] | list[
    JSONDirectlySerializable] | str | int | float | bool | None


@runtime_checkable
class SerializeAble(Protocol):
    def from_json(self, data: JSONObject) -> None: ...

    def json(self) -> JSONObject: ...


type JSONLikeType = SerializeAble | JSONObject | Sequence[JSONLikeType] | Mapping[str, JSONLikeType]


def _deep_unwrap(obj: JSONObject) -> JSONDirectlySerializable:
    if isinstance(obj, ImmutableDict):
        return {k: _deep_unwrap(v) for k, v in obj.get_data().items()}

    if isinstance(obj, tuple):
        return [_deep_unwrap(item) for item in obj]

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj


def _deep_wrap(obj: JSONDirectlySerializable) -> JSONObject:
    if isinstance(obj, dict):
        return ImmutableDict({k: _deep_wrap(v) for k, v in obj.items()})

    if isinstance(obj, list):
        return tuple(_deep_wrap(item) for item in obj)

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj


def json_dumps(obj: JSONObject) -> JSONString:
    try:
        from orjson import dumps as orjson_dumps
        pure_data = _deep_unwrap(obj)
        return orjson_dumps(pure_data).decode('utf-8')
    except ImportError:
        from json import dumps as json_dumps_std
        pure_data = _deep_unwrap(obj)
        return json_dumps_std(pure_data, ensure_ascii=False)


def json_loads(json_str: JSONString) -> JSONObject:
    try:
        from orjson import loads as orjson_loads
        return _deep_wrap(orjson_loads(json_str))
    except ImportError:
        from json import loads as json_loads_std
        return _deep_wrap(json_loads_std(json_str))


def compress(s: str) -> str:
    from base64 import urlsafe_b64encode
    import zstandard as zstd
    b = s.encode()

    compressor = zstd.ZstdCompressor(level=22)

    return urlsafe_b64encode(compressor.compress(b)).decode()


def decompress(s: str) -> str:
    from base64 import urlsafe_b64decode
    import zstandard as zstd
    b = urlsafe_b64decode(s.encode())

    decompressor = zstd.ZstdDecompressor()

    return decompressor.decompress(b).decode()


@overload
def valid[T](data: object, type_: type[T]) -> TypeIs[T]: ...

@overload
def valid[T1, T2](data: object, type_: tuple[type[T1], type[T2]]) -> TypeIs[T1 | T2]: ...

def valid(data: object, type_: type | tuple[type, ...]) -> bool:
    if isinstance(type_, tuple):
        return any(valid(data, t) for t in type_)
    if not isinstance(data, get_origin(type_) or type_):
        return False
    return True

@overload
def assert_[T](data: object, type_: type[T]) -> T: ...

@overload
def assert_[T1, T2](data: object, type_: tuple[type[T1], type[T2]]) -> T1 | T2: ...

def assert_[T1, T2](data: object, type_: type[T1] | tuple[type[T1], type[T2]]):
    if valid(data, type_):
        return data
    if isinstance(type_, tuple):
        expected_types = ", ".join(t.__name__ for t in type_)
    else:
        expected_types = type_.__name__
    raise TypeError(f"Expected type {expected_types}, got {type(data).__name__}")


@overload
def get_with_valid[T](data: object, key: str, type_: type[T]) -> T: ...

@overload
def get_with_valid[T1, T2](data: object, key: str, type_: tuple[type[T1], type[T2]]) -> T1 | T2: ...

def get_with_valid[T1, T2](data: object, key: str, type_: type[T1] | tuple[type[T1], type[T2]]):
    data = assert_(data, ImmutableDict[str, object])
    if key not in data:
        raise KeyError(key)

    return assert_((data[key]), type_)


def jsonify(obj: JSONLikeType) -> JSONObject:
    if isinstance(obj, SerializeAble):
        return obj.json()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Sequence) and not isinstance(obj, str):
        return tuple(jsonify(i) for i in obj)
    if isinstance(obj, Mapping):
        return ImmutableDict({k: jsonify(v) for k, v in obj.items()})
