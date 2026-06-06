from typing import TYPE_CHECKING, Literal, Mapping, Self, Sequence, TypeIs, TypedDict
from ..json_object import JSONObject, JSONDirectlySerializable, deep_wrap, JSONScalar


if TYPE_CHECKING:
    class Template(TypedDict, extra_items=JSONDirectlySerializable):
        type: Literal["value"]
        data: JSONDirectlySerializable
else:
    class Template(TypedDict):
        type: Literal["value"]
        data: JSONDirectlySerializable

        __extra_items__ = JSONDirectlySerializable

def is_value_template(data: JSONDirectlySerializable) -> TypeIs[Template]:
    return isinstance(data, dict) and data.get("type") == "value" and "data" in data

class ValueTemplate:
    def __init__(self):
        return

    def _template(self) -> Template:
        result: Template =  {
            "type": "value",
            "data": None
        }
        return result

    def json(self) -> JSONObject:
        return deep_wrap(self._template())

    @classmethod
    def try_from(cls, data: Template) -> Self | None:
        return cls()

    def __repr__(self) -> str:
        return "?"

    def compose(self) -> Mapping[str, object]:

        return _get_col(
            _get_dummy(height=0.3),
            _get_text(self.__repr__()),
            _get_dummy(height=0.3),
        )

    def web_component(self, board: 'Board') -> Mapping[str, object]:
        if "compose" in type(self).__dict__:
            return self.compose(board)
        return Number(self.__repr__())



class SingleValue(ValueTemplate):
    def __init__(self, value: JSONScalar):
        self.value = value

    def _template(self) -> Template:
        result = super()._template()
        result["_SingleValue"] = True
        result["data"] = self.value

        return result

    @classmethod
    def try_from(cls, data: Template) -> Self | None:
        if not data.get("_SingleValue", False):
            return None

        value = data["data"]
        match value:
            case str() | bool() | int() | float() | None:
                return cls(value)
            case _:
                return None

class SingleNumberValue(SingleValue):
    def __init__(self, value: int | float | tuple[int, int]):
        self.value = value

    def _template(self) -> Template:
        result = super()._template()
        result["_SingleNumberValue"] = True
        result["data"] = self.value

        return result

    @classmethod
    def try_from(cls, data: Template) -> Self | None:
        if not data.get("_SingleNumberValue", False):
            return None

        value = data["data"]

        match value:
            case int() | float():
                return cls(value)
            case (int(a), int(b)):
                return cls((a, b))
            case _:
                return None


class SingleIntValue(SingleNumberValue):
    def __init__(self, value: int):
        self.value = value

    def _template(self) -> Template:
        result = super()._template()
        result["_SingleIntValue"] = True
        result["data"] = self.value

        return result

    @classmethod
    def try_from(cls, data: Template) -> Self | None:
        if not data.get("_SingleIntValue", False):
            return None

        value = data["data"]

        match value:
            case int():
                return cls(value)
            case _:
                return None

class MultiIntValue(ValueTemplate):
    def __init__(self, value: Sequence[int]):
        self.value = tuple(value)

    def _template(self) -> Template:
        result = super()._template()
        result["_MultiIntValue"] = True
        result["data"] = self.value

        return result

    @classmethod
    def try_from(cls, data: Template) -> Self | None:
        if not data.get("_MultiIntValue", False):
            return None

        value = data["data"]

        match value:
            case tuple():
                typed_value = tuple(n for n in value if isinstance(n, int))
                return cls(typed_value)
            case _:
                return None
