from typing import TYPE_CHECKING, Literal, Mapping, Self, Sequence, TypeIs, TypedDict, Dict

from .image_template import Element, get_col, get_dummy, get_text, get_row
from .web_template import MultiNumber
from ..json_object import JSONObject, JSONDirectlySerializable, deep_wrap, JSONScalar

if TYPE_CHECKING:
    class Template(TypedDict, extra_items=JSONDirectlySerializable):
        type: Literal["value", "minevalue"]
        data: JSONDirectlySerializable
else:
    class Template(TypedDict):
        type: Literal["value", "minevalue"]
        data: JSONDirectlySerializable

        __extra_items__ = JSONDirectlySerializable


def is_value_template(data: JSONDirectlySerializable) -> TypeIs[Template]:
    return isinstance(data, Mapping) and data.get("type") in ["value", "minevalue"] and "data" in data


class ValueTemplate:
    def __init__(self, is_mine: bool = False):
        self.is_mine = is_mine

    def _template(self) -> Template:
        result: Template = {
            "type": "minevalue" if self.is_mine else "value",
            "data": None
        }
        return result

    def json(self) -> JSONObject:
        return deep_wrap(self._template())

    @classmethod
    def try_from(cls, data: Template) -> Self | None:
        is_mine = data.get("type") == "minevalue"
        return cls(is_mine=is_mine)

    def __repr__(self) -> str:
        return "?"

    def compose(self) -> Element:
        from minesweepervariants.utils.image_template import get_col, get_text, get_dummy

        color = ("#FFFF00", "#FF7F00") if self.is_mine else ("#FFFFFF", "#000000")
        return get_col(
            get_dummy(height=0.3),
            get_text(self.__repr__(), color=color),
            get_dummy(height=0.3),
        )

    def web_component(self) -> Element:
        from minesweepervariants.utils.web_template import Number
        return Number(self.__repr__())


class SingleValue(ValueTemplate):
    def __init__(self, value: JSONScalar, is_mine: bool = False):
        super().__init__(is_mine=is_mine)
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

    def __repr__(self) -> str:
        return str(self.value)

    def compose(self) -> Element:
        from minesweepervariants.utils.image_template import get_col, get_text, get_dummy

        color = ("#FFFF00", "#FF7F00") if self.is_mine else ("#FFFFFF", "#000000")
        return get_col(
            get_dummy(height=0.3),
            get_text(self.__repr__(), color=color),
            get_dummy(height=0.3),
        )

    def web_component(self) -> Element:
        from minesweepervariants.utils.web_template import Number
        return Number(self.__repr__())


class SingleNumberValue(SingleValue):
    def __init__(self, value: int | float | tuple[int, int], is_mine: bool = False):
        super().__init__(value, is_mine=is_mine)
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
    def __init__(self, value: int, is_mine: bool = False):
        super().__init__(value, is_mine=is_mine)
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
    def __init__(self, value: Sequence[int], is_mine: bool = False):
        super().__init__(is_mine=is_mine)
        self.value = tuple(value)

    def __repr__(self) -> str:
        return '.'.join([str(i) for i in self.value])

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
            case tuple() | list():
                typed_value = tuple(n for n in value if isinstance(n, int))
                return cls(typed_value)
            case _:
                return None

    def compose(self) -> Dict:
        if len(self.value) <= 1:
            value = 0
            if len(self.value) == 1:
                value = self.value[0]
            return get_col(
                get_dummy(height=0.175),
                get_text(str(value)),
                get_dummy(height=0.175),
            )
        if len(self.value) == 2:
            text_a = get_text(str(self.value[0]))
            text_b = get_text(str(self.value[1]))
            return get_col(
                get_dummy(height=0.175),
                get_row(
                    text_a,
                    text_b
                ),
                get_dummy(height=0.175),
            )
        elif len(self.value) == 3:
            text_a = get_text(str(self.value[0]))
            text_b = get_text(str(self.value[1]))
            text_c = get_text(str(self.value[2]))
            return get_col(
                get_row(
                    text_a,
                    text_b,
                    # spacing=0
                ),
                text_c,
            )
        elif len(self.value) == 4:
            text_a = get_text(str(self.value[0]))
            text_b = get_text(str(self.value[1]))
            text_c = get_text(str(self.value[2]))
            text_d = get_text(str(self.value[3]))
            return get_col(
                get_row(
                    text_a,
                    text_b,
                ),
                get_row(
                    text_c,
                    text_d
                )
            )
        else:
            # 我也不知道为什么会出现>5个数字的情况
            return get_text("")

    def web_component(self) -> Dict:
        if not self.value:
            return MultiNumber([""])
        return MultiNumber([str(i) for i in self.value])


class SingleImageValue(ValueTemplate):
    def __init__(self, value: str, is_mine: bool = False):
        super().__init__(is_mine=is_mine)
        self.value = value

    def _template(self) -> Template:
        result = super()._template()
        result["_SingleImageValue"] = True
        result["data"] = self.value

        return result

    @classmethod
    def try_from(cls, data: Template) -> Self | None:
        if not data.get("_SingleImageValue", False):
            return None

        value = data["data"]
        match value:
            case str():
                return cls(value)
            case _:
                return None

    def __repr__(self) -> str:
        return str(self.value)

    def compose(self) -> Mapping[str, object]:
        from minesweepervariants.utils.image_template import get_image

        return get_image(
            self.value,
            cover_pos_label=False,
        )

    def web_component(self) -> Mapping[str, object]:
        from minesweepervariants.utils.image_template import get_image

        return get_image(
            self.value,
            cover_pos_label=False,
        )
