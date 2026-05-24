#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# @Time    : 2025/06/07 13:39
# @Author  : Wu_RH
# @FileName: rule.py

from abc import ABC, ABCMeta, abstractmethod
import locale
from typing import Callable, Iterator, List, Literal, Mapping, NoReturn, Optional, Protocol, Tuple, TYPE_CHECKING, TypedDict, TypeGuard, Union, get_args, ItemsView
from collections.abc import MutableMapping
from minesweepervariants.utils.tool import get_logger

if TYPE_CHECKING:
    from minesweepervariants.abs.board import AbstractBoard, AbstractPosition
    from minesweepervariants.impl.summon.solver import Switch
    from minesweepervariants.abs.board import ImmutableDict, JSONObject


class RuleInfo(TypedDict):
    id: str
    name: str | dict[str, str]
    doc: str | dict[str, str]
    author: tuple[str, int | str]
    tags: list["Tag"]
    creation_time: str
    lib_only: bool

class I18n(MutableMapping[str, str]):
    """Runtime I18n mapping that preserves the original attribute-style API.

    Implements MutableMapping[str,str] so static typing treats it as a mapping
    of locale -> string, while runtime still supports `name.zh_CN = ...`,
    `str(i18n)`, iteration and attribute fallback to `default`.
    """
    # instance storage for mapping values
    _data: dict[str, str]
    default: str

    def __init__(self, default_val: str = "") -> None:
        # store values in a private dict to avoid touching __dict__ directly
        object.__setattr__(self, "_data", {})
        object.__setattr__(self, "default", default_val)

    # MutableMapping required methods
    def __getitem__(self, key: str) -> str:
        return self._data.get(key, self.default)

    def __setitem__(self, key: str, value: str) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        yield from self._data.keys()

    def __len__(self) -> int:
        return len(self._data)

    # Preserve attribute-style assignment used by rule modules
    def __setattr__(self, key: str, value: str) -> None:
        # keep real attributes (default, _data) on the instance; other keys go into mapping
        if key in ("default", "_data"):
            object.__setattr__(self, key, value)
            return
        self._data[key] = value

    def __getattr__(self, name: str) -> str:
        if name in ("default", "_data"):
            return object.__getattribute__(self, name)
        return self._data.get(name, self.default)

    # Convenience string/representation behavior kept as before
    def __str__(self) -> str:
        lang = locale.getdefaultlocale()[0]
        if lang is not None and lang in self._data:
            return self._data[lang]
        else:
            return self.default

    def __repr__(self) -> str:
        attrs = ", ".join({f"{k}={v!r}" for k, v in self._data.items() if k != 'default'})
        if self.default:
            return f"I18n(default={self.default!r}, {attrs})"
        return f"I18n({attrs})"

    # keep old items() iteration shape for callers that expect (k,v)
    def items(self) -> ItemsView[str, str]:
        return self._data.items()



class I18nAutoDict(dict[str, object]):
    from typing import overload, Literal

    @overload
    def __getitem__(self, key: Literal["name"]) -> I18n: ...

    @overload
    def __getitem__(self, key: Literal["doc"]) -> I18n: ...

    @overload
    def __getitem__(self, key: str) -> object: ...

    def __getitem__(self, key: str) -> object:
        if key in ("name", "doc"):
            if key not in self:
                self[key] = I18n()
            return super().__getitem__(key)
        return super().__getitem__(key)

    @overload
    def __setitem__(self, key: Literal["name"], value: I18n | str) -> None: ...

    @overload
    def __setitem__(self, key: Literal["doc"], value: I18n | str) -> None: ...

    @overload
    def __setitem__(self, key: str, value: object) -> None: ...

    def __setitem__(self, key: str, value: object) -> None:
        if key in ('name', 'doc'):
            # name/doc assignments must be string-like or I18n
            current = self.get(key)
            if isinstance(current, I18n):
                # prefer treating incoming value as str
                if isinstance(value, I18n):
                    current.default = value.default
                elif isinstance(value, str):
                    current.default = value
                else:
                    raise TypeError(f"{key} must be str or I18n, got {type(value).__name__}")
                return
            else:
                # wrap provided string into I18n
                if isinstance(value, I18n):
                    value = value
                elif isinstance(value, str):
                    value = I18n(value)
                else:
                    raise TypeError(f"{key} must be str or I18n, got {type(value).__name__}")
        super().__setitem__(key, value)


class I18nLike(Protocol):
    """Protocol representing the mapping-like surface used by I18n.

    We don't inherit from Mapping here to avoid Protocol/MRO issues; instead
    explicitly declare the minimal mapping methods used by callers.
    """
    default: str
    def __getitem__(self, key: str) -> str: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    def items(self) -> ItemsView[str, str]: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class I18nMeta(ABCMeta):
    @classmethod
    def __prepare__(mcs, name: str, bases: tuple[type, ...], **kwargs: object) -> dict[str, object]:
        return I18nAutoDict()
    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict[str, object], **kwargs: object):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        try:
            abstract_methods: frozenset[object] | None = getattr(cls, '__abstractmethods__', None)
            if isinstance(abstract_methods, frozenset):
                abstract: set[str] = set()
                for item in abstract_methods:
                    if isinstance(item, str):
                        abstract.add(item)
                for k in ('name', 'doc'):
                    if k in abstract:
                        abstract.discard(k)
                cls.__abstractmethods__ = frozenset(abstract)
        except Exception:
            pass
        return cls

_Tag = Literal['Original', 'Variant', 'Creative', 'Global', 'Local',
                   'Strict R', 'Strict Shape', 'Strong', 'Weak',
                   'Anti-Construction', 'Connectivity', 'Construction',
                   'Extensive Trial', 'Cryptic', 'Mine-Counting', 'Mine-Value',
                   'Mine-Position', 'Dyed', 'Fun', 'Number Clue',
                   'Arrow Clue', 'Multi-Board', 'Aux Board',
                   'Vanilla Variant', 'Light', 'Heavy',
                   'WIP', 'Parameter', 'Meta', 'Untagged']
type Tag = _Tag
VALID_TAGS: set[Tag] = set(get_args(_Tag))
class AbstractRule(ABC, metaclass=I18nMeta):
    # 规则名称
    id: str
    # `name`/`doc` can be a plain string or an I18n mapping
    name: I18n | str
    doc: I18n | str
    author: tuple[str, int]
    tags: list[Tag]
    creation_time: str

    lib_only = False

    def __init__(self, board: "AbstractBoard | None" = None, data: str | None = None) -> None:
        self.__data = data

    @classmethod
    def get_info(cls) -> RuleInfo:
        """
        返回当前规则类的基础元信息。
        """

        def _normalize_i18n(value: I18n | str) -> str | dict[str, str]:
            if isinstance(value, I18n):
                data: dict[str, str] = {}
                for key in value:
                    if key == "default":
                        continue
                    item = value[key]
                    if item:
                        data[key] = item
                if value.default:
                    data.setdefault("default", value.default)
                return data or value.default
            return value

        def _warn(field_name: str, expected: str, value: object) -> NoReturn:
            logger = get_logger()
            if hasattr(logger, "warning"):
                logger.warning(
                    f"{cls.__name__}.get_info: field '{field_name}' expected {expected}, got {type(value).__name__}: {value!r}"
                )
            raise TypeError(f"{cls.__name__}.get_info: field '{field_name}' expected {expected}, got {type(value).__name__}: {value!r}")

        def _is_str(value: object) -> TypeGuard[str]:
            return isinstance(value, str)

        def _is_valid_author(value: object) -> TypeGuard[tuple[str, int | str]]:
            if not isinstance(value, tuple):
                return False
            return (
                len(value) == 2
                and isinstance(value[0], str)
                and isinstance(value[1], (int, str))
            )

        def _is_valid_tags(value: object) -> TypeGuard[list[Tag]]:
            if not isinstance(value, list):
                return False
            return all((item in VALID_TAGS) for item in value)

        def _is_bool(value: object) -> TypeGuard[bool]:
            return isinstance(value, bool)

        def _check[T](field_name: str, value: T, predicate: Callable[[T], bool], expected: str) -> T:
            if not predicate(value):
                _warn(field_name, expected, value)
            return value

        info: RuleInfo = {
            "id": _check("id", cls.id, _is_str, "str"),
            "name": _normalize_i18n(cls.name),
            "doc": _normalize_i18n(cls.doc),
            "author": _check(
                "author",
                cls.author,
                _is_valid_author,
                "tuple[str, int | str]",
            ),
            "tags": _check(
                "tags",
                cls.tags,
                _is_valid_tags,
                f"list[{' | '.join(sorted(VALID_TAGS))}]",
            ),
            "creation_time": _check(
                "creation_time",
                cls.creation_time,
                _is_str,
                "str",
            ),
            "lib_only": _check(
                "lib_only",
                cls.lib_only,
                _is_bool,
                "bool",
            ),
        }

        return info

    def create_constraints(self, board: 'AbstractBoard', switch: 'Switch') -> None:
        """
        基于当前线索对象向 CP-SAT 模型添加约束。
        此方法根据当前线索的位置与规则，分析题板上的变量布局，并在模型中添加等价的逻辑约束表达式。
        所有变量必须来源于 board.get_variable(pos) 返回的变量。
        model 可以通过 board.get_model() 获取。

        :param board: 输入的题板对象
        :param switch: 接收当前规则，返回一个布尔变量，作为该线索激活开关；约束只在该变量为 True 时生效
        """

    def suggest_total(self, info: dict[str, object]) -> None:
        """
        :param info:
            `info (dict)`：上下文信息字典，包含以下关键字段：
                * `size (dict[str, tuple[int, int]])` 其键为题板的字符串索引 值为size元组
                * `total (dict[str, int])`其键为题板的字符串索引 值为该题板的所有格子的数量
                * `interactive (list[str])`：题板交互权，列表内为题板索引，所有键均为允许求解器主动交互。
                * `hard_fns (list[Callable[[CpModel, IntVar], None]])`：硬约束函数列表。
                    * 规则通过定义函数的形式添加硬约束（如调用 `model.Add(...)`）
                    * 需要将该函数追加到此列表，生成器后续会统一调用执行，确保所有硬约束生效。
                    * 函数签名应为 `(model: CpModel, total: IntVar) -> None`，不返回值。
                * `soft_fn (Callable[[int, int], None])`：软约束函数。
                    * 签名为 `(target_value: int, priority: int)`，用于表示软约束的目标值和优先级。
                    * 规则调用此函数以注册软约束，具体添加到模型的逻辑由生成器统一处理。
                    * 规则只需传入期望的目标值与优先级，无需关心底层实现和返回值。
        规则在生成阶段调用，向`info`添加硬约束，并通过调用 `info` 根键的软约束函数实现软约束。
        """

    def init_board(self, board: 'AbstractBoard') -> None:
        """
        用于生成answer.png 需要将题板填充至无空
        """

    def init_clear(self, board: 'AbstractBoard') -> None:
        """
        在题板生成阶段调用，用于删除题板上必须被清除的线索或对象。
        例如纸笔题目中，某些规则可能要求特定位置不能出现雷或线索。
        """

    def combine(self, rules: List[Tuple['AbstractRule', Optional[str]]]) -> None:
        """
        尝试在规则层面进行特判合并。

        当多条规则同时生效时，单独逐条建立约束可能会导致效率低下。
        本方法会接收当前所有已启用的规则，并允许具体规则实现自行检查、
        判断是否存在可以进行联合优化的情况（如剪枝、约束合并、特解处理等）。

        :param rules: 已启用的规则列表，每项为 (规则对象, 规则的参数(无参为None))。
        """

    def get_name(self) -> str:
        if self.__data is None:
            return self.id
        return f"{self.id}:{self.__data}"

    def onboard_init(self, board: 'AbstractBoard') -> None:
        """
        题板初始化时调用
        :param board:
        :return:
        """

    def get_deps(self) -> List[str]:
        """
        返回当前规则依赖的其他规则名称列表
        """
        return []

class AbstractValue(ABC):
    @classmethod
    def from_json(cls, pos: 'AbstractPosition', data: 'JSONObject') -> 'AbstractValue':
        if isinstance(data, Mapping) and 'old_style' in data and data['old_style'] and 'type' in data:
            if 'code' in data and isinstance((code := data['code']), str):
                from base64 import b64decode
                return cls(pos, **{"code": b64decode(code)})
            else:
                raise ValueError(f"Unsupported clue value type {data['type']}")
        else:
            raise ValueError(f"Unsupported clue value type")

    def json(self) -> 'JSONObject':
        from base64 import b64encode
        from minesweepervariants.abs.board import ImmutableDict
        return ImmutableDict({"old_style": True, "type": b64encode(self.type()).decode(), "code": b64encode(self.code()).decode()})

    @abstractmethod
    def __init__(self, pos: 'AbstractPosition', *args, **kwargs) -> None:
        """
        获取code并初始化 输入值为code函数的返回值
        :param code: 实例对象代码
        """
        self.pos = pos

    def __repr__(self) -> str:
        raise NotImplementedError

    def compose(self, board: 'AbstractBoard') -> Mapping[str, object]:
        """
        返回一个可渲染对象列表
        默认使用__repr__
        """
        ...

    def web_component(self, board: 'AbstractBoard') -> Mapping[str, object]:
        """
        返回一个可渲染对象列表
        默认使用__repr__
        """
        ...

    def invalid(self, board: 'AbstractBoard') -> bool:
        high_light = self.high_light(board)
        if high_light is None:
            return False
        for _pos in high_light:
            if board.get_type(_pos, special='raw') == "N":
                return False
        return True

    @classmethod
    @abstractmethod
    def type(cls) -> bytes:
        """
        返回当前规则的类型 必须所有规则返回是不同的
        如0V返回0V
        :return:
        """
        ...

    def tag(self, board: 'AbstractBoard') -> bytes:
        """
        返回标签
        默认使用type
        """
        return self.type()

    def code(self) -> bytes:
        """
        返回为当前对象的格式化值 返回为str
        返回值会被初始化的时候使用
        返回值不可包含空格
        :return:
        """
        return b''

    def create_constraints(self, board: 'AbstractBoard', switch: 'Switch') -> None:
        """
        基于当前线索对象向 CP-SAT 模型添加约束。
        此方法根据当前线索的位置与规则，分析题板上的变量布局，并在模型中添加等价的逻辑约束表达式。
        所有变量必须来源于 board.get_variable(pos) 返回的变量。
        model 可以通过 board.get_model() 获取。

        :param board: 输入的题板对象
        :param switch: get接收当前线索对象与位置，返回一个布尔变量，作为该线索激活开关；约束只在该变量为 True 时生效
        """
        ...

    def high_light(self, board: 'AbstractBoard') -> List['AbstractPosition'] | None:
        """
        输入一个题板 随后返回所有应该显示的高光位置(web)
        :param board: 题板
        :return: 位置列表
        """
        return None

    def deduce_cells(self, board: 'AbstractBoard') -> Union[bool, None]:
        """
        快速检查当前题板并修改可以直接得出结论的地方
        :param board: 输入题板
        :return: 是否修改了 True 修改 False 未修改  None:未实现该方法
        """
        return None

    def weaker(self, board: 'AbstractBoard') -> 'AbstractValue':
        """
        返回一个比当前对象更弱的对象 用于生成阶段的多次删除线索
        """
        return self

    def weaker_times(self) -> int:
        """
        返回可以被弱化的次数
        """
        return 0
