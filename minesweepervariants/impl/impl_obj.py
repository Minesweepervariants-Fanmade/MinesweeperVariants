#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time    : 2025/06/07 13:45
# @Author  : Wu_RH
# @FileName: impl_obj.py

from __future__ import annotations
from functools import cache
from minesweepervariants.board import Board


import base64
from ctypes import Union
import os
import sys
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from minesweepervariants.json_object import JSONObject
from minesweepervariants.utils.tool import get_logger

from ..utils.impl_obj import POSITION_TAG, VALUE_QUESS, MINES_TAG, decode_singleton, serialize

from ..abs.rule import AbstractValue, AbstractRule
from minesweepervariants.board import Position
from ..abs.Lrule import AbstractMinesRule
from ..abs.Mrule import AbstractMinesClueRule, AbstractMinesValue
from ..abs.Rrule import AbstractClueRule, AbstractClueValue

from .board import version3
from . import rule

if TYPE_CHECKING:
    from minesweepervariants.impl.summon.summon import Summon

TOTAL = -1
hypothesis_board = [version3]


class ModelGenerateError(Exception):
    """模型求解器错误"""


def recursive_import(module):
    if module.__file__ is None:
        return None
    base_path = Path(module.__file__).parent
    base_name = module.__name__

    for dirpath, _, filenames in os.walk(base_path):
        for f in filenames:
            if f.endswith('.py') and f != '__init__.py':
                full_path = Path(dirpath) / f
                rel = full_path.relative_to(base_path).with_suffix('')
                mod_name = base_name + '.' + '.'.join(rel.parts)

                # 如果模块已经加载，跳过
                if mod_name in sys.modules:
                    continue

                # 尝试动态导入模块
                try:
                    spec = importlib.util.spec_from_file_location(str(mod_name), str(full_path))
                    if not spec or not spec.loader:
                        continue  # 跳过无效的 spec

                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[mod_name] = mod  # 先注册到 sys.modules，避免循环导入问题
                    spec.loader.exec_module(mod)  # 执行模块代码
                except Exception as e:
                    get_logger().warning(f"Failed to import RULE script [{mod_name}]: {e}")  # 打印错误信息（可选）
                    continue  # 跳过失败的模块


def get_all_subclasses(cls):
    subclasses = set()
    direct_subs = cls.__subclasses__()
    subclasses.update(direct_subs)
    for sub in direct_subs:
        subclasses.update(get_all_subclasses(sub))
    return subclasses



def _iter_concrete_rule_classes():
    for cls in get_all_subclasses(AbstractRule):
        if cls in [
            AbstractClueRule,
            AbstractMinesClueRule,
            AbstractMinesRule
        ]:
            continue
        yield cls


def _match_alias(alias_name: str, target: str, ignore_case: bool) -> bool:
    if ignore_case:
        return alias_name.casefold() == target.casefold()
    return alias_name == target


def _resolve_rule_alias(name: str) -> str:
    current = name
    seen = set()
    while True:
        if current in seen:
            return current
        seen.add(current)

        matched_base = None
        for cls in _iter_concrete_rule_classes():
            aliases = getattr(cls, 'aliases', ()) or ()
            for ignore_case in (False, True):
                for alias in aliases:
                    alias_name = alias[0] if isinstance(alias, tuple) else alias
                    if not isinstance(alias_name, str):
                        continue
                    if not _match_alias(alias_name, current, ignore_case):
                        continue

                    if isinstance(alias, tuple) and len(alias) > 1:
                        alias_obj = alias[1]
                        base = getattr(alias_obj, 'base', None)
                        if isinstance(base, str) and base.strip():
                            matched_base = base.strip()
                        else:
                            matched_base = cls.id
                    else:
                        matched_base = cls.id
                    break
                if matched_base is not None:
                    break
            if matched_base is not None:
                break

        if matched_base is None:
            return current
        current = matched_base


@cache
def valid_rule_ids():
    ids = set()
    for cls in _iter_concrete_rule_classes():
        if hasattr(cls, 'id'):
            assert cls.id not in ids, f"规则ID冲突: {cls.id}, 请检查规则类 {cls}"
            ids.add(cls.id)
    return ids

@cache
def valid_value_ids():
    ids = set()
    for cls in get_all_subclasses(AbstractValue):
        if hasattr(cls, 'id') and not hasattr(cls, '__abstractmethods__'):
            if not isinstance(cls.id, str):
                raise ValueError(f"线索类型ID必须为字符串: {cls.id} in {cls}")
            assert cls.id not in ids, f"线索类型ID冲突: {cls.id}, 请检查线索类型类 {cls}"
            ids.add(cls.id)
    return ids

def get_rule(name: str) -> type:
    valid_rule_ids()
    rule_name = _resolve_rule_alias(name)
    all_sub_rule = [
        i for i in get_all_subclasses(AbstractRule) if i not in [
            AbstractClueRule,
            AbstractMinesClueRule,
            AbstractMinesRule
        ]
    ]

    def _show_rule_info(rule_cls: type) -> type:
        get_logger().debug(f"rule info: {rule_cls.get_info()}")
        return rule_cls

    for _name in [name, rule_name]:
        for i in all_sub_rule:
            if hasattr(i, 'id') and i.id == _name:
                return _show_rule_info(i)
        for i in all_sub_rule:
            if hasattr(i, 'id') and i.id.casefold() == _name.casefold():
                return _show_rule_info(i)
    raise ValueError(f"未找到规则[{name}]")


def get_value_type(clue_type: str) -> Optional[type[AbstractValue]]:
    valid_value_ids()
    for i in get_all_subclasses(AbstractValue):
        if (
            hasattr(i, 'id')
            and i.id == clue_type
        ):
            return i
    return None


def get_value(pos: Optional[Position], clue_type: str, data: JSONObject):
    valid_value_ids()
    singleton = decode_singleton(clue_type)
    if singleton is not None:
        return singleton

    clue_cls = get_value_type(clue_type)
    if pos is None:
        pos = POSITION_TAG
    if clue_cls is not None:
        return clue_cls.from_json(pos, data)
    return None

def decode_board(data: JSONObject, name: Optional[str] = None):
    return Board.from_json(data)


for pkg in [rule] + hypothesis_board:
    recursive_import(pkg)
