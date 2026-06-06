#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time    : 2025/06/07 13:45
# @Author  : Wu_RH
# @FileName: impl_obj.py

from __future__ import annotations
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


def add_rule(
    summon: Summon,
    board: Board,
    rule_id: str,
    data: str | None = None,
    add: bool = True
) -> AbstractRule | None:
    """
    增量式添加单个规则及其依赖到 summon.rules

    :param board: 目标题板对象
    :param rule: 单个规则ID字符串（可含 delimiter 分隔的 data）
    """

    if summon.rules is None:
        summon.rules = {
            "clue_rules": [],
            "mines_rules": [],
            "mines_clue_rules": []
        }

    # 检查是否已添加过该规则（避免重复）
    all_rules: list[AbstractRule] = (
        summon.rules.get("clue_rules", []) +
        summon.rules.get("mines_rules", []) +
        summon.rules.get("mines_clue_rules", [])
    )

    # for existing_rule in all_rules:
    #     if existing_rule.id == rule_id and existing_rule.__data == data:
    #         return existing_rule

    # # 不需要添加直接返回
    # if not add:
    #     return None

    # 实例化规则
    rule_instance: AbstractRule = get_rule(rule_id)(board=board, data=data)

    result_rule = rule_instance

    # 递归处理依赖
    rule_deps = rule_instance.get_deps()
    for dep in rule_deps:
        add_rule(summon, board, dep, data, add)

    if add:
        # 根据类型分类添加
        if isinstance(rule_instance, AbstractClueRule):
            summon.rules["clue_rules"].append(rule_instance)
        elif isinstance(rule_instance, AbstractMinesRule):
            summon.rules["mines_rules"].append(rule_instance)
        elif isinstance(rule_instance, AbstractMinesClueRule):
            summon.rules["mines_clue_rules"].append(rule_instance)
        else:
            raise ValueError(f"Unknown Rule: {rule_id}")

    # 检查是否为 lib_only 规则
    if rule_instance.lib_only:
        # 检查该规则是否被其他规则依赖
        for existing_rule in all_rules:
            if rule_id in existing_rule.get_deps():
                break
        else:  # 未被依赖
            v_rule: AbstractRule = get_rule("V''")(board=board, data=rule_id)
            if add:
                summon.rules["clue_rules"].append(v_rule)
            result_rule = v_rule

    result_rule.onboard_init(board)

    if add:
        # 更新所有规则的 combine 信息
        all_rules = (summon.rules["clue_rules"] +
                     summon.rules["mines_rules"] +
                     summon.rules["mines_clue_rules"])
        rules_info: list[tuple[AbstractRule, str | None]] = [(r, None) for r in all_rules]  # 简化版本，data 信息在规则实例中

        for r in all_rules:
            r.combine(rules_info)

    return result_rule


def get_rule(name: str) -> type:
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
    for i in get_all_subclasses(AbstractValue):
        if (
            hasattr(i, 'type')
            and i.type() is not None
            and i.type().decode('ascii') == clue_type
        ):
            return i
    return None


def get_value(pos: Optional[Position], clue_type: str, data: JSONObject):
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
