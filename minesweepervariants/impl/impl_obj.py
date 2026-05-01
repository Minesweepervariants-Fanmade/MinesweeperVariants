#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time    : 2025/06/07 13:45
# @Author  : Wu_RH
# @FileName: impl_obj.py
import base64
import os
import sys
import importlib.util
from pathlib import Path
from typing import Optional

from minesweepervariants.utils.tool import get_logger

from ..utils.impl_obj import VALUE_QUESS, MINES_TAG

from ..abs.rule import AbstractValue, AbstractRule
from ..abs.board import AbstractBoard
from ..abs.Lrule import AbstractMinesRule
from ..abs.Mrule import AbstractMinesClueRule, AbstractMinesValue
from ..abs.Rrule import AbstractClueRule, AbstractClueValue

from .board import version3
from . import rule

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


def set_total(total: int):
    global TOTAL
    TOTAL = total


def get_all_subclasses(cls):
    subclasses = set()
    direct_subs = cls.__subclasses__()
    subclasses.update(direct_subs)
    for sub in direct_subs:
        subclasses.update(get_all_subclasses(sub))
    return subclasses


def get_board(name: Optional[str] = None):
    if name is None:
        v = -1
        b = None
        for i in AbstractBoard.__subclasses__():
            if v < i.version:
                v = i.version
                b = i
        if b is None:
            raise ValueError("未找到棋盘")
        return b
    else:
        for i in AbstractBoard.__subclasses__():
            if i.name == name:
                return i


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


def get_rule(name: str) -> type:
    rule_name = _resolve_rule_alias(name)
    all_sub_rule = get_all_subclasses(AbstractRule)
    for i in all_sub_rule:
        if i in [
            AbstractClueRule,
            AbstractMinesClueRule,
            AbstractMinesRule
        ]:
            continue
        if hasattr(i, 'id') and i.id == rule_name:
            return i
    for i in all_sub_rule:
        if i in [
            AbstractClueRule,
            AbstractMinesClueRule,
            AbstractMinesRule
        ]:
            continue
        if hasattr(i, 'id') and i.id.casefold() == rule_name.casefold():
            return i
    raise ValueError(f"未找到规则[{name}]")


def get_value(pos, code):
    code = code.split(b"|", 1)
    if code[0] == b"?":
        return VALUE_QUESS
    if code[0] == b"F":
        return MINES_TAG
    for i in get_all_subclasses(AbstractValue):
        if i in [
            AbstractValue,
            AbstractClueValue,
            AbstractMinesValue
        ]:
            continue
        if i.type() == code[0]:
            return i(pos=pos, code=code[1])
    return None


def encode_board(code: bytes) -> str:
    code = code[:]
    padding = len(code) % 4
    if padding:
        code += b'\xff' * (4 - padding)
    return base64.urlsafe_b64encode(code).decode("ascii")


def decode_board(base64data: str, name: Optional[str] = None):
    board_bytes = base64.urlsafe_b64decode(base64data.encode("ascii"))
    board_bytes = board_bytes.rstrip(b"\xff")
    return get_board(name)(rules={}, code=board_bytes)


for pkg in [rule] + hypothesis_board:
    recursive_import(pkg)
