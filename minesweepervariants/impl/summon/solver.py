#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time    : 2025/06/07 19:40
# @Author  : Wu_RH
# @FileName: solver.py
import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import time
from ortools.sat.python.cp_model import IntVar

from minesweepervariants.utils.timer import timer

from ...abs.rule import AbstractRule, AbstractValue
from ortools.sat.python import cp_model
from collections import defaultdict
from typing import Dict, List, Tuple, Union, Optional

from ...utils.tool import get_logger

from ...abs.Mrule import AbstractMinesClueRule
from ...abs.Rrule import AbstractClueRule
from ...abs.board import AbstractBoard, AbstractPosition
from ...abs.Lrule import MinesRules, Rule0R
from ..impl_obj import ModelGenerateError

from minesweepervariants.config.config import DEFAULT_CONFIG

# ==== 获取默认值 ====
CONFIG = {}
CONFIG.update(DEFAULT_CONFIG)


def get_solver(b: bool):
    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 42  # 启发式多样性
    solver.parameters.num_search_workers = CONFIG["workes_number"]  # 多线程并行搜索
    solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
    solver.parameters.linearization_level = 2  # 启用线性化加速
    # solver.parameters.cp_model_presolve = b
    solver.parameters.use_optional_variables = True
    solver.parameters.randomize_search = True
    if CONFIG["timeout"] > 0 and b:
        solver.parameters.max_time_in_seconds = CONFIG["timeout"]  # 时间限制
    return solver


class Switch:

    def __init__(self):
        # 对象名称到索引计数器的映射（用于自动生成索引）
        self.name_counter: Dict[str, int] = defaultdict(lambda: 1)

        # 存储所有创建的变量
        self.all_vars: List[cp_model.IntVar] = []

        # 对象到其所有开关的映射 {obj_str: [(index_str, var), ...]}
        self.obj_switches: Dict[str, List[Tuple[str, cp_model.IntVar]]] = defaultdict(list)

        # 注册表：存储已创建的变量 {(obj_str, index_str): var}
        self.var_registry: Dict[Tuple[str, str], cp_model.IntVar] = {}

        # 反向映射：变量索引到(obj_str, index_str)的映射
        self.var_to_key: Dict[int, Tuple[str, str]] = {}

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    @staticmethod
    def to_str(obj: Union[AbstractRule, AbstractValue, AbstractPosition, str]) -> str:
        if isinstance(obj, AbstractRule):
            name = f"RULE|{obj.get_name()}"
        elif (
                isinstance(obj, AbstractPosition) or
                isinstance(obj, AbstractValue)
        ):
            if isinstance(obj, AbstractValue):
                pos = obj.pos
            else:
                pos = obj
            if pos is None:
                raise ValueError("pos cannot be None")
            pos = f"|{pos.x}|{pos.y}|{pos.board_key}"
            name = "POS" + pos
        elif isinstance(obj, str):
            name = obj
        else:
            raise ValueError("obj must be an Rule or Value")
        return name

    def get(
            self,
            model: cp_model.CpModel,
            obj: Union[AbstractRule, AbstractValue, AbstractPosition, str],
            index_str: Optional[str] = None
    ) -> cp_model.IntVar:
        """
        创建或获取一个开关变量

        参数:
        model: CpModel实例
        obj: 关联的对象（规则、位置或值）
        index_str: 可选，指定索引字符串。如果未提供则自动生成

        返回:
        布尔变量（如果是新创建或已存在的变量）
        """
        obj_str = self.to_str(obj)

        # 自动生成索引字符串（如果需要）
        if index_str is None:
            # 获取下一个索引并递增
            current_index = self.name_counter[obj_str]
            index_str = str(current_index)
            self.name_counter[obj_str] += 1

        # 检查是否已存在相同的变量
        key = (obj_str, index_str)
        if key in self.var_registry:
            return self.var_registry[key]

        # 创建新变量
        var = model.NewBoolVar(f"{obj_str}_{index_str}")

        # 存储变量
        self.var_registry[key] = var
        self.all_vars.append(var)
        self.obj_switches[obj_str].append((index_str, var))
        self.var_to_key[var.Index()] = (obj_str, index_str)

        # 尝试更新计数器（仅当索引是数字时）
        try:
            idx_val = int(index_str)
            self.name_counter[obj_str] = max(self.name_counter[obj_str], idx_val + 1)
        except ValueError:
            # 非数字索引时不更新计数器
            pass

        return var

    def get_switches_by_obj(
            self,
            obj: Union[AbstractRule, AbstractValue, AbstractPosition, str]
    ) -> List[Tuple[str, cp_model.IntVar]]:
        """
        获取指定对象的所有开关

        参数:
        obj: 查询的对象

        返回:
        该对象的所有开关列表，格式为[(index_str, var), ...]
        """
        obj_str = self.to_str(obj)
        return self.obj_switches.get(obj_str, [])

    def get_all_switches(
            self
    ) -> Dict[str, List[Tuple[str, cp_model.IntVar]]]:
        """
        获取所有对象及其开关

        返回:
        字典格式: {obj_str: [(index_str, var), ...]}
        """
        return dict(self.obj_switches)

    def get_all_vars(self) -> List[cp_model.IntVar]:
        """
        获取所有创建的开关变量

        返回:
        变量列表
        """
        return self.all_vars

    def get_obj_and_index_by_var(
            self,
            var: cp_model.IntVar
    ) -> Tuple[str, Optional[str]]:
        """
        根据开关变量获取对应的对象字符串和索引字符串

        参数:
        var: 开关变量

        返回:
        元组 (obj_str, index_str)

        如果找不到变量，抛出KeyError
        """
        var_index = var.Index()
        if var_index not in self.var_to_key:
            raise KeyError(f"Variable with index {var_index} not found in switch registry")
        key = self.var_to_key[var_index]
        if len(self.get_switches_by_obj(key[0])) == 1:
            return key[0], None
        return key


def solver_by_csp(
        mines_rules: MinesRules,
        clue_rule: Union[AbstractClueRule, None],
        mines_clue_rule: Union[AbstractMinesClueRule, None],
        board: AbstractBoard,
        drop_r=False,
        bool_mode=False,
        answer_board=None,
        model=None
) -> int:
    """
    返回int 0表示无解 1表示唯一解 2表示多解
    """
    # -*- csp推导 -*- #
    logger = get_logger()

    if model is None:
        logger.trace("求解器输入:\n" + board.show_board())
        logger.trace("构建新模型")
        board = board.clone()
        board.clear_variable()
        model = board.get_model()
        switch = Switch()

        # 2.获取所有规则约束
        for rule in (
                mines_rules.rules +
                [clue_rule, mines_clue_rule]
        ):
            if rule is None:
                continue
            if drop_r and isinstance(rule, Rule0R):
                continue
            rule: AbstractRule
            rule.create_constraints(board, switch)

        for key in board.get_board_keys():
            for pos, obj in board(key=key):
                if obj is None:
                    continue
                obj: AbstractValue
                obj.create_constraints(board, switch)

        # 3.获取所有变量并赋值已解完的部分
        for key in board.get_board_keys():
            for _, var in board.core("C", mode="variable", key=key, special='raw'):
                model.Add(var == 0)
                logger.trace(f"var: {var} == 0")
            for _, var in board("F", mode="variable", key=key, special='raw'):
                model.Add(var == 1)
                logger.trace(f"var: {var} == 1")

        model: cp_model.CpModel
        # model.AddAssumptions(switch.var_map.values())
        for switch_var in switch.get_all_vars():
            model.Add(switch_var == 1)

    # 4.获取求解器并推导
    solver = get_solver(True)

    if answer_board is not None:
        current_solution = []
        logger.trace("设置预设不同值")
        for key in board.get_interactive_keys():
            for pos, _ in board("N", key=key):
                value = 1 if answer_board.get_type(pos, special='raw') == "F" else 0
                val = board.get_variable(pos, special='raw')
                tmp = model.NewBoolVar(f"answer_tmp{pos}")
                current_solution.append(tmp)
                logger.trace(f"[{pos}]{val} != {value} ({answer_board.get_type(pos, special='raw')})")
                model.Add(val != value).OnlyEnforceIf(tmp)
        model.AddBoolOr(current_solution)  # 新增排除当前解约束（至少有一个变量取反）

        status2 = timer(solver.Solve)(model)
        if status2 == cp_model.FEASIBLE or status2 == cp_model.OPTIMAL:
            logger.trace(f"求解器多解")
            return 2
        return 1

    status = timer(solver.Solve)(model)

    # 5.检查是否无解或者多解
    if status == cp_model.INFEASIBLE:
        logger.trace("求解器无解")
        return 0

    if status == cp_model.UNKNOWN:
        logger.trace("求解器遇到未知问题")
        return 0

    if status == cp_model.MODEL_INVALID:
        # 这个bug太玄乎了 修不好直接爆了
        # 关键他重新跑一遍就正常了 我没法debug
        logger.trace("\n" + board.show_board())
        logger.trace("求解器模型构建出现错误")

        # 输出状态码名称
        logger.trace(f"Solver status: {solver.StatusName(status)}")

        # 输出求解统计信息
        logger.trace(f"Num conflicts: {solver.NumConflicts()}")
        logger.trace(f"Num branches: {solver.NumBranches()}")
        logger.trace(f"Wall time (s): {solver.WallTime()}")

        for key in board.get_board_keys():
            logger.trace(f"board key: {key}")
            logger.trace("varlist:{}".format([(var.index, var) for _, var in board(mode="variable", key=key, special='raw')]))
            logger.trace("obj:{}".format([var for _, var in board(mode="object", key=key)]))
            logger.trace("type:{}".format([var for _, var in board(mode="type", key=key, special='raw')]))
            logger.trace("dye:{}".format([var for _, var in board(mode="dye", key=key)]))

        raise ModelGenerateError("model Error")

    if logger.print_level <= logger.TRACE:
        if not os.path.exists(CONFIG["output_path"]):
            os.makedirs(CONFIG["output_path"])
        model.ExportToFile("log/model.txt")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.trace(f"求解器status:{status}")
        return 0

    if bool_mode:
        # 如果打开 那么就将状态压缩至 有解/无解
        return 1

    current_solution = []
    for key in board.get_interactive_keys():
        for pos, _ in board("N", key=key):
            var = board.get_variable(pos, special='raw')
            val = solver.Value(var)
            if val == 1:
                current_solution.append(var.Not())  # 这个变量不能为1，取反
            else:
                current_solution.append(var)  # 这个变量不能为0，保持原变量

    model.AddBoolOr(current_solution)  # 新增排除当前解约束（至少有一个变量取反）

    status2 = timer(solver.Solve)(model)
    if status2 == cp_model.FEASIBLE or status2 == cp_model.OPTIMAL:
        logger.trace(f"求解器多解")
        return 2
    return 1


def deduced_by_csp(
        board: AbstractBoard,
        answer_board: AbstractBoard,
        pos: AbstractPosition,
):
    """
    检查是否无解
    """
    if board[pos] is not None:
        return None
    model = board.get_model().clone()
    model: cp_model.CpModel

    target_var = board.get_variable(pos, special='raw')
    model.Add(target_var == (0 if answer_board.get_type(pos, special='raw') == "F" else 1))

    solver = get_solver(False)
    state = timer(solver.Solve)(model)

    if state == cp_model.INFEASIBLE:
        return True
    elif state in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return False
    else:
        return None


def hint_by_csp(
    board: AbstractBoard,
    answer_board: AbstractBoard,
    switch: Switch,
    pos: AbstractPosition,
    upper_bound=None
):
    if board[pos] is not None:
        return None
    model = board.get_model().clone()
    model: cp_model.CpModel

    target_var = board.get_variable(pos, special='raw')
    model.Add(target_var == (0 if answer_board.get_type(pos, special='raw') == "F" else 1))
    assumptions = switch.get_all_vars()

    get_logger().trace(f"pos {pos}: start\n", end="")
    results = _hint_by_csp(
        model, assumptions,
        upper_bound, 0, pos
    )
    get_logger().trace(f"pos {pos}: {results}\n", end="")
    if results is None:
        return None
    return [switch.get_obj_and_index_by_var(r) for r in results]


def _hint_by_csp(
    model: cp_model.CpModel,
    assumptions: List[IntVar],
    upper_bound=None,
    offset=0,
    pos=None
):
    logger = get_logger()
    logger.trace(f"pos {pos} off {offset}: start\n", end="")
    future_to_param = {}
    _results = []
    with ThreadPoolExecutor(max_workers=CONFIG["workes_number"]) as executor:
        for var in assumptions:
            _model = model.clone()
            _model.Add(var == 0)
            _model.AddBoolAnd([v for v in assumptions if v != var])
            solver = get_solver(True)
            fut = executor.submit(
                timer(solver.Solve),
                _model, None
            )
            future_to_param[fut] = var
        for fut in as_completed(future_to_param):
            try:
                var = future_to_param[fut]
                logger.trace(f"pos {pos} off {offset} wait: {var}")
                status = fut.result()
                logger.trace(f"pos {pos} off {offset} end wait: {status}")
                if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
                    _results.append(var)
            except Exception as e:
                logger.trace(e)
                continue

    logger.trace(f"pos {pos} off {offset} end AND [{_results}]")

    if upper_bound is not None and len(_results) - 2 > (upper_bound[0] - offset):
        logger.trace(f"pos {pos}, off {offset}: fail (len>ub)\n", end="")
        return None
    # 将与的状态保留至model
    model.AddBoolAnd(_results)
    # 将其他约束加入进行测试是否无解(可推)
    _model = model.clone()
    _model.Add(sum([v for v in assumptions if v not in _results]) == 0)
    solver = get_solver(False)
    logger.trace(f"pos {pos} off {offset} verification and start")
    status = timer(solver.Solve)(_model)
    logger.trace(f"pos {pos} off {offset} verification and end {status}")
    if status == cp_model.INFEASIBLE:
        # 无解说明已经是包含所有的约束了 是当前层的MUS
        if upper_bound is not None:
            with upper_bound[1]:
                if (len(_results) + offset) < upper_bound[0]:
                    upper_bound[0] = len(_results) + offset
        logger.trace(f"pos {pos} off {offset}: done {_results}\n", end="")
        return _results
    elif status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # 未知状况直接返回None
        logger.trace(f"pos {pos} off {offset}: error state:{status}\n", end="")
        return None
    # 有解说明当前层依旧有或节点未被发现 需要继续遍历
    # 获取当前层的MCS
    mcs = [v for v in assumptions if v not in _results]

    # 使用二分推理或关系
    R = 0
    L = len(mcs)
    mid = 2 * len(mcs) // 3

    logger.trace(f"pos {pos} off {offset}: start OR lever\n", end="")
    while True:
        _model = model.clone()

        _model.Add(sum(mcs) >= mid)

        solver = get_solver(False)
        logger.trace(f"pos {pos} off {offset}: OR => start {len(mcs)}-{L}-{mid}-{R}\n", end="")
        status = timer(solver.Solve)(_model)
        logger.trace(f"pos {pos} off {offset}: OR => end {len(mcs)}-{L}-{mid}-{R} => {status}\n", end="")
        if status == cp_model.INFEASIBLE:
            # 无解
            L = mid
            mid = (L - R) // 2 + R
        elif status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # 有解
            R = mid
            mid = (L - R) // 2 + R
            if R == mid: break
        else: return None

    _mcs = [i for i in mcs if solver.Value(i) == 0]
    logger.trace(f"pos {pos} off {offset} ORlever: {_mcs} status {status}\n", end="")
    results = []

    for var in _mcs:
        _model = model.clone()
        # 只开启其中一个或节点进行广搜
        _model.Add(var == 1)
        _model.AddBoolAnd([v.Not() for v in _mcs if v != var])
        _assumptions = [v for v in assumptions if v not in _mcs + _results]

        result = _hint_by_csp(
            _model,
            _assumptions[:],
            upper_bound=upper_bound,
            offset=offset + len(_results) + len(_mcs),
            pos=pos
        )
        if result is None:
            # 该节点求解失败或者出现错误或者提前步出
            continue
        results.append(result + [var] + _results)

    if not results:
        logger.trace(f"pos {pos} off {offset}: fail (OR)\n", end="")
        return None
    min_length = min([len(r) for r in results])
    logger.trace(f"pos {pos} off {offset}: {results}, min:{min_length}\n", end="")
    return [result for result in results if (len(result) == min_length)][0]


def solver_board(
    board: AbstractBoard,
    all_rules: List[AbstractRule]
):
    model = board.get_model()
    switch = Switch()

    for rule in all_rules:
        if rule is None:
            continue
        rule: AbstractRule
        rule.create_constraints(board, switch)

    for key in board.get_board_keys():
        for pos, obj in board(key=key):
            if obj is None:
                continue
            obj: AbstractValue
            obj.create_constraints(board, switch)

    # 3.获取所有变量并赋值已解完的部分
    for key in board.get_board_keys():
        for _, var in board("C", mode="variable", key=key, special='raw'):
            model.Add(var == 0)
        for _, var in board("F", mode="variable", key=key, special='raw'):
            model.Add(var == 1)

    model.AddBoolAnd(switch.get_all_vars())

    solver = get_solver(True)
    status = timer(solver.Solve)(model)
    positions = []

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    for key in board.get_board_keys():
        for pos, var in board(key=key, mode="variable", special='raw'):
            if (
                board.get_type(pos, special='raw') == "N" and
                solver.Value(var)
            ):
                positions.append(pos)

    return positions


def solver_model(
    model: cp_model,
    back_solver: bool = False
):
    solver = get_solver(False)
    status = timer(solver.Solve)(model)
    if back_solver:
        return status in (cp_model.OPTIMAL, cp_model.FEASIBLE), solver
    return status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
