from collections import defaultdict
from typing import Any, Iterable, Literal, Optional, cast, overload

from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import IntVar, cp_model_pb2

from ...utils.timer import timer
from ...utils.tool import get_logger
from ...config.config import DEFAULT_CONFIG
from ...abs.board_set import BoardSet

CONFIG = {}
CONFIG.update(DEFAULT_CONFIG)


def get_solver(enable_timeout: bool = True) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.random_seed = 42
    solver.parameters.num_search_workers = CONFIG.get("workes_number", 4)
    solver.parameters.search_branching = cp_model.AUTOMATIC_SEARCH
    solver.parameters.linearization_level = 2
    solver.parameters.use_optional_variables = True
    solver.parameters.randomize_search = True

    if enable_timeout and CONFIG.get("timeout", 0) > 0:
        solver.parameters.max_time_in_seconds = CONFIG["timeout"]

    return solver

SWITCH_CATEGORIES = Literal["enforce", "rule", "subrule", "clue"]

class ModelV2:
    board_set: BoardSet
    model: cp_model.CpModel

    _logger = get_logger("ModelV2")

    switch_registry: dict[SWITCH_CATEGORIES, dict[str, IntVar]]

    last_solver: Optional[cp_model.CpSolver]
    last_status: Optional[cp_model_pb2.CpSolverStatus]


    def __init__(self, board_set: BoardSet):
        self.board_set = board_set
        self.reset()

    def reset(self):
        """重置求解器"""
        self.model = cp_model.CpModel()
        self.switch_registry = {
            cat: {} for cat in SWITCH_CATEGORIES.__args__
        }
        self.last_solver = None
        self.last_status = None

    def get_switch(self, category: SWITCH_CATEGORIES, key: str) -> IntVar:
        # 检查是否已存在
        if key in self.switch_registry[category]:
            return self.switch_registry[category][key]

        # 创建新变量
        var = self.model.NewBoolVar(f"SW|{category}|{key}")
        self.switch_registry[category][key] = var
        return var

    @overload
    def query_switch(self, category: SWITCH_CATEGORIES) -> Iterable[str]: ...

    @overload
    def query_switch(self, category: SWITCH_CATEGORIES, key: str) -> bool: ...

    @overload
    def query_switch(self, category: SWITCH_CATEGORIES, *, prefix: str) -> Iterable[str]: ...

    @overload
    def query_switch(self) -> Iterable[tuple[SWITCH_CATEGORIES, str]]: ...

    def query_switch(self, category: Optional[SWITCH_CATEGORIES] = None, key: Optional[str] = None, *, prefix: Optional[str] = None) -> bool | Iterable[str] | Iterable[tuple[SWITCH_CATEGORIES, str]]:
        if category is None:
            return (cast(tuple[SWITCH_CATEGORIES, str], (cat, key))
                for cat, cat_dict in self.switch_registry.items()
                for key in cat_dict.keys()
            )

        if key is not None:
            return key in self.switch_registry[category]

        if prefix is not None:
            return (key
                for key in self.switch_registry[category].keys()
                if key.startswith(prefix)
            )

        return self.switch_registry[category].keys()

    def add_constraint_callback(
        self,
        callback,
        *args,
        **kwargs
    ):
        # 过渡方案
        callback(self.model, self.board_set, *args, **kwargs)

    def solve(self, enable_timeout=True) -> bool:
        """求解模型"""
        solver = get_solver(enable_timeout=enable_timeout)
        status = timer(solver.Solve)(self.model)

        self.last_solver = solver
        self.last_status = status

        return status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    def solve_model(self, model: cp_model.CpModel, enable_timeout: bool = False) -> tuple[bool, cp_model_pb2.CpSolverStatus, cp_model.CpSolver]:
        solver = get_solver(enable_timeout=enable_timeout)
        status = timer(solver.Solve)(model)
        return status in (cp_model.OPTIMAL, cp_model.FEASIBLE), status, solver

    def value(self, var: IntVar) -> Optional[int]:
        """获取变量的解值"""
        if not hasattr(self, 'last_solver'):
            return None
        if self.last_solver is None:
            return None
        return self.last_solver.Value(var)

    def _model_with_switch_subset(
        self,
        switches: list[IntVar],
        active_switches: list[IntVar],
        force_others_zero: bool = True
    ) -> cp_model.CpModel:
        model = self.model.clone()
        active_idx = {var.Index() for var in active_switches}
        for var in switches:
            if var.Index() in active_idx:
                model.Add(var == 1)
            elif force_others_zero:
                model.Add(var == 0)
        return model

    def unique(self, switches: Optional[list[IntVar]]) -> Literal[0, 1, 2]:
        """
        对给定开关集判定解数量：
        - 0: 无解
        - 1: 唯一解
        - 2: 多解
        """
        if switches is None:
            switches = [self.get_switch(cat, key) for cat, key in self.query_switch()]

        base = self.model.clone()
        feasible, status, solver = self.solve_model(base, enable_timeout=True)
        if not feasible:
            return 0

        first = {var.Index(): solver.Value(var) for var in switches}
        block = []
        for var in switches:
            if first[var.Index()] == 1:
                block.append(var.Not())
            else:
                block.append(var)

        check = self.model.clone()
        if block:
            check.AddBoolOr(block)
        feasible2, _, _ = self.solve_model(check, enable_timeout=True)
        return 2 if feasible2 else 1

    def mus(self, switches: list[IntVar]) -> Optional[list[IntVar]]:
        """
        求给定开关集的一个子集极小不可满足集（subset-minimal UNSAT subset）。
        若全集可满足则返回 None。
        """
        if switches is None:
            switches = [self.get_switch(cat, key) for cat, key in self.query_switch()]

        full_model = self._model_with_switch_subset(switches, switches)
        feasible, _, _ = self.solve_model(full_model, enable_timeout=False)
        if feasible:
            return None

        mus = switches[:]
        for var in switches:
            trial = [v for v in mus if v.Index() != var.Index()]
            trial_model = self._model_with_switch_subset(switches, trial)
            trial_feasible, _, _ = self.solve_model(trial_model, enable_timeout=False)
            if not trial_feasible:
                mus = trial
        return mus

    def mss(self, switches: list[IntVar]) -> list[IntVar]:
        """
        求给定开关集的一个极大可满足子集（subset-maximal SAT subset）。
        """
        if switches is None:
            switches = [self.get_switch(cat, key) for cat, key in self.query_switch()]

        selected: list[IntVar] = []
        for candidate in switches:
            trial = selected + [candidate]
            trial_model = self._model_with_switch_subset(switches, trial)
            feasible, _, _ = self.solve_model(trial_model, enable_timeout=False)
            if feasible:
                selected = trial
        return selected

    def _entails_switch_on(self, universe: list[IntVar], assumptions: list[IntVar], target: IntVar) -> bool:
        test_model = self._model_with_switch_subset(universe, assumptions, force_others_zero=False)
        test_model.Add(target == 0)
        feasible, _, _ = self.solve_model(test_model, enable_timeout=False)
        return not feasible

    def mes_on_switches(self, switches: list[IntVar]) -> Optional[list[IntVar]]:
        """
        求给定开关集的一个最小等价子集（subset-minimal equivalent set）。
        """
        full_model = self._model_with_switch_subset(switches, switches, force_others_zero=False)
        feasible, _, _ = self.solve_model(full_model, enable_timeout=False)
        if not feasible:
            # 本身不可满足
            return None

        # 返回子集 E，使得 M ∧ E 可满足，且 E 蕴含全集中每个开关都为 1
        mes = switches[:]
        changed = True
        while changed:
            changed = False
            for var in mes.copy():
                trial = [v for v in mes if v.Index() != var.Index()]
                trial_model = self._model_with_switch_subset(switches, trial, force_others_zero=False)
                trial_feasible, _, _ = self.solve_model(trial_model, enable_timeout=False)
                if not trial_feasible:
                    continue

                equivalent = True
                for must_on in switches:
                    if not self._entails_switch_on(switches, trial, must_on):
                        equivalent = False
                        break

                if equivalent:
                    mes = trial
                    changed = True
                    break
        return mes
