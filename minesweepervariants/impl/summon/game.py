#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/07/03 00:27
# @Author  : Wu_RH
# @FileName: game.py
from ctypes import pointer
from enum import Enum, Flag
import queue
from re import A
import threading
import time
from typing import Any, Union, Callable, List, Tuple, Optional

from ...abs.Lrule import Rule0R
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...abs.Mrule import AbstractMinesValue
from ...abs.Rrule import AbstractClueValue, ValueQuess
from ...abs.board import AbstractBoard
from ...abs.board import AbstractPosition
from . import Summon
from .solver import solver_by_csp, hint_by_csp, Switch, deduced_by_csp, solver_model
from ...utils.impl_obj import MINES_TAG, VALUE_QUESS, POSITION_TAG, serialize as tag_serialize, decode as tag_decode
from ...utils.tool import get_logger, get_random

from minesweepervariants.config.config import DEFAULT_CONFIG

# ==== 获取默认值 ====
CONFIG = {}
CONFIG.update(DEFAULT_CONFIG)


class Mode(Enum):
    NORMAL = 0  # 普通模式
    EXPERT = 1  # 专家模式
    ULTIMATE = 2  # 终极模式
    PUZZLE = 3  # 纸笔模式(用于调试)


class UMode(Flag):
    ULTIMATE_A = 1
    ULTIMATE_F = 2
    ULTIMATE_S = 4
    ULTIMATE_R = 8
    ULTIMATE_P = 16


NORMAL = Mode.NORMAL
EXPERT = Mode.EXPERT
ULTIMATE = Mode.ULTIMATE
PUZZLE = Mode.PUZZLE

ULTIMATE_A = UMode.ULTIMATE_A
ULTIMATE_F = UMode.ULTIMATE_F
ULTIMATE_S = UMode.ULTIMATE_S
ULTIMATE_R = UMode.ULTIMATE_R
ULTIMATE_P = UMode.ULTIMATE_P


class ValueAsterisk(AbstractClueValue):
    def __init__(self, pos: 'AbstractPosition', code: bytes = b''):
        pass

    def __repr__(self) -> str:
        return "*"

    @classmethod
    def type(cls) -> bytes:
        return b"*"

    def code(self) -> bytes:
        return b""

    def high_light(self, board: 'AbstractBoard') -> List['AbstractPosition'] | None:
        return []


class MinesAsterisk(AbstractMinesValue):
    def __init__(self, pos: 'AbstractPosition', code: bytes = b''):
        pass

    def __repr__(self) -> str:
        return "#"

    @classmethod
    def type(cls) -> bytes:
        return b"F*"

    def code(self) -> bytes:
        return b""

    def high_light(self, board: 'AbstractBoard') -> List['AbstractPosition'] | None:
        return []


# VALUE_TAG = ValueAsterisk(POSITION_TAG)
# VALUE_MINES = MinesAsterisk(POSITION_TAG)


class Manger:
    def __init__(self, fn):
        self.fn: Callable = fn
        self.wait_args = None
        self.lock = threading.Lock()
        self.fn_lock = threading.Lock()

    def wait(self):
        my_id = threading.get_ident()  # 当前线程唯一ID

        if not self.fn_lock.locked():
            with self.fn_lock:
                self.fn(thread=True)
            return
        with self.lock:
            self.wait_args = my_id
        while True:
            if self.wait_args is not my_id:
                return
            if not self.fn_lock.locked():
                break
            time.sleep(0.1)
        with self.fn_lock:
            self.fn(thread=True)


class GameSession:
    flag_tag: Any
    clue_tag: Any

    def __init__(
            self, summon: Summon = None, mode=NORMAL, drop_r=False,
            ultimate_mode=ULTIMATE_A | ULTIMATE_F | ULTIMATE_S,
    ):
        self.logger = get_logger()
        self.summon = summon
        self.drop_r = drop_r
        if mode == ULTIMATE:
            if ultimate_mode & ULTIMATE_R:
                self.drop_r = False
            else:
                self.drop_r = True

        self.answer_board = None
        self.board = None

        self.mode = mode
        self.ultimate_mode = ultimate_mode

        self.last_deduced = [None, []]
        self.last_hint = [None, {}]
        self.deducedManger = Manger(self.deduced)
        self.hintManger = Manger(self.hint)
        self.deduced_queue = queue.Queue(maxsize=1)
        self.hint_queue = queue.Queue(maxsize=1)

        self.flag_tag = MinesAsterisk(POSITION_TAG)
        self.clue_tag = ValueAsterisk(POSITION_TAG)

        self.create_schedule_data = [0, time.time()]

    @property
    def answer_board(self):
        result = self.__dict__["answer_board"]
        if result is None:
            raise AttributeError("answer_board未初始化")
        return result

    @answer_board.setter
    def answer_board(self, value):
        self.__dict__["answer_board"] = value

    @property
    def board(self):
        result = self.__dict__["board"]
        if result is None:
            raise AttributeError("board未初始化")
        return result

    @board.setter
    def board(self, value):
        self.__dict__["board"] = value
        try:
            self.origin_board = value.clone()
        except:
            self.origin_board = value

    def unbelievable(self, pos, action: int):
        """
        action
            0: 左键点击/翻开/设置非雷
            1: 右键点击/标雷/设置必雷
        """
        if self.answer_board.get_type(pos) == "F" and action == 0:
            return [
                pos for key in self.answer_board.get_board_keys()
                for pos, _ in self.answer_board(key=key)
                if (self.board.get_type(pos) == "N" and
                    self.answer_board.get_type(pos) == "F")
            ]
        if self.answer_board.get_type(pos) == "C" and action == 1:
            return [
                pos for key in self.answer_board.get_board_keys()
                for pos, _ in self.answer_board(key=key)
                if (self.board.get_type(pos) == "N" and
                    self.answer_board.get_type(pos) == "F")
            ]
        all_rules = self.summon.mines_rules.rules[:]
        all_rules += [self.summon.clue_rule, self.summon.mines_clue_rule]
        if self.drop_r:
            all_rules = [rule for rule in all_rules if not isinstance(rule, Rule0R)]
        board = self.board.clone()
        if action == 0:
            board[pos] = self.flag_tag
        else:
            board[pos] = self.clue_tag
        board: AbstractBoard
        print("=" * 20)
        print(board)
        print(board)
        print("=" * 20)
        model = board.get_model()
        switch = Switch()
        for rule in all_rules:
            rule.create_constraints(board, switch)
        for key in board.get_board_keys():
            for pos, obj in board(key=key):
                obj_type = board.get_type(pos)
                var = board.get_variable(pos)
                if obj_type == "F":
                    model.Add(var == 1)
                if obj_type == "C":
                    model.Add(var == 0)
                if obj in [
                    None, MINES_TAG, VALUE_QUESS,
                    self.clue_tag, self.flag_tag
                ]:
                    continue
                obj.create_constraints(board, switch)
        model.AddBoolAnd(switch.get_all_vars())
        state, solver = solver_model(model, True)
        print("unbelievable state:", state)
        if not state:
            return None
        mines_list = []
        for key in board.get_board_keys():
            for pos, var in board(mode="var", key=key):
                if self.board.get_type(pos) != "N":
                    continue
                if solver.Value(var) == 0:
                    continue
                mines_list.append(pos)
        return mines_list

    def thread_hint(self):
        threading.Thread(target=self.hintManger.wait).start()

    def thread_deduced(self):
        threading.Thread(target=self.deducedManger.wait).start()

    def create_schedule(self) -> Tuple[float, float, float]:
        """
        生成题板的进度
        返回: 进度(0-1), 已用时间(seconds), 预估总用时(seconds)
        """
        schedule, start_time = self.create_schedule_data
        elapsed = time.time() - start_time
        total = elapsed / schedule if schedule > 0 else float('inf')
        return schedule, elapsed, total

    def _create_board(self) -> Optional["AbstractBoard"]:
        board = self.answer_board.clone()
        random = get_random()
        board: AbstractBoard
        model = board.get_model()

        for rule in self.summon.mines_rules.rules + [
            self.summon.clue_rule, self.summon.mines_clue_rule
        ]:
            rule.init_clear(board)

        # 尝试直接初始化所有未使用类型检查的规则
        used_type_rule = []     # 使用了类型检查的规则
        board.used_type()
        for rule in self.summon.mines_rules.rules + [
            self.summon.clue_rule, self.summon.mines_clue_rule
        ]:
            if isinstance(rule, Rule0R):
                continue
            switch = Switch()
            rule.create_constraints(board, switch)
            if board.used_type():
                used_type_rule.append((rule, switch.get_all_vars()))
            else:
                # 直接强制实现所有左线
                model.AddBoolAnd(switch.get_all_vars())

        # 尝试直接初始化所有未使用类型检查的线索
        used_type_pos = []      # 使用了类型检查的对象
        pos_switchs = {}
        for key in board.get_board_keys():
            for pos, obj in board(key=key):
                if obj is None:
                    continue
                switch = Switch()
                obj.create_constraints(board, switch)
                pos_switch = model.NewBoolVar(f"{pos}:{obj}(switch)")
                model.AddBoolAnd(
                    switch.get_all_vars()
                ).OnlyEnforceIf(pos_switch)
                model.AddBoolAnd(
                    [v.Not() for v in switch.get_all_vars()]
                ).OnlyEnforceIf(pos_switch.Not())
                if board.used_type():
                    used_type_pos.append((pos, pos_switch))
                pos_switchs[pos] = pos_switch

        # if not used_type_pos + used_type_rule:
        #     # 如果全部没有使用过type检查 就走untype生成
        #     return self.create_board_by_untype()

        positions = [pos for pos, _ in board()]
        random.shuffle(positions)

        while positions:
            pos = positions.pop()
            if pos in pos_switchs:
                # 如果在里面代表未使用类型检查
                ...

        self.board = board
        return board

    def create_board_by_untype(
            self
    ) -> Union["AbstractBoard", None]:
        """
        未使用任何type检查的生成题板
        """

    def create_board(self) -> Union["AbstractBoard", None]:
        """
        一层具象
        终极模式的规则是 直到推无可推再给下一步线索 如果倒过来想呢
        倒过来就是删无可删 再重头删一遍

        具体操作
        初始化: 将所有雷设置为None
        第一步: 将整个版面的线索都替换为雷试下能不能无解 如果无解代表矛盾
        第二步: 现在整个题板都遍历完了一遍 看起来已经是删无可删了 那么就
        """
        board = self.answer_board.clone()
        for rule in (self.summon.mines_rules.rules
                     + [self.summon.clue_rule,
                        self.summon.mines_clue_rule]):
            rule.init_clear(board)
        clues = [i for i in board("CF")]
        all_schedule = len(clues)
        self.create_schedule_data = [0.0, time.time()]
        print("game init:", board.show_board(), clues)
        get_random().shuffle(clues)
        while clues:
            while True:
                self.create_schedule_data[0] = (all_schedule - len(clues)) / all_schedule
                if not clues:
                    break
                pos, clue = clues.pop()
                if board.get_type(pos, special='raw') == "C":
                    board.set_value(pos, MINES_TAG)
                elif board.get_type(pos, special='raw') == "F":
                    board.set_value(pos, self.clue_tag)
                if solver_by_csp(
                        self.summon.mines_rules,
                        self.summon.clue_rule,
                        self.summon.mines_clue_rule,
                        board.clone(), drop_r=False) == 0:
                    board.set_value(pos, None)
                    break
                board.set_value(pos, clue)
        if solver_by_csp(
            self.summon.mines_rules,
            self.summon.clue_rule,
            self.summon.mines_clue_rule,
            board.clone(),
            answer_board=self.answer_board,
            drop_r=True
        ) == 1:
            # 不使用R推会导致多解
            self.drop_r = False
        self.board = board
        self.origin_board = board.clone()
        return board

    def chord_clue(self, clue_pos: AbstractPosition) -> list[AbstractPosition]:
        # 看最后一次提示有没有包含该格的单线索
        VALUE = self.board.get_config(clue_pos.board_key, "VALUE")
        MINES = self.board.get_config(clue_pos.board_key, "MINES")
        self.logger.trace("chord")
        if self.board[clue_pos] in [VALUE, MINES, self.clue_tag, self.flag_tag, None]:
            return []
        if self.board == self.last_hint[0]:
            for objs, positions in self.last_hint[1].items():
                if len(objs) == 1 and objs[0] == clue_pos:
                    return positions
            return []
        obj = self.board.get_value(clue_pos)
        board: AbstractBoard = self.board.clone()
        chord_positions = []
        if obj.deduce_cells(board) is not None:
            for pos, obj in self.board():
                if (obj is None) and (board[pos] is not None):
                    chord_positions.append(pos)
            return chord_positions

        for pos, obj_type in board(mode="type", special='raw'):
            if clue_pos == pos:
                continue
            if obj_type == "C":
                board[pos] = self.clue_tag
            elif obj_type == "F":
                board[pos] = MINES_TAG

        chord_positions = self._deduced(board, [
            self.summon.mines_clue_rule
            if self.board.get_type(clue_pos, special='raw') == "F" else
            self.summon.clue_rule,
        ])

        self.logger.trace(f"chord pos: {clue_pos}, {self.board[clue_pos]}, {chord_positions}")

        return chord_positions

    def apply(self, pos: AbstractPosition, action: int) -> Union["AbstractBoard", None]:
        """
        :param pos: 交互位置
        :param action: 操作代码
            0: 左键点击/翻开/设置非雷
            1: 右键点击/标雷/设置必雷
        """
        global NORMAL, EXPERT, ULTIMATE, PUZZLE
        _board = None
        if self.mode == PUZZLE:
            if action == 1:
                value_tag = self.board.get_config(pos.board_key, "MINES")
                self.board.set_value(pos, self.flag_tag if value_tag == VALUE_QUESS else value_tag)
            elif action == 0:
                value_tag = self.board.get_config(pos.board_key, "VALUE")
                self.board.set_value(pos, self.clue_tag if value_tag == VALUE_QUESS else value_tag)
            return self.board
        if self.board.get_type(pos, special='raw') != "N":
            # 点击了线索
            chord_positions = self.chord_clue(pos)
            if self.mode in [NORMAL, EXPERT]:
                # 普通和专家直接设置值
                for _pos in chord_positions:
                    self.board[_pos] = self.answer_board[_pos]
            elif self.mode in [ULTIMATE, PUZZLE]:
                # 如果是纸笔和专家就放标志
                for _pos in chord_positions:
                    if _pos.board_key not in self.board.get_interactive_keys():
                        if self.answer_board.get_type(_pos, special='raw') == "C":
                            self.board[_pos] = self.board.get_config(_pos.board_key, "VALUE")
                        elif self.answer_board.get_type(_pos, special='raw') == "F":
                            self.board[_pos] = self.board.get_config(_pos.board_key, "MINES")
                    elif self.answer_board.get_type(_pos, special='raw') == "F":
                        self.board[_pos] = self.flag_tag
                    elif self.answer_board.get_type(_pos, special='raw') == "C":
                        self.board[_pos] = self.clue_tag

        elif self.mode == NORMAL:
            # 普通模式
            if not action and self.answer_board.get_type(pos, special='raw') == "F":
                return None
            if action and self.answer_board.get_type(pos, special='raw') == "C":
                return None
            self.board[pos] = self.answer_board[pos]
        elif self.mode in [EXPERT, ULTIMATE, PUZZLE]:
            # 非普通模式
            _board = self.board.clone()
            if pos not in self.last_deduced[1]:
                print(f"apply {pos} 未命中 {self.last_deduced[1]}")
                if self.unbelievable(pos, action) is not None:
                    return None
                else:
                    self.last_deduced[1].append(pos)
            if action and self.answer_board.get_type(pos, special='raw') == "C":
                return None
            if not action and self.answer_board.get_type(pos, special='raw') == "F":
                return None
            if self.mode in [ULTIMATE, PUZZLE]:
                self.board[pos] = self.flag_tag if action else self.clue_tag
            else:
                self.board[pos] = self.answer_board[pos]
        else:
            return None
        if pos.board_key not in self.board.get_interactive_keys():
            if self.board[pos] is self.clue_tag:
                self.board[pos] = self.board.get_config(pos.board_key, "VALUE")
            elif self.board[pos] is self.flag_tag:
                self.board[pos] = self.board.get_config(pos.board_key, "MINES")
        if (
            self.mode in [ULTIMATE] and
            self.ultimate_mode & ULTIMATE_A
        ):
            self.step()
            if self.drop_r:
                flag = True
                if self.ultimate_mode & ULTIMATE_F:
                    for pos in self.deduced():
                        if self.answer_board.get_type(pos, special='raw') != "F":
                            continue
                        if pos.board_key not in self.board.get_interactive_keys():
                            continue
                        flag = False
                        break
                if self.ultimate_mode & ULTIMATE_S:
                    for pos in self.deduced():
                        if pos.board_key in self.board.get_interactive_keys():
                            continue
                        flag = False
                        break
                for pos in self.deduced():
                    if pos.board_key not in self.board.get_interactive_keys():
                        continue
                    if self.answer_board.get_type(pos, special='raw') == "F":
                        continue
                    flag = False
                    break
                if flag:
                    self.drop_r = False
                    self.last_deduced[0] = None
                    self.deduced()
            self.thread_deduced()
        return self.board

    def click(self, pos: "AbstractPosition") -> Union["AbstractBoard", None]:
        """
        翻开/点击 某个空白格
        :param pos: 翻开的位置
        """
        return self.apply(pos, 0)

    def mark(self, pos: AbstractPosition) -> Union["AbstractBoard", None]:
        """
        右键标雷
        """
        return self.apply(pos, 1)

    def step(self) -> bool:
        print("step")
        if self.ultimate_mode & ULTIMATE_F:
            for pos in self.deduced():
                if self.answer_board.get_type(pos, special='raw') != "F":
                    continue
                if pos.board_key not in self.board.get_interactive_keys():
                    continue
                print("[step]has flag")
                return False
        if self.ultimate_mode & ULTIMATE_S:
            for pos in self.deduced():
                if pos.board_key in self.board.get_interactive_keys():
                    continue
                print("[step]has mark")
                return False
        for pos in self.deduced():
            if pos.board_key not in self.board.get_interactive_keys():
                continue
            if self.answer_board.get_type(pos, special='raw') == "F":
                continue
            print("[step]has clue")
            return False
        for key in self.board.get_board_keys():
            for pos, obj in self.board(key=key):
                if obj not in [self.clue_tag, self.flag_tag]:
                    continue
                self.board[pos] = self.answer_board[pos]

        self.last_deduced[0] = None
        self.last_hint[0] = None
        self.thread_hint()
        return True

    def deduced(self, thread=True):
        """
        收集所有必然能推出的位置及其不可能的值
        """
        if self.last_deduced[0] == self.board:
            return self.last_deduced[1]
        # 如果是终极模式
        if self.mode == ULTIMATE and self.last_deduced[1]:
            for pos in self.last_deduced[1][:]:
                if self.board[pos] is not None:
                    self.last_deduced[1].remove(pos)
            if self.ultimate_mode & ULTIMATE_F:
                for pos in self.last_deduced[1]:
                    if self.answer_board.get_type(pos, special='raw') != "F":
                        continue
                    if pos.board_key not in self.board.get_interactive_keys():
                        continue
                    print("[deduced]has flag")
                    return self.last_deduced[1]
            if self.ultimate_mode & ULTIMATE_S:
                for pos in self.last_deduced[1]:
                    if pos.board_key in self.board.get_interactive_keys():
                        continue
                    print("[deduced]has mark")
                    return self.last_deduced[1]
            for pos in self.last_deduced[1]:
                if pos.board_key not in self.board.get_interactive_keys():
                    continue
                if self.answer_board.get_type(pos, special='raw') == "F":
                    continue
                print("[deduced]has clue")
                return self.last_deduced[1]
        self.deduced_queue.put("process")  # 请求处理权
        try:
            t = time.time()
            self.logger.debug(f"deduced start (last: {str(self.last_deduced[1])})")
            deduced = []
            board = self.board.clone()
            for pos in self.last_deduced[1]:
                if board.get_type(pos, special='raw') != "N":
                    continue
                if self.answer_board.get_type(pos, special='raw') == "F":
                    board[pos] = self.flag_tag
                elif self.answer_board.get_type(pos, special='raw') == "C":
                    board[pos] = self.clue_tag
                deduced.append(pos)
            all_rules = self.summon.mines_rules.rules[:]
            all_rules += [self.summon.clue_rule, self.summon.mines_clue_rule]
            deduced += self._deduced(board, all_rules)

            self.last_deduced[1] = deduced
            self.last_deduced[0] = self.board.clone()

            self.logger.debug(f"last_deduced {str(self.last_deduced[1])}")
            self.logger.debug(f"deduced used time {time.time() - t}")

            return deduced
        finally:
            self.deduced_queue.get()
            self.deduced_queue.task_done()

    def hint(self, thread=False):
        if self.last_hint[0] == self.board:
            return self.last_hint[1]

        if None not in [obj for pos, obj in self.board()]:
            return {}

        deduced = self.deduced()
        if not deduced and thread:
            return {}
        if not deduced and self.mode != ULTIMATE:
            self.logger.error("题板无可推格")
            return {}
        if self.mode == ULTIMATE and not deduced:
            _board = self.board.clone()
            self.step()
            positions = []
            if not (self.drop_r and self.deduced()):
                self.drop_r = False
                positions.append(("R", None))
            self.last_deduced[0] = None
            self.last_hint[0] = None
            for pos, obj in _board():
                if type(obj) in [type(self.clue_tag), type(self.flag_tag)]:
                    positions.append(pos)
            print("step: ", positions)
            return {tuple(positions): []}

        self.hint_queue.put("process")  # 请求处理权
        try:
            board = self.board.clone()
            t = time.time()
            hint = self._hint(board)
            self.logger.debug(f"all_hint {hint}")
            self.logger.debug(f"hint used_time {time.time() - t}")

            return hint
        finally:
            self.hint_queue.get()
            self.hint_queue.task_done()

    def _deduced(self, board, all_rules):
        self.logger.trace("构建新模型")
        self.logger.trace(f"deduced all_rules: {all_rules}")
        self.logger.trace(f"deduced drop_r: {self.drop_r}")
        board.clear_variable()
        model = board.get_model()
        switch = Switch()

        # 2.获取所有规则约束
        for rule in all_rules:
            if rule is None:
                continue
            if self.drop_r and isinstance(rule, Rule0R):
                continue
            rule.create_constraints(board, switch)

        for key in board.get_board_keys():
            for pos, obj in board(key=key):
                if obj is None:
                    continue
                obj.create_constraints(board, switch)

        # 3.获取所有变量并赋值已解完的部分
        for key in board.get_board_keys():
            for _, var in board("C", mode="variable", key=key, special='raw'):
                model.Add(var == 0)
                self.logger.trace(f"var: {var} == 0")
            for _, var in board("F", mode="variable", key=key, special='raw'):
                model.Add(var == 1)
                self.logger.trace(f"var: {var} == 1")

        for switch_var in switch.get_all_vars():
            model.Add(switch_var == 1)

        results = []
        future_to_param = {}

        with ThreadPoolExecutor(max_workers=CONFIG["workes_number"]) as executor:
            # 提交任务
            for key in board.get_board_keys():
                for pos, _ in board("N", key=key):
                    fut = executor.submit(
                        deduced_by_csp,
                        board,
                        self.answer_board,
                        pos
                    )
                    future_to_param[fut] = pos  # 记录参数以便出错追踪

            # 收集结果
            for fut in as_completed(future_to_param):
                pos = future_to_param[fut]
                try:
                    self.logger.trace(f"deduced pos {pos} wait")
                    result = fut.result()
                    self.logger.trace(f"deduced pos {pos} end: {result}")
                    if result is None:
                        continue
                    if result:
                        results.append(pos)
                except Exception as exc:
                    raise exc

        self.logger.trace(f"deduced done: {results}")

        return results

    def _hint(self, board) -> dict[tuple, list[AbstractPosition]]:
        """
        返回每一类推理依据及其能推出的位置
        """
        deduced = self.deduced()
        get_random().shuffle(deduced)
        future_to_param = {}
        results = {}
        with ThreadPoolExecutor(max_workers=CONFIG["workes_number"]) as executor:
            # 提交任务
            for pos, obj in board("CF"):
                fut = executor.submit(
                    self.chord_clue,
                    pos
                )
                future_to_param[fut] = pos  # 记录参数以便出错追踪

            # 收集结果
            for fut in as_completed(future_to_param):
                pos = future_to_param[fut]
                try:
                    result = fut.result()
                    if not result:
                        continue
                    results[tuple([pos])] = result
                except Exception as exc:
                    raise exc
        if results:
            self.last_hint[1] = results
            self.last_hint[0] = board.clone()
            return results

        self.logger.trace("构建新模型")
        board.clear_variable()
        model = board.get_model()
        switch = Switch()

        all_rules = self.summon.mines_rules.rules[:]
        all_rules += [self.summon.clue_rule, self.summon.mines_clue_rule]

        # 2.获取所有规则约束
        for rule in all_rules:
            if rule is None:
                continue
            if self.drop_r and isinstance(rule, Rule0R):
                continue
            rule.create_constraints(board, switch)

        for key in board.get_board_keys():
            for pos, obj in board(key=key):
                if obj is None:
                    continue
                if obj.invalid(board):
                    continue
                obj.create_constraints(board, switch)

        # 3.获取所有变量并赋值已解完的部分
        for key in board.get_board_keys():
            for _, var in board("C", mode="variable", key=key, special='raw'):
                model.Add(var == 0)
                self.logger.trace(f"var: {var} == 0")
            for _, var in board("F", mode="variable", key=key, special='raw'):
                model.Add(var == 1)
                self.logger.trace(f"var: {var} == 1")

        results = {}
        future_to_param = {}
        # upper_bound = None
        upper_bound = [float("inf"), threading.Lock()]

        with ThreadPoolExecutor(max_workers=CONFIG["workes_number"]) as executor:
            # 提交任务
            for pos in deduced:
                fut = executor.submit(
                    hint_by_csp, board,
                    self.answer_board,
                    switch, pos, upper_bound
                )
                future_to_param[fut] = pos  # 记录参数以便出错追踪

            # 收集结果
            for fut in as_completed(future_to_param):
                pos = future_to_param[fut]
                try:
                    self.logger.trace(f"pos[{pos}]: start")
                    _result = fut.result()
                    self.logger.trace(f"pos[{pos}]: {_result}")
                    if _result is None:
                        continue
                    self.logger.trace(deduced)
                    _result.sort()
                    _result = list(tuple(set(_result)))
                    result = set()
                    for k in _result:
                        bes_type = k[0].split("|", 1)[0]
                        name = k[0].split("|", 1)[1]
                        if bes_type == "RULE":
                            result.add((name, k[1]))
                        elif bes_type == "POS":
                            info = name.split("|", 2)
                            result.add(
                                board.get_pos(
                                    int(info[0]),
                                    int(info[1]),
                                    info[2]
                                ))
                    result = tuple(result)
                    if result not in results:
                        results[result] = []
                    results[result].append(pos)
                except Exception as exc:
                    raise exc

        self.last_hint[0] = board.clone()
        self.last_hint[1] = results

        return results

    def check_difficulty(self, diff=None):
        """
        diff: (min, max) or None
        如果diff为None则返回所有线索线索数
        min为n的时候表示线索数至少包含n线索推理
        max为n的时候表示线索数至多包含n线索推理
        """
        clue_freq = {}
        _board = self.board.clone()
        n_num = len([None for key in _board.get_board_keys()
                     for _ in _board('N', key=key)])
        while self.board.has("N"):
            if diff is not None and clue_freq:
                if diff[1] is not None:
                    if max(clue_freq.keys()) > diff[1]:
                        break
                elif diff[0] is not None:
                    if max(clue_freq.keys()) >= diff[0]:
                        break
                else:
                    break

            print(f"{n_num - len([None for key in self.board.get_board_keys() for _ in self.board('N', key=key)])}"
                  f"/{n_num}", end="\r")
            num_clues_used = float("inf")

            n_length = len([None for key in self.board.get_board_keys() for _ in self.board('N', key=key)])
            print(f"{n_num - n_length}/{n_num}", end="\r")
            self.logger.debug("\n" + self.board.show_board())
            self.logger.debug(clue_freq)
            grouped_hints = self.hint().items()
            if not grouped_hints:
                self.logger.warn("hint无返回值")
                self.logger.warn("\n" + self.board.show_board())
                clue_freq = None
                break
            self.logger.debug("\n" + self.board.show_board())
            [self.logger.debug(str(i[0]) + " -> " + str(i[1])) for i in grouped_hints]
            pos_clues = {}
            for hints, deduceds in grouped_hints:
                if ("R", None) in hints:
                    hints_length = 1 + (len(hints) // 4)
                else:
                    hints_length = len(hints)
                if hints_length > num_clues_used:
                    continue
                elif hints_length < num_clues_used:
                    num_clues_used = hints_length
                    pos_clues.clear()
                for deduced in deduceds:
                    pos_clues[deduced] = num_clues_used
            for pos in pos_clues:
                imposs = self.answer_board.get_type(pos, special='raw')
                self.apply(pos, 0 if imposs == "C" else 1)
                if pos_clues[pos] not in clue_freq:
                    clue_freq[pos_clues[pos]] = 0
                clue_freq[pos_clues[pos]] += 1
            self.logger.debug("\n" + self.board.show_board())
            self.logger.debug(clue_freq)
        self.board = _board
        return clue_freq if clue_freq else None

    def get_generation_progress(self) -> tuple[float, float, float]:
        # returns (progress(0..1), used_time(s), total_time(s))
        # if not generating: raise RuntimeError

        raise NotImplementedError

    def serialize(self):
        data = {
            'mode': self.mode.value,
            'drop_r': self.drop_r,
            'ultimate_mode': self.ultimate_mode.value,
            'flag_tag': tag_serialize(self.flag_tag),
            'clue_tag': tag_serialize(self.clue_tag),
            'answer_board': self.answer_board.serialize(),
            'board': self.board.serialize(),
        }
        return data


    @classmethod
    def from_dict(cls, data):
        self = cls(
            mode=Mode(data['mode']),
            drop_r=data['drop_r'],
            ultimate_mode=UMode(data['ultimate_mode']),
        )
        self.flag_tag = tag_decode(data['flag_tag'])
        self.clue_tag = tag_decode(data['clue_tag'])
        self.answer_board = AbstractBoard.from_str(data['answer_board'])
        self.board = AbstractBoard.from_str(data['board'])

        return self

def main():
    get_logger(log_lv="TRACE")
    # get_random(new=True, seed=8894987)
    # get_random(seed=5474554)
    size = (5, 5)
    rules = ["*3T"]
    s = Summon(size, -1, rules)
    g = GameSession(s, ULTIMATE, False, ULTIMATE_R)
    # g.board = s.create_puzzle()
    # g.answer_board = s.answer_board
    g.answer_board = s.summon_board()
    g.create_board()
    # for p, i in g.board("C"):
    #     g.chord_clue(p)
    # g.drop_r = True
    print("=" * 20)
    print(g.board)
    print("=" * 20)
    print(d := g.deduced())
    print("=" * 20)
    print(g.board)
    print("=" * 20)
    # print(f"deduced: {d}")
    print(h := g.hint())
    print("=" * 20)
    print(g.board)
    print("=" * 20)
    for b, t in h.items():
        print(b, "->", t)
    # g.create_board()
    # while "N" in g.board:
    #     print(g.hint())
    #     print(g.deduced_values)
    #     print(g.board)
    #     for p, v in list(g.deduced_values.items())[:]:
    #         if v is MINES_TAG:
    #             g.click(p)
    #         else:
    #             g.mark(p)
    # print(g.__dict__)
    # print(g.answer_board)
    # print(g.answer_board[g.answer_board.get_pos(1, 1)].high_light(g.answer_board))


if __name__ == '__main__':
    main()
