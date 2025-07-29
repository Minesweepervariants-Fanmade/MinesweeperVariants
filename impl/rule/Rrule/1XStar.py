#!/usr/bin/env python3

"""
[1X*] 王后 (Queen)：线索数表示斜向和横纵所有格子中的雷数
"""
from abs.Rrule import AbstractClueRule, AbstractClueValue
from abs.board import AbstractBoard, AbstractPosition

from utils.tool import get_logger
from utils.solver import get_model
from utils.impl_obj import VALUE_QUESS, MINES_TAG


class Rule1XStar(AbstractClueRule):
    name = "1X*"

    def fill(self, board: 'AbstractBoard') -> 'AbstractBoard':
        logger = get_logger()
        for pos, _ in board("N"):
            # 计算斜向和横纵所有格子中的雷数
            queen_positions = self._get_queen_positions(board, pos)
            value = len([_pos for _pos in queen_positions if board.get_type(_pos) == "F"])
            board.set_value(pos, Value1XStar(pos, count=value))
            logger.debug(f"Set {pos} to 1X*[{value}]")
        return board

    def _get_queen_positions(self, board: 'AbstractBoard', pos: AbstractPosition):
        """获取与给定位置在王后移动范围内的所有位置（横纵+斜向）"""
        positions = []
        # 获取棋盘的边界
        boundary = board.boundary()
        max_x, max_y = boundary.x, boundary.y

        # 横向方向（同行，相同x，不同y）
        for y in range(max_y + 1):
            other_pos = type(pos)(pos.x, y, pos.board_key)
            if other_pos != pos and board.in_bounds(other_pos):
                positions.append(other_pos)

        # 纵向方向（同列，相同y，不同x）
        for x in range(max_x + 1):
            other_pos = type(pos)(x, pos.y, pos.board_key)
            if other_pos != pos and board.in_bounds(other_pos):
                positions.append(other_pos)

        # 右上斜线方向 (x+1, y+1)
        for i in range(1, max(max_x, max_y) + 1):
            other_pos = type(pos)(pos.x + i, pos.y + i, pos.board_key)
            if board.in_bounds(other_pos):
                positions.append(other_pos)
            else:
                break

        # 左下斜线方向 (x-1, y-1)
        for i in range(1, max(max_x, max_y) + 1):
            other_pos = type(pos)(pos.x - i, pos.y - i, pos.board_key)
            if board.in_bounds(other_pos):
                positions.append(other_pos)
            else:
                break

        # 左上斜线方向 (x-1, y+1)
        for i in range(1, max(max_x, max_y) + 1):
            other_pos = type(pos)(pos.x - i, pos.y + i, pos.board_key)
            if board.in_bounds(other_pos):
                positions.append(other_pos)
            else:
                break

        # 右下斜线方向 (x+1, y-1)
        for i in range(1, max(max_x, max_y) + 1):
            other_pos = type(pos)(pos.x + i, pos.y - i, pos.board_key)
            if board.in_bounds(other_pos):
                positions.append(other_pos)
            else:
                break

        return positions

    def clue_class(self):
        return Value1XStar


class Value1XStar(AbstractClueValue):
    def __init__(self, pos: AbstractPosition, count: int = 0, code: bytes = None):
        super().__init__(pos, code)
        if code is not None:
            # 从字节码解码
            self.count = code[0]
        else:
            # 直接初始化
            self.count = count
        self.queen_positions = self._calculate_queen_positions()

    def _calculate_queen_positions(self):
        """计算与当前位置在王后移动范围内的所有位置"""
        positions = []
        # 这里我们需要在使用时获取board的大小，所以暂时返回空列表
        # 实际的位置计算会在具体方法中进行
        return positions

    def _get_queen_positions(self, board: 'AbstractBoard'):
        """获取与给定位置在王后移动范围内的所有位置（横纵+斜向）"""
        positions = []
        # 获取棋盘的边界
        boundary = board.boundary()
        max_x, max_y = boundary.x, boundary.y

        # 横向方向（同行，相同x，不同y）
        for y in range(max_y + 1):
            other_pos = type(self.pos)(self.pos.x, y, self.pos.board_key)
            if other_pos != self.pos and board.in_bounds(other_pos):
                positions.append(other_pos)

        # 纵向方向（同列，相同y，不同x）
        for x in range(max_x + 1):
            other_pos = type(self.pos)(x, self.pos.y, self.pos.board_key)
            if other_pos != self.pos and board.in_bounds(other_pos):
                positions.append(other_pos)

        # 右上斜线方向 (x+1, y+1)
        for i in range(1, max(max_x, max_y) + 1):
            other_pos = type(self.pos)(self.pos.x + i, self.pos.y + i, self.pos.board_key)
            if board.in_bounds(other_pos):
                positions.append(other_pos)
            else:
                break

        # 左下斜线方向 (x-1, y-1)
        for i in range(1, max(max_x, max_y) + 1):
            other_pos = type(self.pos)(self.pos.x - i, self.pos.y - i, self.pos.board_key)
            if board.in_bounds(other_pos):
                positions.append(other_pos)
            else:
                break

        # 左上斜线方向 (x-1, y+1)
        for i in range(1, max(max_x, max_y) + 1):
            other_pos = type(self.pos)(self.pos.x - i, self.pos.y + i, self.pos.board_key)
            if board.in_bounds(other_pos):
                positions.append(other_pos)
            else:
                break

        # 右下斜线方向 (x+1, y-1)
        for i in range(1, max(max_x, max_y) + 1):
            other_pos = type(self.pos)(self.pos.x + i, self.pos.y - i, self.pos.board_key)
            if board.in_bounds(other_pos):
                positions.append(other_pos)
            else:
                break

        return positions

    def __repr__(self):
        return f"{self.count}"

    @classmethod
    def type(cls) -> bytes:
        return b'1X*'

    def code(self) -> bytes:
        return bytes([self.count])

    def deduce_cells(self, board: 'AbstractBoard') -> bool:
        queen_positions = self._get_queen_positions(board)
        type_dict = {"N": [], "F": []}

        for pos in queen_positions:
            t = board.get_type(pos)
            if t in ("", "C"):
                continue
            type_dict[t].append(pos)

        n_num = len(type_dict["N"])
        f_num = len(type_dict["F"])

        if n_num == 0:
            return False

        # 如果已找到的雷数等于目标数，剩余格子都是安全的
        if f_num == self.count:
            for i in type_dict["N"]:
                board.set_value(i, VALUE_QUESS)
            return True

        # 如果已找到的雷数加上未知格子数等于目标数，剩余格子都是雷
        if f_num + n_num == self.count:
            for i in type_dict["N"]:
                board.set_value(i, MINES_TAG)
            return True

        return False

    def create_constraints(self, board: 'AbstractBoard'):
        """创建CP-SAT约束：王后移动范围内格子的雷数等于count"""
        model = get_model()

        # 收集王后移动范围内格子的布尔变量
        queen_positions = self._get_queen_positions(board)
        neighbor_vars = []

        for neighbor in queen_positions:
            if board.in_bounds(neighbor):
                var = board.get_variable(neighbor)
                neighbor_vars.append(var)

        # 添加约束：王后移动范围内格子的雷数等于count
        if neighbor_vars:
            model.Add(sum(neighbor_vars) == self.count)

    def check(self, board: 'AbstractBoard') -> bool:
        queen_positions = self._get_queen_positions(board)
        neighbor_types = [board.get_type(pos) for pos in queen_positions]
        f_num = neighbor_types.count("F")
        n_num = neighbor_types.count("N")

        # 检查当前雷数是否在合理范围内
        return f_num <= self.count <= f_num + n_num

    def method_choose(self) -> int:
        return 3
