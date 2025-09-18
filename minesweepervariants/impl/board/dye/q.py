#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/06/10 09:52
# @Author  : xxx
# @FileName: c.py

from . import AbstractDye


class DyeC(AbstractDye):
    name = "q" # quadrant
    fullname = "2x2棋盘格染色"
    doc =   "args: [~][[x][,[y]];][w][,[h]][;[px][,[py]]][;[bx][,[by]]]\n" \
            "bbox(x,y;w,h;px,py;bx,by)\n" \
            "~: 反转染色\n" \
            "x: x偏移 y: y偏移\n" \
            "w: 宽度 h: 高度\n" \
            "px: x方向的间隔 py: y方向的间隔\n" \
            "bx: x方向的边界 by: y方向的边界\n"

    @staticmethod
    def parse_pair(value: str, default: tuple[int, int] = (0, 0)):
        match value.count(','):
            case 0:
                return (int(value or default[0]), int(value or default[1]))
            case 1:
                i = value.index(',')
                return (int(value[:i] or default[0]), int(value[i + 1:] or default[1]))
            case _:
                raise ValueError("Invalid pair format")

    def __init__(self, args):

        _args = args
        if args.startswith('~'):
            self.inv = True
            args = args[1:]
        else:
            self.inv = False

        if args == '':
            self.x = self.y = 0
            self.w = self.h = 2
            self.px = self.py = 0
            self.bx = self.by = 0
            self.sx = self.sy = 0
            return

        all_args = args.split(';')
        match len(all_args):
            case 1:
                self.x = self.y = 0
                self.w , self.h = self.parse_pair(all_args[0], (0, 0))
                self.px = self.py = 0
                self.bx = self.by = 0
                self.sx = self.sy = 0
                return
            case 2:
                self.x, self.y = self.parse_pair(all_args[0], (0, 0))
                self.w, self.h = self.parse_pair(all_args[1], (2, 2))
                self.px = self.py = 0
                self.bx = self.by = 0
                self.sx = self.sy = 0
                return
            case 3:
                self.x, self.y = self.parse_pair(all_args[0], (0, 0))
                self.w, self.h = self.parse_pair(all_args[1], (2, 2))
                self.px, self.py = self.parse_pair(all_args[2], (0, 0))
                self.bx = self.by = 0
                self.sx = self.sy = 0
                return
            case 4:
                self.x, self.y = self.parse_pair(all_args[0], (0, 0))
                self.w, self.h = self.parse_pair(all_args[1], (2, 2))
                self.px, self.py = self.parse_pair(all_args[2], (0, 0))
                self.bx, self.by = self.parse_pair(all_args[3], (0, 0))
            case _:
                raise ValueError(f"Invalid bbox format {_args}")

    def dye(self, board):
        dye = True
        for key in board.get_interactive_keys():
            dye = not dye
            for pos, _ in board(key=key):
                a1 = (pos.x + self.x) // (self.w + self.px + self.bx) % 2
                a2 = (pos.y + self.y) // (self.h + self.py + self.by) % 2
                b1 = (pos.x + self.x) % (self.w + self.px + self.bx) < self.w
                b2 = (pos.x + self.x) % (self.w + self.px + self.bx) >= self.w + self.px
                c1 = (pos.y + self.y) % (self.h + self.py + self.bx) < self.h
                c2 = (pos.y + self.y) % (self.h + self.py + self.bx) >= self.h + self.py

                A = (a1 ^ a2)
                B = (A and b1) or b2
                C = (A and c1) or c2

                _dye = dye ^ (B and C) ^ self.inv
                board.set_dyed(pos, _dye)
