#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/08/07 12:50
# @Author  : Wu_RH
# @FileName: config.py.py

DEFAULT_CONFIG = {
    # 图片生成默认参数
    "output_file": "output",       # 默认的img文件名(不含后缀)
    "cell_size": 100,              # 一个单元格的尺寸
    "white_base": False,           # 默认黑底
    "board_class": None,           # 调用默认版本最高的board_class
    "image_debug": False,          # 生成图片时启用debug显示

    # 题板生成默认参数
    "seed": -1,                    # 随机种子
    "logger_lv": "INFO",           # 默认logger为info级别
    "attempts": -1,                # 默认无限次尝试（无-q参数默认20次）
    "total": -1,                   # 默认自动计算总雷数数量
    "dye": "",                     # 默认无染色
    "used_r": False,               # 默认不启用R推理
    "no_image": False,             # 默认生成图片

    "output_path": ".\\output",   # 保存路径，默认为工作目录下的output
    "timeout": 0,                 # 求解器超时时间（秒），0为无限制
    "workes_number": 12           # 多线程数量
}

IMAGE_CONFIG = config = {
    # 背景色设置
    "background": {
        "white": "#FFFFFF",
        "black": "#000000"
    },

    # 网格线设置
    "grid_line": {
        "white_bg": "#000000",
        "black_bg": "#FFFFFF",
        "width": 0.032
    },

    # 文本设置
    "text": {
        "black": "#FFFFFF",
        "white": "#000000",
        "anchor": "mm"
    },

    # 染色区域
    "dye": {
        "white_bg": "#B3B3B3",
        "black_bg": "#4C4C4C"
    },

    # 描边颜色
    "stroke": {
        "white_bg": "#808080",
        "black_bg": "#D3D3D3"
    },

    # 位置标签（例如 X=E）
    "pos_label": {
        # 标签X=E设置
        "white_bg": "#808080",
        "black_bg": "#808080",
        "size": 0.25
    },

    "assets": "assets",

    # 字体设置
    "font": {
        "name": "CopperplateCC-Heavy.ttf"
    },

    # 边距设置
    "margin": {
        "top_left_right_ratio": 0.7,
        "bottom_ratio": 0.7
    },

    # 坐标轴标签字体比例
    "axis_label": {
        "font_ratio": 0.5
    },

    # 角标尺寸
    "corner": {
        "mini_font_ratio": 0.23
    }
}

PUZZLE_CONFIG = {
    'delimiter': ':',    # 规则键入data的分割符
    'boardRuleDelimiter': ['(', ')']
}
