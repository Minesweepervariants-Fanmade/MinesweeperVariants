#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/06/29 16:12
# @Author  : Wu_RH
# @FileName: image_create.py
import os
import pathlib

from .tool import get_logger
from .. import __path__ as basepath
from ..abs.board import AbstractBoard
from ..config.config import IMAGE_CONFIG, DEFAULT_CONFIG
import minesweepervariants


def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def get_text(
    text: str,
    width: float = "auto",
    height: float = "auto",
    cover_pos_label: bool = True,
    color: tuple[str, str] = ("#FFFFFF", "#000000"),
    dominant_by_height: bool = True,
    style: str = "",
):
    """
    :param text:文本内容
    :param width: 宽度
    :param height: 高度
    :param cover_pos_label: 覆盖格子内的X=N标识
    :param dominant_by_height: 高主导的对齐 否则宽主导
    :param color: 色号字符串#RRGGBB 第一个表示黑底 第二个表示白底 '#FFFFFF'表示白色
    :param style: (web) 样式内容
    """
    if dominant_by_height is None:
        dominant = None
    else:
        dominant = "height" if dominant_by_height else "width"
    return {
        "type": "text",
        "text": text,
        "content": text,
        'color_black': _hex_to_rgb(color[0]),
        'color_white': _hex_to_rgb(color[1]),
        'width': width,
        'height': height,
        "font_size": 1,
        "cover": cover_pos_label,
        "dominant": dominant,
        "style": style,
    }


def get_image(
    image_path: str,
    image_width: float = "auto",
    image_height: float = "auto",
    cover_pos_label: bool = True,
    dominant_by_height: bool = True,
    style: str = "",
):
    """
    :param image_path:图片在data下的路径位置
    :param image_width:图片的水平缩放比例
    :param image_height:图片的垂直缩放比例
    :param cover_pos_label:是否覆盖X=N标识
    :param dominant_by_height: 高主导的对齐 否则宽主导
    :param style: (web) 样式内容
    """
    if dominant_by_height is None:
        dominant = None
    else:
        dominant = "height" if dominant_by_height else "width"
    return {
        'type': 'image',
        'image': image_path,  # 图片对象
        'height': image_height,  # 高度（单元格单位或auto）
        'width': image_width,   # 宽度（单元格单位或auto）
        "cover": cover_pos_label,
        "dominant": dominant,
        "style": style,
    }


def get_row(
    *args,
    spacing=0,
    dominant_by_height=True
):
    """
    水平排列元素
    :param args: 子元素列表
    :param spacing: 每个元素之间的间距值
    :param dominant_by_height: 高主导的对齐 否则宽主导
    """
    if dominant_by_height is None:
        dominant = None
    else:
        dominant = "height" if dominant_by_height else "width"
    for child in args:
        if child["dominant"] is None:
            child["dominant"] = "height"
    height = [e["height"] for e in args if type(e["height"]) is int]
    if height:
        height = max(height)
    else:
        height = "auto"
    return {
        "type": "row",
        "children": args,
        "spacing": spacing,
        "cover": all(e["cover"] for e in args),
        "height": height,
        "width": "auto",
        "dominant": dominant
    }


def get_col(
    *args,
    spacing=0,
    dominant_by_height=False
):
    """
    水平排列元素
    :param args: 子元素列表
    :param spacing: 每个元素之间的间距值
    :param dominant_by_height: 高主导的对齐 否则宽主导
    """
    if dominant_by_height is None:
        dominant = None
    else:
        dominant = "height" if dominant_by_height else "width"
    for child in args:
        if child["dominant"] is None:
            child["dominant"] = "width"
    width = [e["width"] for e in args if type(e["width"]) is int]
    if width:
        width = max(width)
    else:
        width = "auto"
    return {
        "type": "col",
        "children": args,
        "spacing": spacing,
        "cover": all(e["cover"] for e in args),
        "height": "auto",
        "width": width,
        "dominant": dominant
    }


def get_dummy(
        width: float = 0.01,
        height: float = 0.01
) -> object:
    """
    创建占位符元素

    :param width: 宽度（单元格单位）
    :param height: 高度（单元格单位）
    """
    return {
        "type": "placeholder",
        "width": width,
        "height": height,
        "cover": True,
        "dominant": None
    }


def draw_board(
        board: AbstractBoard,
        background_white: bool = None,
        bottom_text: str = "",
        cell_size: int = 100,
        output="output"
) -> bytes:
    """
    绘制多个题板图像，支持横向拼接。
    :param board: AbstractBoard 实例，支持 get_board_keys。
    :param background_white: 是否白底。
    :param bottom_text: 底部文字。
    :param cell_size: 单元格大小。
    :param output: 输出文件名（不含扩展名）。
    """
    from PIL import Image, ImageDraw, ImageFont
    from .element_renderer import Renderer

    CONFIG = {}

    CONFIG.update(DEFAULT_CONFIG)
    CONFIG.update(IMAGE_CONFIG)
    CONFIG["output_path"] = DEFAULT_CONFIG["output_path"]

    if background_white is None:
        background_white = CONFIG["white_base"]

    def load_font(size: int) -> ImageFont.FreeTypeFont:
        path = pathlib.Path(basepath[0])
        path /= CONFIG["assets"]
        path /= CONFIG["font"]["name"]
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            return ImageFont.load_default()

    def cfg(_path):
        return _hex_to_rgb(CONFIG[_path]["white_bg" if background_white else "black_bg"])

    def int_to_roman(num: int) -> str:
        # 定义数值与罗马符号的映射表（按数值降序排列）
        val_symbols = [
            (1000, "M"),
            (900, "CM"),
            (500, "D"),
            (400, "CD"),
            (100, "C"),
            (90, "XC"),
            (50, "L"),
            (40, "XL"),
            (10, "X"),
            (9, "IX"),
            (5, "V"),
            (4, "IV"),
            (1, "I")
        ]

        _roman = []  # 存储结果字符
        for _value, symbol in val_symbols:
            # 当剩余数字大于等于当前值
            while num >= _value:
                num -= _value  # 减去当前值
                _roman.append(symbol)  # 添加对应符号
            if num == 0:  # 提前终止
                break
        return "".join(_roman)

    board_keys = board.get_board_keys()
    configs = {k: {
        "by_mini": board.get_config(k, "by_mini"),
        "pos_label": board.get_config(k, "pos_label"),
        "row_col": board.get_config(k, "row_col"),
    } for k in board_keys}

    margin_ratio = CONFIG["margin"]["top_left_right_ratio"]
    bottom_ratio = CONFIG["margin"]["bottom_ratio"]
    axis_ratio = CONFIG["axis_label"]["font_ratio"]
    mini_ratio = CONFIG["corner"]["mini_font_ratio"]

    # 色彩配置
    bg_color = _hex_to_rgb(CONFIG["background"]["white" if background_white else "black"])
    text_color = _hex_to_rgb(CONFIG["text"]["white" if background_white else "black"])
    grid_color = cfg("grid_line")
    dye_color = cfg("dye")
    stroke_color = cfg("stroke")
    pos_label_color = cfg("pos_label")

    margin = cell_size * margin_ratio
    bottom_margin = cell_size * bottom_ratio

    sizes = {}
    for key in board_keys:
        br = board.boundary(key=key)
        rows = len(board.get_row_pos(br))
        cols = len(board.get_col_pos(br))
        sizes[key] = (cols, rows)

    total_width = int(sum(c for _, c in sizes.values()) * cell_size + (len(board_keys) + 1) * margin)
    max_rows = max(r for r, _ in sizes.values())
    total_height = int(margin + max_rows * cell_size + bottom_margin)

    image = Image.new("RGBA", (total_width, total_height), bg_color)
    draw = ImageDraw.Draw(image)

    x_offset = margin
    for key in board_keys:
        rows, cols = sizes[key]
        by_mini = configs[key]["by_mini"]
        pos_label = configs[key]["pos_label"]
        row_col = configs[key]["row_col"]

        # 题板左上角编号
        if len(board_keys) > 2:
            roman = int_to_roman(board_keys.index(key) + 1)

            roman_margin = margin * 1.3
            max_w = roman_margin  # 目标最大宽度不应超过 margin 区域
            low, high = 8, int(roman_margin * 0.63)
            best = low
            while low <= high:
                mid = (low + high) // 2
                font = load_font(mid)
                if font.getlength(roman) <= max_w:
                    best = mid
                    low = mid + 1
                else:
                    high = mid - 1

            font = load_font(best)

            center_x = x_offset - cell_size * 0.3

            paste_y = int(margin * 0.5)
            draw.text((center_x, paste_y),
                      roman,
                      fill=text_color,
                      font=font,
                      anchor="mm",
                      stroke_width=1 if best <= 14 else 0,
                      stroke_fill=stroke_color)

        # 坐标轴标签
        if row_col:
            axis_font_size = int(cell_size * axis_ratio)
            axis_font = load_font(axis_font_size)

            for col in range(cols):
                x = x_offset + col * cell_size + cell_size // 2
                y = margin / 2
                text = chr(64 + col // 26) if col > 25 else ''
                text += chr(65 + col % 26)
                draw.text((x, y), text, fill=text_color, font=axis_font, anchor="mm")

            for row in range(rows):
                x = x_offset - cell_size * 0.25
                y = margin + row * cell_size + cell_size / 2
                draw.text((x, y), str(row + 1), fill=text_color, font=axis_font, anchor="mm")

        # 绘制背景图片（如果配置指定）
        try:
            bg_cfg = CONFIG.get("background", {})
            img_name = None

            if isinstance(bg_cfg, dict):
                img_name = bg_cfg.get("image")
                opacity = bg_cfg.get("opacity", 1.0)
            else:
                raise ValueError("image not found")

            if img_name:
                if not img_name.startswith(("http://","https://")):
                    img_path = pathlib.Path(minesweepervariants.__path__[0]) / CONFIG["assets"] / img_name
                    if img_path.exists():
                        bg_img = Image.open(img_path)
                    else:
                        raise FileNotFoundError("background image not found")
                else:
                    import requests
                    from io import BytesIO
                    resp = requests.get(img_name, timeout=10)
                    resp.raise_for_status()
                    bg_img = Image.open(BytesIO(resp.content))

                bg_img = bg_img.convert("RGBA")
                target_w = cols * cell_size
                target_h = rows * cell_size

                iw, ih = bg_img.size

                scale = max(target_w / iw, target_h / ih)
                if scale <= 0:
                    raise ValueError("invalid background scale computed")

                new_size = (max(1, int(iw * scale)), max(1, int(ih * scale)))
                bg_img = bg_img.resize(new_size)

                # 居中裁剪到目标大小（若图片比目标小则会在边缘截取）
                left = max(0, (bg_img.width - target_w) // 2)
                top = max(0, (bg_img.height - target_h) // 2)
                bg_img = bg_img.crop((left, top, left + target_w, top + target_h))

                # 应用透明度
                if opacity < 1.0:
                    print(opacity)
                    alpha = bg_img.split()[3].point(lambda p: int(p * opacity))
                    bg_img.putalpha(alpha)

                image.paste(bg_img, (int(x_offset), int(margin)), bg_img)
        except Exception as exc:
            get_logger().error(f"Failed to draw background image: {exc}")

        # 染色
        for pos, _ in board(key=key):
            r, c = pos.x, pos.y
            x0 = x_offset + c * cell_size
            y0 = margin + r * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            if board.get_dyed(pos):
                draw.rectangle([x0, y0, x1, y1], fill=dye_color)

        # 网格线
        line_width = CONFIG["grid_line"]["width"]
        for r in range(rows + 1):
            y = margin + r * cell_size
            draw.line([(int(x_offset - cell_size * (line_width * 0.3)), y),
                       (int(x_offset + cols * cell_size + cell_size * (line_width * 0.3)), y)],
                      fill=grid_color, width=int(cell_size * line_width))
        for c in range(cols + 1):
            x = x_offset + c * cell_size
            draw.line([(x, int(margin - cell_size * (line_width * 0.3))),
                       (x, int(margin + rows * cell_size + cell_size * (line_width * 0.3)))],
                      fill=grid_color, width=int(cell_size * line_width))

        # X=N标签
        if pos_label:
            label_font = load_font(int(cell_size * CONFIG["pos_label"]["size"]))
            for pos, obj in board(mode="object", key=key):
                if board.get_type(pos, special='raw') == "C":
                    continue
                txt = board.pos_label(pos)
                if not txt:
                    continue
                r, c = pos.x, pos.y
                x = x_offset + c * cell_size + cell_size / 2
                y = margin + r * cell_size + cell_size / 2
                draw.text((x, y), txt, fill=pos_label_color, font=label_font, anchor="mm")

        # 内容渲染 - 使用ElementRenderer
        for pos, obj in board(mode="object", key=key):
            r, c = pos.x, pos.y
            x0_cell = x_offset + c * cell_size
            y0_cell = margin + r * cell_size
            value = board.get_value(pos)
            if value is None:
                continue

            # 创建元素渲染器
            renderer = Renderer(
                cell_size=cell_size,
                background_white=background_white,
                origin=(x0_cell, y0_cell),
                font_path=CONFIG["font"]["name"],
                assets=CONFIG["assets"]
            )

            renderer.render(image, value.compose(board))

        # 渲染角标
        if by_mini:
            for pos, obj in board(mode="object", key=key):
                r, c = pos.x, pos.y
                x0_cell = x_offset + c * cell_size
                y0_cell = margin + r * cell_size
                x1_cell = x0_cell + cell_size
                y1_cell = y0_cell + cell_size
                value = board.get_value(pos)
                if value is None:
                    continue

                if type(obj) is type(board.get_config(key, "VALUE")):
                    continue
                if type(obj) is type(board.get_config(key, "MINES")):
                    continue

                mini_font = load_font(int(cell_size * mini_ratio))
                draw.text((x1_cell - cell_size * 0.02, y1_cell + cell_size * 0.05),
                          value.tag(board).decode('utf-8', 'ignore'),
                          fill=text_color, font=mini_font, anchor="rd")

        x_offset += cols * cell_size + margin

    # 底部文本
    if bottom_text:
        bottom_y = margin + max_rows * cell_size
        max_w = total_width
        low, high = 8, int(bottom_margin * 0.63)
        best = low
        while low <= high:
            mid = (low + high) // 2
            font = load_font(mid)
            if font.getlength(bottom_text) <= max_w:
                best = mid
                low = mid + 1
            else:
                high = mid - 1

        font = load_font(best)
        h = font.getbbox(bottom_text)[3] - font.getbbox(bottom_text)[1]
        y = bottom_y + (bottom_margin / 2) + (h / 4)
        y = min(y, total_height - h / 2)
        draw.text((total_width / 2, y),
                  bottom_text,
                  fill=text_color,
                  font=font,
                  anchor="ms",
                  stroke_width=1 if best <= 14 else 0,
                  stroke_fill=stroke_color)

    if not os.path.exists(CONFIG["output_path"]):
        os.makedirs(CONFIG["output_path"])
    filepath = os.path.join(CONFIG["output_path"], f"{output}.png")
    image.save(filepath)
    get_logger().info(f"Image saved to: {filepath}")

    with open(filepath, "rb") as f:  # 'rb' 表示二进制读取
        image_bytes = f.read()  # 直接获取字节数据
    return image_bytes
