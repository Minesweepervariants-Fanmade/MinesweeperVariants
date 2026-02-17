#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/06/29 16:12
# @Author  : Wu_RH
# @FileName: image_create.py
import math
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

    def safe_get_config(board_key: str, config_name: str, default=None):
        try:
            return board.get_config(board_key, config_name)
        except Exception:
            return default

    def get_hex_metrics(_cell_size: float):
        hex_size = _cell_size / math.sqrt(3)
        hex_width = 2 * hex_size
        hex_height = math.sqrt(3) * hex_size
        x_spacing = 1.5 * hex_size
        y_spacing = hex_height
        return hex_size, hex_width, hex_height, x_spacing, y_spacing

    def get_cell_box(pos, grid_type: str, x_offset: float, margin_top: float, _cell_size: float, metrics):
        r, c = pos.x, pos.y
        if grid_type == "hex":
            hex_size, hex_width, hex_height, x_spacing, y_spacing = metrics
            x_center = x_offset + c * x_spacing + hex_width / 2
            y_center = margin_top + r * y_spacing + (y_spacing / 2 if c % 2 == 0 else 0) + hex_height / 2
            x0 = x_center - _cell_size / 2
            y0 = y_center - _cell_size / 2
            return x0, y0, x_center, y_center
        x0 = x_offset + c * _cell_size
        y0 = margin_top + r * _cell_size
        x_center = x0 + _cell_size / 2
        y_center = y0 + _cell_size / 2
        return x0, y0, x_center, y_center

    def get_hex_points(x_center: float, y_center: float, hex_size: float):
        angles = [0, 60, 120, 180, 240, 300]
        return [
            (x_center + hex_size * math.cos(math.radians(a)),
             y_center + hex_size * math.sin(math.radians(a)))
            for a in angles
        ]

    def get_hex_neighbor_positions(pos, _board: AbstractBoard):
        x, y = pos.x, pos.y
        board_key = pos.board_key
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if y % 2 == 1:
            directions += [(-1, -1), (-1, 1)]
        else:
            directions += [(1, -1), (1, 1)]
        neighbors = []
        for dx, dy in directions:
            npos = type(pos)(x + dx, y + dy, board_key)
            if _board.in_bounds(npos):
                neighbors.append(npos)
        return neighbors

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
    def infer_grid_type(board_key: str):
        grid_type = safe_get_config(board_key, "grid_type", None)
        if grid_type:
            return grid_type
        for _, obj in board(mode="object", key=board_key):
            if obj is None:
                continue
            try:
                if obj.type() == b"3H":
                    return "hex"
            except Exception:
                continue
        return "square"

    configs = {k: {
        "by_mini": safe_get_config(k, "by_mini", False),
        "pos_label": safe_get_config(k, "pos_label", False),
        "row_col": safe_get_config(k, "row_col", False),
        "grid_type": infer_grid_type(k),
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
    pixel_sizes = {}
    for key in board_keys:
        br = board.boundary(key=key)
        cols = len(board.get_row_pos(br))
        rows = len(board.get_col_pos(br))
        sizes[key] = (rows, cols)
        grid_type = configs[key]["grid_type"]
        if grid_type == "hex":
            _, hex_width, _, x_spacing, y_spacing = get_hex_metrics(cell_size)
            board_width = (cols - 1) * x_spacing + hex_width if cols > 0 else 0
            board_height = rows * y_spacing + y_spacing / 2 if rows > 0 else 0
        else:
            board_width = cols * cell_size
            board_height = rows * cell_size
        pixel_sizes[key] = (board_width, board_height)

    total_width = int(sum(pixel_sizes[k][0] for k in board_keys) + (len(board_keys) + 1) * margin)
    max_height = max(pixel_sizes[k][1] for k in board_keys) if board_keys else 0
    total_height = int(margin + max_height + bottom_margin)

    image = Image.new("RGBA", (total_width, total_height), bg_color)
    draw = ImageDraw.Draw(image)

    x_offset = margin
    for key in board_keys:
        rows, cols = sizes[key]
        by_mini = configs[key]["by_mini"]
        pos_label = configs[key]["pos_label"]
        row_col = configs[key]["row_col"]
        grid_type = configs[key]["grid_type"]
        hex_metrics = get_hex_metrics(cell_size) if grid_type == "hex" else None

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
                if grid_type == "hex":
                    _, hex_width, _, x_spacing, _ = hex_metrics
                    x = x_offset + col * x_spacing + hex_width / 2
                else:
                    x = x_offset + col * cell_size + cell_size // 2
                y = margin / 2
                text = chr(64 + col // 26) if col > 25 else ''
                text += chr(65 + col % 26)
                draw.text((x, y), text, fill=text_color, font=axis_font, anchor="mm")

            for row in range(rows):
                x = x_offset - cell_size * 0.25
                if grid_type == "hex":
                    _, _, hex_height, _, y_spacing = hex_metrics
                    y = margin + row * y_spacing + (y_spacing / 2) + hex_height / 2
                else:
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
                if grid_type == "hex":
                    target_w = pixel_sizes[key][0]
                    target_h = pixel_sizes[key][1]
                else:
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

        line_width = CONFIG["grid_line"]["width"]
        stroke_px = max(1, int(cell_size * line_width))

        # 染色
        for pos, _ in board(key=key):
            x0, y0, x_center, y_center = get_cell_box(pos, grid_type, x_offset, margin, cell_size, hex_metrics)
            if board.get_dyed(pos):
                if grid_type == "hex":
                    base_hex_size = hex_metrics[0]
                    draw.polygon(get_hex_points(x_center, y_center, base_hex_size), fill=dye_color)
                else:
                    draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size], fill=dye_color)

        # 网格线
        if grid_type == "hex":
            for pos, _ in board(key=key):
                _, _, x_center, y_center = get_cell_box(pos, grid_type, x_offset, margin, cell_size, hex_metrics)
                base_hex_size = hex_metrics[0]
                points = get_hex_points(x_center, y_center, base_hex_size)
                edges = []
                for i in range(6):
                    p1 = points[i]
                    p2 = points[(i + 1) % 6]
                    mid_x = (p1[0] + p2[0]) / 2
                    mid_y = (p1[1] + p2[1]) / 2
                    dir_x = mid_x - x_center
                    dir_y = mid_y - y_center
                    length = math.hypot(dir_x, dir_y)
                    if length == 0:
                        continue
                    edges.append((p1, p2, dir_x / length, dir_y / length))

                neighbors = get_hex_neighbor_positions(pos, board)
                neighbor_dirs = []
                for npos in neighbors:
                    _, _, n_center_x, n_center_y = get_cell_box(npos, grid_type, x_offset, margin, cell_size, hex_metrics)
                    ndx = n_center_x - x_center
                    ndy = n_center_y - y_center
                    nlen = math.hypot(ndx, ndy)
                    if nlen == 0:
                        continue
                    neighbor_dirs.append((npos, ndx / nlen, ndy / nlen))

                for p1, p2, dir_x, dir_y in edges:
                    best_dot = -1.0
                    best_neighbor = None
                    for npos, ndx, ndy in neighbor_dirs:
                        dot = dir_x * ndx + dir_y * ndy
                        if dot > best_dot:
                            best_dot = dot
                            best_neighbor = npos
                    if best_dot >= 0.85:
                        if (pos.x, pos.y, pos.board_key) < (best_neighbor.x, best_neighbor.y, best_neighbor.board_key):
                            draw.line([p1, p2], fill=grid_color, width=stroke_px)
                    else:
                        draw.line([p1, p2], fill=grid_color, width=stroke_px)
        else:
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
                _, _, x, y = get_cell_box(pos, grid_type, x_offset, margin, cell_size, hex_metrics)
                draw.text((x, y), txt, fill=pos_label_color, font=label_font, anchor="mm")

        # 内容渲染 - 使用ElementRenderer
        for pos, obj in board(mode="object", key=key):
            x0_cell, y0_cell, _, _ = get_cell_box(pos, grid_type, x_offset, margin, cell_size, hex_metrics)
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
                x0_cell, y0_cell, _, _ = get_cell_box(pos, grid_type, x_offset, margin, cell_size, hex_metrics)
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

        x_offset += pixel_sizes[key][0] + margin

    # 底部文本
    if bottom_text:
        bottom_y = margin + max_height
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
