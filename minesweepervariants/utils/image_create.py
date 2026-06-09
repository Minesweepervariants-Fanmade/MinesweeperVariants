#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2025/06/29 16:12
# @Author  : Wu_RH
# @FileName: image_create.py
import math
import os
import pathlib
from typing import Callable

from minesweepervariants.position import Position
from minesweepervariants.size import Size
from minesweepervariants.utils.image_template import get_dummy, get_text, hex_to_rgb, get_col

from .tool import get_logger
from .. import __path__ as basepath
from minesweepervariants.board import Board
from ..config.config import IMAGE_CONFIG, DEFAULT_CONFIG
import minesweepervariants



def register_final_image_postprocess_callback(callback: Callable, key: str = None):
    """注册最终题板图后处理回调。回调签名: fn(image, board=None, config=None) -> Image|None"""
    if key:
        callback_map = IMAGE_CONFIG.setdefault("final_image_postprocess_callback_map", {})
        callback_map[key] = callback
        return
    callbacks = IMAGE_CONFIG.setdefault("final_image_postprocess_callbacks", [])
    if callback not in callbacks:
        callbacks.append(callback)


def _apply_final_image_postprocess_callbacks(image, board, config):
    callbacks = list(IMAGE_CONFIG.get("final_image_postprocess_callbacks", []))
    callback_map = IMAGE_CONFIG.get("final_image_postprocess_callback_map", {})
    if isinstance(callback_map, dict):
        callbacks.extend(callback_map.values())
    for callback in callbacks:
        try:
            result = callback(image, board=board, config=config)
            if result is not None:
                image = result
        except Exception as exc:
            get_logger().error(f"Final image postprocess callback failed: {exc}")
    return image



def draw_board(
        board: Board,
        background_white: bool = None,
        bottom_text: str = "",
        cell_size: int = 100,
        output="output",
        hint_because: list[Position] = None,
        hint_deduced: list[Position] = None,
) -> bytes:
    """
    绘制多个题板图像，支持横向拼接。
    :param board: Board 实例，支持 get_board_keys。
    :param background_white: 是否白底。
    :param bottom_text: 底部文字。
    :param cell_size: 单元格大小。
    :param output: 输出文件名（不含扩展名）。
    :param hint_because: 提示的由于格子列表
    :param hint_deduced: 提示的导致格子列表
    """
    from PIL import Image, ImageDraw, ImageFont
    from .element_renderer import Renderer

    CONFIG = {}

    CONFIG.update(DEFAULT_CONFIG)
    CONFIG.update(IMAGE_CONFIG)
    CONFIG["output_path"] = DEFAULT_CONFIG["output_path"]

    if background_white is None:
        background_white = CONFIG["white_base"]

    if hint_because is None:
        hint_because = []
    if hint_deduced is None:
        hint_deduced = []

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
        r, c = pos.row, pos.col
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

    def get_hex_neighbor_positions(pos, _board: Board):
        col, row = pos.col, pos.row
        board_key = pos.board_key
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if row % 2 == 1:
            directions += [(-1, -1), (-1, 1)]
        else:
            directions += [(1, -1), (1, 1)]
        neighbors = []
        for dx, dy in directions:
            npos = type(pos)(col + dx, row + dy, board_key)
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
        return hex_to_rgb(CONFIG[_path]["white_bg" if background_white else "black_bg"])

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
    bg_color = hex_to_rgb(CONFIG["background"]["white" if background_white else "black"])
    text_color = hex_to_rgb(CONFIG["text"]["white" if background_white else "black"])
    grid_color = cfg("grid_line")
    dye_color = cfg("dye")
    stroke_color = cfg("stroke")
    pos_label_color = cfg("pos_label")

    margin = cell_size * margin_ratio
    bottom_margin = cell_size * bottom_ratio

    sizes = {}
    pixel_sizes = {}
    for key in board_keys:
        br: Position = board.boundary(key=key)
        cols = len(board.get_row_pos(br))
        rows = len(board.get_col_pos(br))
        sizes[key] = Size(cols=cols, rows=rows)
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
        cols, rows = sizes[key]
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
                if not img_name.startswith(("http://", "https://")):
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

        # ========== 使用混合模式 + 动态采样底色 ==========
        def blend_multiply(base_rgb, overlay_rgb):
            return tuple((b * o) // 255 for b, o in zip(base_rgb, overlay_rgb))

        def blend_screen(base_rgb, overlay_rgb):
            return tuple(int(255 - ((255 - b) * (255 - o * 0.5)) // 255) for b, o in zip(base_rgb, overlay_rgb))

        BLEND_MODE = blend_screen  # 根据需要选择

        # 处理 hint_because
        if hint_because:
            bc_color_hex = CONFIG["hint_because"]["white_bg" if background_white else "black_bg"]
            overlay_rgb = hex_to_rgb(bc_color_hex)

            for pos in hint_because:
                if getattr(pos, 'board_key', None) != key:
                    continue

                # 计算格子中心坐标
                x0, y0, x_center, y_center = get_cell_box(
                    pos, grid_type, x_offset, margin, cell_size, hex_metrics
                )

                # 从当前图像中采样该点的实际颜色（RGBA → RGB）
                pixel = image.getpixel((int(x_center), int(y_center)))
                base_rgb = pixel[:3]  # 忽略 Alpha

                # 混合得到最终颜色（不透明）
                mixed_rgb = BLEND_MODE(base_rgb, overlay_rgb)

                if grid_type == "hex":
                    points = get_hex_points(x_center, y_center, hex_metrics[0])
                    draw.polygon(points, fill=mixed_rgb)
                else:
                    draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size], fill=mixed_rgb)

        # 处理 hint_deduced —— 同理
        if hint_deduced:
            dc_color_hex = CONFIG["hint_deduced"]["white_bg" if background_white else "black_bg"]
            overlay_rgb = hex_to_rgb(dc_color_hex)

            for pos in hint_deduced:
                if getattr(pos, 'board_key', None) != key:
                    continue

                x0, y0, x_center, y_center = get_cell_box(
                    pos, grid_type, x_offset, margin, cell_size, hex_metrics
                )
                pixel = image.getpixel((int(x_center), int(y_center)))
                base_rgb = pixel[:3]
                mixed_rgb = BLEND_MODE(base_rgb, overlay_rgb)

                if grid_type == "hex":
                    points = get_hex_points(x_center, y_center, hex_metrics[0])
                    draw.polygon(points, fill=mixed_rgb)
                else:
                    draw.rectangle([x0, y0, x0 + cell_size, y0 + cell_size], fill=mixed_rgb)

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
                    _, _, n_center_x, n_center_y = get_cell_box(npos, grid_type, x_offset, margin, cell_size,
                                                                hex_metrics)
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
                        if (pos.col, pos.row, pos.board_key) < (
                        best_neighbor.col, best_neighbor.row, best_neighbor.board_key):
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
            if value is None and pos not in hint_deduced:
                continue

            # 创建元素渲染器
            renderer = Renderer(
                cell_size=cell_size,
                background_white=background_white,
                origin=(x0_cell, y0_cell),
                font_path=CONFIG["font"]["name"],
                assets=CONFIG["assets"]
            )

            if value is not None:
                renderer.render(image, value.compose(board))
            else:
                renderer.render(image, get_col(
                    get_dummy(height=0.3),
                    get_text("!"),
                    get_dummy(height=0.3)
                ))

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

    # 最终题板图后处理回调（例如规则注入A/B融合并直接替换最终图）
    image = _apply_final_image_postprocess_callbacks(image, board, CONFIG)

    if not os.path.exists(CONFIG["output_path"]):
        os.makedirs(CONFIG["output_path"])
    filepath = os.path.join(CONFIG["output_path"], f"{output}.png")
    image.save(filepath)
    get_logger().info(f"Image saved to: {filepath}\n", end="")

    with open(filepath, "rb") as f:  # 'rb' 表示二进制读取
        image_bytes = f.read()  # 直接获取字节数据
    return image_bytes
