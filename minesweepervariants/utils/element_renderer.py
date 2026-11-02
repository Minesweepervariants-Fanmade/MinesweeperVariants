import os
import pathlib
import traceback

from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple

import minesweepervariants
from minesweepervariants.utils.tool import get_logger
from minesweepervariants.utils.font_stack import (
    FontStack,
    draw_text as draw_stack_text,
    resolve_fallback_fonts,
)

from PIL import Image, ImageDraw, ImageFont

# 同一渲染进程内大量复用同尺寸字体，缓存 FontStack 避免重复解析 cmap
_FONT_STACK_CACHE: Dict[Tuple[str, str, Tuple[str, ...], int], FontStack] = {}


def _get_font_stack(
    primary: pathlib.Path, assets: str, size: int,
    fallback_paths: Optional[Tuple[str, ...]] = None,
) -> FontStack:
    """按 (主字体, 资源目录, 回退字体集合, 字号) 缓存 FontStack。"""
    if not fallback_paths:
        # 未配置或配置全部失效时自动探测，避免退化为“无回退”
        fallback_paths = tuple(str(p) for p in resolve_fallback_fonts(assets))
    key = (str(primary), assets, tuple(fallback_paths), int(size))
    cached = _FONT_STACK_CACHE.get(key)
    if cached is not None:
        return cached

    stack = FontStack.load(primary, size, fallback_paths)
    _FONT_STACK_CACHE[key] = stack
    return stack


class Renderer:
    def __init__(self, cell_size: float, background_white: bool,
                 origin: Tuple[float, float], font_path: str,
                 assets: str, debug=False,
                 fallback_fonts: Optional[Sequence[str]] = None):
        from PIL import Image, ImageDraw, ImageFont

        self.cell_size = cell_size
        self.background_white = background_white
        self.origin = origin
        self.assets_path = assets
        self.font_path = pathlib.Path(minesweepervariants.__path__[0])
        self.font_path /= assets
        self.font_path /= font_path

        # 解析回退字体：显式配置优先，否则自动探测
        if fallback_fonts:
            resolved: list[str] = []
            for item in fallback_fonts:
                p = pathlib.Path(item)
                if not p.is_absolute():
                    # 相对 assets 目录
                    p = pathlib.Path(minesweepervariants.__path__[0]) / assets / p
                if p.is_file():
                    resolved.append(str(p))
                else:
                    get_logger().warning(f"[Renderer] 回退字体不存在，已忽略: {p}")
            self.fallback_fonts: tuple[str, ...] = tuple(resolved)
        else:
            self.fallback_fonts = tuple(str(p) for p in resolve_fallback_fonts(assets))

        try:
            self._font(1)
        except Exception as e:
            get_logger().error(traceback.format_exc())
            get_logger().error("Font loading failed. Path: {}".format(self.font_path))
            get_logger().error("cwd: {}".format(os.getcwd()))
            raise ValueError

        # 创建用于文本测量的临时图像和绘图对象
        self.temp_img = Image.new('RGB', (1, 1))
        self.temp_draw = ImageDraw.Draw(self.temp_img)
        self.element_sizes = {}

        self.debug = debug

    def _font(self, size: int) -> FontStack:
        """获取指定字号的字体栈（带缓存）。"""
        return _get_font_stack(
            self.font_path, self.assets_path, size, self.fallback_fonts
        )

    def render(self, image: 'Image.Image', element: Dict):
        abs_x, abs_y = self.origin
        root_box = (abs_x, abs_y, self.cell_size, self.cell_size)
        self._render_element(image, element, root_box)

    def _render_element(self, image: 'Image.Image', element: Dict,
                        box: Tuple[float, float, float, float]):
        """
        递归渲染单个元素 - 添加调试边框

        :param image: PIL图像对象
        :param element: 元素字典
        :param box: 可用空间 (x, y, width, height) 单位像素
        """
        # 创建绘图对象用于调试
        draw = ImageDraw.Draw(image, 'RGBA')

        # 保存原始元素类型
        e_type = element['type']

        # 渲染前绘制容器边框（红色）
        box_x, box_y, box_w, box_h = box

        # 实际渲染元素
        if e_type == 'text':
            self._render_text(image, element, box)
        elif e_type == 'image':
            self._render_image(image, element, box)
        elif e_type == 'row':
            self._render_row(image, element, box)
        elif e_type == 'col':
            self._render_col(image, element, box)
        elif e_type == 'placeholder':  # 添加占位符渲染
            # 占位符不需要渲染内容，只占用空间
            pass

        if self.debug:
            # 渲染后绘制元素边框（蓝色）
            if e_type in ['text', 'image']:
                # 绘制元素边框
                draw.rectangle(
                    [box_x, box_y, box_x + box_w, box_y + box_h],
                    outline=(0, 0, 255, 255),  # 蓝色边框
                    width=int(self.cell_size * 0.01)
                )

            if e_type in ['row', 'col']:
                # 绘制容器边框
                draw.rectangle(
                    [box_x, box_y, box_x + box_w, box_y + box_h],
                    outline=(255, 0, 0, 255),  # 红色边框
                    width=int(self.cell_size * 0.01)
                )
            if e_type == 'placeholder':
                if box_h < 0:
                    box_y -= box_h
                    box_h *= -1
                if box_w < 0:
                    box_x -= box_w
                    box_w *= -1
                draw.rectangle(
                    [box_x, box_y, box_x + box_w, box_y + box_h],
                    outline=(255, 0, 0, 255),
                    width=int(self.cell_size * 0.01)
                )

    def _calculate_element_size(self, element: Dict, max_width: float, max_height: float) -> Tuple[float, float]:
        """带缓存和主导方向处理的尺寸计算 - 添加固定尺寸支持"""
        # 生成唯一缓存键
        cache_key = (id(element), max_width, max_height)

        # 检查缓存
        if cache_key in self.element_sizes:
            return self.element_sizes[cache_key]

        # 使用非常大的值代替 inf，但不超过合理范围
        VERY_LARGE = 10000  # 10,000像素，足够大但不会导致计算问题

        # === 新增：处理固定尺寸 ===
        fixed_width = element.get('width')
        fixed_height = element.get('height')

        # 处理固定宽度
        if fixed_width != 'auto':
            fixed_width = fixed_width * self.cell_size
            # 确保不超过约束
            fixed_width = min(fixed_width, max_width)

        # 处理固定高度
        if fixed_height != 'auto':
            fixed_height = fixed_height * self.cell_size
            # 确保不超过约束
            fixed_height = min(fixed_height, max_height)

        # 如果同时设置了固定宽度和高度
        if fixed_width != 'auto' and fixed_height != 'auto':
            size = (fixed_width, fixed_height)
            self.element_sizes[cache_key] = size
            return size

        # 根据元素类型计算尺寸
        e_type = element.get('type')
        if e_type == 'text':
            # 如果有固定宽度则使用固定宽度
            width_constraint = fixed_width if fixed_width != 'auto' else min(max_width, VERY_LARGE)
            # 如果有固定高度则使用固定高度
            height_constraint = fixed_height if fixed_height != 'auto' else min(max_height, VERY_LARGE)
            size = self._calculate_text_size(
                element,
                width_constraint,
                height_constraint
            )
        elif e_type == 'image':
            # 如果有固定尺寸则使用固定尺寸
            width_constraint = fixed_width if fixed_width != 'auto' else min(max_width, VERY_LARGE)
            height_constraint = fixed_height if fixed_height != 'auto' else min(max_height, VERY_LARGE)
            size = self._calculate_image_size(
                element,
                width_constraint,
                height_constraint
            )
        elif e_type == 'row':
            # 如果有固定尺寸则使用固定尺寸
            width_constraint = fixed_width if fixed_width != 'auto' else min(max_width, VERY_LARGE)
            height_constraint = fixed_height if fixed_height != 'auto' else min(max_height, VERY_LARGE)
            size = self._calculate_row_size(
                element,
                width_constraint,
                height_constraint
            )
        elif e_type == 'col':
            # 如果有固定尺寸则使用固定尺寸
            width_constraint = fixed_width if fixed_width != 'auto' else min(max_width, VERY_LARGE)
            height_constraint = fixed_height if fixed_height != 'auto' else min(max_height, VERY_LARGE)
            size = self._calculate_col_size(
                element,
                width_constraint,
                height_constraint
            )
        elif e_type == 'placeholder':  # 添加占位符支持
            width = element['width'] * self.cell_size
            height = element['height'] * self.cell_size
            # 确保不超过约束
            width = min(width, max_width)
            height = min(height, max_height)
            size = (width, height)
        else:
            size = (0, 0)

        # 缓存结果
        self.element_sizes[cache_key] = size
        return size

    def _calculate_text_size(self, element: Dict, max_width: float, max_height: float) -> Tuple[float, float]:
        """精确计算文本在给定约束下的实际尺寸"""
        text = element['text']
        if not text:
            return 0, 0

        # 寻找最佳字体大小
        min_font_size = 1
        max_font_size = int(max_height * 2)  # 最大字体大小不超过高度的2倍
        best_font_size = min_font_size
        best_width = 0
        best_height = 0

        # 使用二分查找找到最佳字体大小（测量走字体栈，保证缺字回退后宽度正确）
        low, high = min_font_size, max_font_size
        while low <= high:
            mid = (low + high) // 2
            font = self._font(mid)
            bbox = font.getbbox(text, anchor="la")
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            if width <= max_width and height <= max_height:
                # 当前字体大小可行，尝试更大的
                best_font_size = mid
                best_width = width
                best_height = height
                low = mid + 1
            else:
                # 当前字体大小太大，尝试更小的
                high = mid - 1

        # 如果没有找到可行的大小，使用最小字号测量结果
        if best_font_size == min_font_size and best_width == 0:
            font = self._font(min_font_size)
            bbox = font.getbbox(text, anchor="la")
            best_width = bbox[2] - bbox[0]
            best_height = bbox[3] - bbox[1]

        return best_width, best_height

    def _calculate_image_size(self, element: Dict, max_width: float, max_height: float) -> Tuple[float, float]:
        """计算图像尺寸 - 考虑主导方向"""
        path = pathlib.Path(minesweepervariants.__path__[0])
        path /= self.assets_path
        path /= f"{element['image']}.png"
        if not os.path.exists(path):
            return 0.0, 0.0
        img = Image.open(path).convert("RGBA")
        orig_width, orig_height = img.size
        aspect_ratio = orig_width / orig_height

        if (element.get("height") != "auto" and
                element.get("width") != "auto"):
            return (self.cell_size * element.get("width"),
                    self.cell_size * element.get("height"))

        # 根据主导方向计算
        if element.get('dominant') == 'height':
            # 高度主导 - 高度设为最大高度
            child_height = min(max_height, orig_height)
            child_width = child_height * aspect_ratio

            # 如果宽度超过约束，则等比例缩小
            if child_width > max_width:
                scale = max_width / child_width
                child_width = max_width
                child_height = child_height * scale
        else:
            # 宽度主导 - 宽度设为最大宽度
            child_width = min(max_width, orig_width)
            child_height = child_width / aspect_ratio

            # 如果高度超过约束，则等比例缩小
            if child_height > max_height:
                scale = max_height / child_height
                child_height = max_height
                child_width = child_width * scale

        if element.get("width") != "auto":
            child_height *= self.cell_size * element.get("width") / child_width

        if element.get("height") != "auto":
            child_width *= self.cell_size * element.get("height") / child_height

        return child_width, child_height

    def _calculate_row_size(self, element: Dict, max_width: float, max_height: float) -> Tuple[float, float]:
        """计算行容器尺寸 - 考虑主导方向"""
        # 行容器尺寸计算不处理主导方向，留给渲染阶段
        children = element.get('children', [])
        spacing = element.get('spacing', 0) * self.cell_size

        # 计算行内所有子元素的尺寸（使用保守约束）
        child_sizes = []
        total_width = 0
        max_child_height = 0

        for child in children:
            # 使用保守约束（平均分配空间）
            child_max_width = max_width / len(children)
            child_max_height = max_height

            # 递归计算子元素尺寸
            child_width, child_height = self._calculate_element_size(child, child_max_width, child_max_height)
            child_sizes.append((child_width, child_height))
            total_width += child_width
            if child_height > max_child_height:
                max_child_height = child_height

        # 添加间距
        total_width += spacing * (len(children) - 1)

        return total_width, max_child_height

    def _calculate_col_size(self, element: Dict, max_width: float, max_height: float) -> Tuple[float, float]:
        """计算列容器尺寸 - 考虑主导方向"""
        # 列容器尺寸计算不处理主导方向，留给渲染阶段
        children = element.get('children', [])
        spacing = element.get('spacing', 0) * self.cell_size

        # 计算列内所有子元素的尺寸（使用保守约束）
        child_sizes = []
        total_height = 0
        max_child_width = 0

        for child in children:
            # 使用保守约束（平均分配空间）
            child_max_height = max_height / len(children)
            child_max_width = max_width

            # 递归计算子元素尺寸
            child_width, child_height = self._calculate_element_size(child, child_max_width, child_max_height)
            child_sizes.append((child_width, child_height))
            total_height += child_height
            if child_width > max_child_width:
                max_child_width = child_width

        # 添加间距
        total_height += spacing * (len(children) - 1)

        return max_child_width, total_height

    def _render_text(self, image: 'Image.Image', element: Dict,
                     box: Tuple[float, float, float, float]):
        # 获取文本颜色
        color = element['color_white' if self.background_white else 'color_black']
        text = element['text']
        box_x, box_y, box_w, box_h = box

        # 计算文本实际尺寸
        text_width, text_height = self._calculate_text_size(element, box_w, box_h)

        # 创建临时图像
        temp_img = Image.new('RGBA', (int(text_width), int(text_height * 1.5)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(temp_img)

        # 动态调整字体大小（使用字体栈，缺字自动回退）
        best_font = self._font(1)
        best_font_size = 1
        max_font_size = int(max(min(box_h, 1000), 1))

        # 寻找最佳字体大小
        for fs in range(1, max_font_size + 1):  # 设置合理上限
            font = self._font(fs)
            bbox = font.getbbox(text, anchor="la")
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            if text_w <= text_width and text_h <= text_height:
                best_font = font
                best_font_size = fs
            else:
                break

        # 计算文本位置
        bbox = best_font.getbbox(text, anchor="la")
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        text_x = (text_width - text_w) / 2
        text_y = (text_height * 0.4 - text_h) / 2

        # 绘制文本（逐段回退）
        draw_stack_text(draw, best_font, (text_x, text_y), text, fill=color, anchor="la")

        # 计算在父容器中的位置（居中）
        elem_x = box_x + (box_w - text_width) / 2
        elem_y = box_y + (box_h - text_height) / 2

        # 合成到主图像
        image.alpha_composite(temp_img, (int(elem_x), int(elem_y)))

    def _render_image(self, image: 'Image.Image', element: Dict,
                      box: Tuple[float, float, float, float]):
        path = pathlib.Path(minesweepervariants.__path__[0])
        path /= self.assets_path
        path /= f"{element['image']}.png"
        if not os.path.exists(path):
            return None
        img = Image.open(path).convert("RGBA")

        box_x, box_y, box_w, box_h = box

        # 获取原始尺寸
        orig_width, orig_height = img.size

        # 计算缩放比例（保持宽高比）
        width_ratio = box_w / orig_width
        height_ratio = box_h / orig_height
        scale = min(width_ratio, height_ratio)

        # 计算目标尺寸
        target_width = int(orig_width * scale)
        target_height = int(orig_height * scale)

        if (element.get("height") != "auto" and
                element.get("width") != "auto"):
            target_width = element["width"] * self.cell_size
            target_height = element["height"] * self.cell_size
        elif element.get("height") != "auto":
            target_width *= self.cell_size * element.get("height") / target_height
            target_height = element["height"] * self.cell_size
        elif element.get("width") != "auto":
            target_height *= self.cell_size * element.get("width") / target_width
            target_width = element["width"] * self.cell_size

        # 缩放图像
        scaled_img = img.resize((int(target_width), int(target_height)), Image.LANCZOS)

        # 计算居中位置
        elem_x = box_x + (box_w - target_width) / 2
        elem_y = box_y + (box_h - target_height) / 2

        # 合成到主图像
        image.alpha_composite(scaled_img, (int(elem_x), int(elem_y)))

    def _render_row(self, image: 'Image.Image', element: Dict,
                    box: Tuple[float, float, float, float]):
        """
        渲染行元素（水平排列） - 添加固定尺寸支持
        """
        children = element['children']
        spacing = element.get('spacing', 0) * self.cell_size
        box_x, box_y, box_w, box_h = box

        # === 修改：新增固定尺寸分类 ===
        fixed_children = []  # 设置了固定尺寸的元素
        height_dominant_children = []  # 高度主导元素
        width_dominant_children = []  # 宽度主导元素
        placeholder_children = []  # 占位符元素

        for child in children:
            if child.get('type') == 'placeholder':
                placeholder_children.append(child)
            elif child.get('width', 'auto') != 'auto' or child.get('height', 'auto') != 'auto':
                # 设置了固定尺寸
                fixed_children.append(child)
            elif child.get('dominant') == 'height' or child.get('dominant_by_height', False):
                height_dominant_children.append(child)
            else:
                width_dominant_children.append(child)

        # 处理固定尺寸元素
        fixed_sizes = []
        total_fixed_width = 0

        for child in fixed_children:
            # 计算固定尺寸元素的尺寸（考虑约束）
            child_width, child_height = self._calculate_element_size(child, self.cell_size * 3, box_h)
            fixed_sizes.append((child_width, child_height))
            total_fixed_width += child_width

        # 处理高度主导元素
        height_dominant_sizes = []
        total_height_dominant_width = 0

        for child in height_dominant_children:
            # 高度设为容器高度，计算宽度
            child_width, child_height = self._calculate_element_size(child, self.cell_size * 3, box_h)
            height_dominant_sizes.append((child_width, child_height))
            total_height_dominant_width += child_width

        # 处理宽度主导元素
        width_dominant_sizes = []
        total_width_dominant_width = 0

        if width_dominant_children:
            remaining_width = box_w - total_fixed_width - total_height_dominant_width - spacing * (len(children) - 1)
            if remaining_width < 0:
                remaining_width = 0

            avg_width = remaining_width / len(width_dominant_children)

            for child in width_dominant_children:
                child_width, child_height = self._calculate_element_size(child, avg_width, box_h)
                width_dominant_sizes.append((child_width, child_height))
                total_width_dominant_width += child_width

        # 处理占位符元素
        placeholder_sizes = []
        total_placeholder_width = 0

        for child in placeholder_children:
            child_width = child['width'] * self.cell_size
            child_height = child['height'] * self.cell_size
            placeholder_sizes.append((child_width, child_height))
            total_placeholder_width += child_width

        # 合并所有尺寸
        child_sizes = []
        for child in children:
            if child in placeholder_children:
                idx = placeholder_children.index(child)
                child_sizes.append(placeholder_sizes[idx])
            elif child in fixed_children:
                idx = fixed_children.index(child)
                child_sizes.append(fixed_sizes[idx])
            elif child in height_dominant_children:
                idx = height_dominant_children.index(child)
                child_sizes.append(height_dominant_sizes[idx])
            else:
                idx = width_dominant_children.index(child)
                child_sizes.append(width_dominant_sizes[idx])

        # 计算总宽度
        total_width = (total_fixed_width +
                       total_height_dominant_width +
                       total_width_dominant_width +
                       total_placeholder_width)
        total_width += spacing * (len(children) - 1)

        # 整体缩放至容器内
        scale = 1.0
        if total_width > box_w:
            scale = box_w / total_width

        # === 新增：计算水平偏移以实现居中 ===
        total_scaled_width = total_width * scale
        horizontal_offset = (box_w - total_scaled_width) / 2

        # 渲染所有子元素
        current_x = box_x + horizontal_offset  # 添加水平偏移实现居中
        for i, child in enumerate(children):
            child_width, child_height = child_sizes[i]
            child_width *= scale
            child_height *= scale

            # 使用容器高度垂直居中
            child_y = box_y + (box_h - child_height) / 2

            # 递归渲染子元素
            child_box = (current_x, child_y, child_width, child_height)
            self._render_element(image, child, child_box)

            # 更新X位置
            current_x += child_width + spacing * scale

        return

    def _render_col(self, image: 'Image.Image', element: Dict,
                    box: Tuple[float, float, float, float]):
        """
        渲染列元素（垂直排列） - 添加固定尺寸支持
        """
        children = element['children']
        spacing = element.get('spacing', 0) * self.cell_size
        box_x, box_y, box_w, box_h = box

        # === 修改：新增固定尺寸分类 ===
        fixed_children = []  # 设置了固定尺寸的元素
        width_dominant_children = []  # 宽度主导元素
        height_dominant_children = []  # 高度主导元素
        placeholder_children = []  # 占位符元素

        for child in children:
            if child.get('type') == 'placeholder':
                placeholder_children.append(child)
            elif child.get('width', 'auto') != 'auto' or child.get('height', 'auto') != 'auto':
                # 设置了固定尺寸
                fixed_children.append(child)
            elif child.get('dominant') == 'width' or not child.get('dominant_by_height', True):
                width_dominant_children.append(child)
            else:
                height_dominant_children.append(child)

        # 处理固定尺寸元素
        fixed_sizes = []
        total_fixed_height = 0

        for child in fixed_children:
            # 计算固定尺寸元素的尺寸（考虑约束）
            child_width, child_height = self._calculate_element_size(child, box_w, self.cell_size * 3)
            fixed_sizes.append((child_width, child_height))
            total_fixed_height += child_height

        # 处理宽度主导元素
        width_dominant_sizes = []
        total_width_dominant_height = 0

        for child in width_dominant_children:
            # 宽度设为容器宽度，计算高度
            child_width, child_height = self._calculate_element_size(child, box_w, self.cell_size * 3)
            width_dominant_sizes.append((child_width, child_height))
            total_width_dominant_height += child_height

        # 处理高度主导元素
        height_dominant_sizes = []
        total_height_dominant_height = 0

        if height_dominant_children:
            remaining_height = box_h - total_fixed_height - total_width_dominant_height - spacing * (len(children) - 1)
            if remaining_height < 0:
                remaining_height = 0

            avg_height = remaining_height / len(height_dominant_children)

            for child in height_dominant_children:
                child_width, child_height = self._calculate_element_size(child, box_w, avg_height)
                height_dominant_sizes.append((child_width, child_height))
                total_height_dominant_height += child_height

        # 处理占位符元素
        placeholder_sizes = []
        total_placeholder_height = 0

        for child in placeholder_children:
            child_width = child['width'] * self.cell_size
            child_height = child['height'] * self.cell_size
            placeholder_sizes.append((child_width, child_height))
            total_placeholder_height += child_height

        # 合并所有尺寸
        child_sizes = []
        for child in children:
            if child in placeholder_children:
                idx = placeholder_children.index(child)
                child_sizes.append(placeholder_sizes[idx])
            elif child in fixed_children:
                idx = fixed_children.index(child)
                child_sizes.append(fixed_sizes[idx])
            elif child in width_dominant_children:
                idx = width_dominant_children.index(child)
                child_sizes.append(width_dominant_sizes[idx])
            else:
                idx = height_dominant_children.index(child)
                child_sizes.append(height_dominant_sizes[idx])

        # 计算总高度
        total_height = (total_fixed_height +
                        total_width_dominant_height +
                        total_height_dominant_height +
                        total_placeholder_height)
        total_height += spacing * (len(children) - 1)

        # 整体缩放至容器内
        scale = 1.0
        if total_height > box_h:
            scale = box_h / total_height

        # === 新增：计算垂直偏移以实现居中 ===
        total_scaled_height = total_height * scale
        vertical_offset = (box_h - total_scaled_height) / 2

        # 渲染所有子元素
        current_y = box_y + vertical_offset  # 添加垂直偏移实现居中
        for i, child in enumerate(children):
            child_width, child_height = child_sizes[i]
            child_width *= scale
            child_height *= scale

            # 使用容器宽度水平居中
            child_x = box_x + (box_w - child_width) / 2

            # 递归渲染子元素
            child_box = (child_x, current_y, child_width, child_height)
            self._render_element(image, child, child_box)

            # 更新Y位置
            current_y += child_height + spacing * scale
