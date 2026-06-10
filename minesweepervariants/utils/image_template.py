from typing import Literal, Union, Tuple, List, TypedDict


# ========== TypedDict 定义 ==========

class BaseElement(TypedDict):
    """所有元素的公共字段"""
    cover: bool
    dominant: Union[Literal['width', 'height'], None]


class TextElement(BaseElement):
    type: Literal['text']
    text: str
    content: str
    color_black: Tuple[int, int, int]  # RGB tuple
    color_white: Tuple[int, int, int]  # RGB tuple
    width: Union[float, Literal['auto']]
    height: Union[float, Literal['auto']]
    font_size: int
    style: str


class ImageElement(BaseElement):
    type: Literal['image']
    image: str
    width: Union[float, Literal['auto']]
    height: Union[float, Literal['auto']]
    style: str


class RowElement(BaseElement):
    type: Literal['row']
    children: List[Union['TextElement', 'ImageElement', 'RowElement', 'ColElement', 'PlaceholderElement']]
    spacing: int
    width: Union[float, Literal['auto']]
    height: Union[float, Literal['auto']]


class ColElement(BaseElement):
    type: Literal['col']
    children: List[Union['TextElement', 'ImageElement', 'RowElement', 'ColElement', 'PlaceholderElement']]
    spacing: int
    width: Union[float, Literal['auto']]
    height: Union[float, Literal['auto']]


class PlaceholderElement(BaseElement):
    type: Literal['placeholder']
    width: float
    height: float


# 联合类型，方便引用
Element = Union[TextElement, ImageElement, RowElement, ColElement, PlaceholderElement]


# ========== 函数实现 ==========

def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """将十六进制颜色字符串转换为 RGB 元组"""
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def get_text(
        text: str,
        width: Union[float, Literal['auto']] = "auto",
        height: Union[float, Literal['auto']] = "auto",
        cover_pos_label: bool = True,
        color: Tuple[str, str] = ("#FFFFFF", "#000000"),
        dominant_by_height: bool = True,
        style: str = "",
) -> TextElement:
    """
    创建文本元素

    :param text: 文本内容
    :param width: 宽度（单元格单位或'auto'）
    :param height: 高度（单元格单位或'auto'）
    :param cover_pos_label: 是否覆盖格子内的“X=N”标识
    :param color: 颜色对，第一个用于黑底，第二个用于白底，格式为"#RRGGBB"
    :param dominant_by_height: True 表示高主导对齐，False 表示宽主导对齐
    :param style: Web 样式内容
    """
    dominant: Union[Literal['width', 'height']] = "height" if dominant_by_height else "width"
    return {
        "type": "text",
        "text": text,
        "content": text,
        "color_black": hex_to_rgb(color[0]),
        "color_white": hex_to_rgb(color[1]),
        "width": width,
        "height": height,
        "font_size": 1,
        "cover": cover_pos_label,
        "dominant": dominant,
        "style": style,
    }


def get_image(
        image_path: str,
        image_width: Union[float, Literal['auto']] = "auto",
        image_height: Union[float, Literal['auto']] = "auto",
        cover_pos_label: bool = True,
        dominant_by_height: bool = True,
        style: str = "",
) -> ImageElement:
    """
    创建图片元素

    :param image_path: 图片在 data 目录下的路径
    :param image_width: 宽度（单元格单位或'auto'）
    :param image_height: 高度（单元格单位或'auto'）
    :param cover_pos_label: 是否覆盖“X=N”标识
    :param dominant_by_height: True 表示高主导对齐，False 表示宽主导对齐
    :param style: Web 样式内容
    """
    dominant: Literal['width', 'height'] = "height" if dominant_by_height else "width"
    return {
        "type": "image",
        "image": image_path,
        "height": image_height,
        "width": image_width,
        "cover": cover_pos_label,
        "dominant": dominant,
        "style": style,
    }


def get_row(
        *args: Element,
        spacing: int = 0,
        dominant_by_height: bool = True,
) -> RowElement:
    """
    水平排列元素（行）

    :param args: 子元素列表
    :param spacing: 元素之间的间距
    :param dominant_by_height: True 表示高主导对齐，False 表示宽主导对齐
    """
    dominant: Literal['width', 'height'] = "height" if dominant_by_height else "width"
    for child in args:
        if child["dominant"] is None:
            # 类型检查器可能认为 child["dominant"] 不是 None，这里强制赋值
            child["dominant"] = "height"  # type: ignore

    # 计算高度：如果所有子元素高度都是数值，取最大值；否则为 'auto'
    numeric_heights: List[Union[int, float]] = []
    for e in args:
        _height = e.get("height")
        if isinstance(_height, (int, float)):
            numeric_heights.append(_height)
    if numeric_heights:
        height_val: Union[float, Literal['auto']] = max(numeric_heights)
    else:
        height_val = "auto"

    return {
        "type": "row",
        "children": list(args),
        "spacing": spacing,
        "cover": all(e["cover"] for e in args),
        "height": height_val,
        "width": "auto",
        "dominant": dominant,
    }


def get_col(
        *args: Element,
        spacing: int = 0,
        dominant_by_height: bool = False,
) -> ColElement:
    """
    垂直排列元素（列）

    :param args: 子元素列表
    :param spacing: 元素之间的间距
    :param dominant_by_height: True 表示高主导对齐，False 表示宽主导对齐
    """
    dominant: Literal['width', 'height'] = "height" if dominant_by_height else "width"
    for child in args:
        if child["dominant"] is None:
            child["dominant"] = "width"  # type: ignore

    numeric_widths: List[Union[int, float]] = []
    for e in args:
        _width = e.get("width")
        if isinstance(_width, (int, float)):
            numeric_widths.append(_width)
    if numeric_widths:
        width_val: Union[float, Literal['auto']] = max(numeric_widths)
    else:
        width_val = "auto"

    return {
        "type": "col",
        "children": list(args),
        "spacing": spacing,
        "cover": all(e["cover"] for e in args),
        "height": "auto",
        "width": width_val,
        "dominant": dominant,
    }


def get_dummy(width: float = 0.01, height: float = 0.01) -> PlaceholderElement:
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
        "dominant": None,
    }
