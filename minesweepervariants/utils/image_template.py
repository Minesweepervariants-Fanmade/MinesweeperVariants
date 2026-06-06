

from typing import Literal


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )
def get_text(
        text: str,
        width: float | Literal['auto'] = "auto",
        height: float | Literal['auto'] = "auto",
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
    dominant = "height" if dominant_by_height else "width"
    return {
        "type": "text",
        "text": text,
        "content": text,
        'color_black': hex_to_rgb(color[0]),
        'color_white': hex_to_rgb(color[1]),
        'width': width,
        'height': height,
        "font_size": 1,
        "cover": cover_pos_label,
        "dominant": dominant,
        "style": style,
    }


def get_image(
        image_path: str,
        image_width: float | Literal['auto'] = "auto",
        image_height: float | Literal['auto'] = "auto",
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
    dominant = "height" if dominant_by_height else "width"
    return {
        'type': 'image',
        'image': image_path,  # 图片对象
        'height': image_height,  # 高度（单元格单位或auto）
        'width': image_width,  # 宽度（单元格单位或auto）
        "cover": cover_pos_label,
        "dominant": dominant,
        "style": style,
    }


def get_row(*args: dict[str, object], spacing: int = 0, dominant_by_height: bool = True) -> dict[str, object]:
    """
    水平排列元素
    :param args: 子元素列表
    :param spacing: 每个元素之间的间距值
    :param dominant_by_height: 高主导的对齐 否则宽主导
    """
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


def get_col(*args: dict[str, object], spacing: int = 0, dominant_by_height: bool = False) -> dict[str, object]:
    """
    垂直排列元素
    :param args: 子元素列表
    :param spacing: 每个元素之间的间距值
    :param dominant_by_height: 高主导的对齐 否则宽主导
    """
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
