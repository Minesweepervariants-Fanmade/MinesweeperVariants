

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def get_dummy(width: float = 0.01, height: float = 0.01) -> dict[str, object]:
    return {
        "type": "placeholder",
        "width": width,
        "height": height,
        "cover": True,
        "dominant": None,
    }


def get_text(
    text: str,
    width: float | str = "auto",
    height: float | str = "auto",
    cover_pos_label: bool = True,
    color: tuple[str, str] = ("#FFFFFF", "#000000"),
    dominant_by_height: bool = True,
    style: str = "",
) -> dict[str, object]:
    dominant = "height" if dominant_by_height else "width"
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
    image_width: float | str = "auto",
    image_height: float | str = "auto",
    cover_pos_label: bool = True,
    dominant_by_height: bool = True,
    style: str = "",
) -> dict[str, object]:
    dominant = "height" if dominant_by_height else "width"
    return {
        "type": "image",
        "image": image_path,
        "height": image_height,
        "width": image_width,
        "cover": cover_pos_label,
        "dominant": dominant,
        "style": style,
    }

def get_row(*args: dict[str, object], spacing: int = 0, dominant_by_height: bool = False) -> dict[str, object]:
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
    dominant= "height" if dominant_by_height else "width"
    for child in args:
        if child.get("dominant") is None:
            child["dominant"] = "width"
    width_values = [item["width"] for item in args if isinstance(item["width"], int)]
    width = max(width_values) if width_values else "auto"
    return {
        "type": "col",
        "children": args,
        "spacing": spacing,
        "cover": all(item["cover"] for item in args),
        "height": "auto",
        "width": width,
        "dominant": dominant,
    }
