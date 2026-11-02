#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
字体栈与回退渲染。

Pillow 的 ``ImageFont.truetype`` 只能加载单一字体文件，当主字体缺少某个字形
（例如 Copperplate 缺 CJK）时，Pillow 会退化成 ``.notdef`` 方框。
本模块实现了多字体回退：按字符挑选第一个拥有该字形的字体，再逐段绘制，
同时保持 Pillow 的 ``anchor`` 布局语义。

对外提供：
- :class:`FontStack`  —— 字体栈，支持 ``getlength`` / ``getbbox`` / ``getmetrics``
- :func:`draw_text` / :func:`draw_multiline_text` —— 替代 ``ImageDraw.text``
- :func:`resolve_fallback_fonts` —— 解析可用的回退字体文件列表
"""
from __future__ import annotations

import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw, ImageFont

from minesweepervariants.utils.tool import get_logger

__all__ = [
    "FontStack",
    "draw_text",
    "draw_multiline_text",
    "resolve_fallback_fonts",
]


# ---------------------------------------------------------------------------
# 加载缓存
# ---------------------------------------------------------------------------


@lru_cache(maxsize=256)
def _load_freetype(path: str, size: int, index: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, index=index)


@lru_cache(maxsize=64)
def _load_cmap(path: str, index: int) -> Optional[frozenset[int]]:
    """读取字体的 Unicode 码位集合；失败时返回 None（视为全部支持）。"""
    try:
        from fontTools.ttLib import TTFont

        with TTFont(path, fontNumber=index, lazy=True) as font:
            cmap = font.getBestCmap()
            if not cmap:
                return None
            return frozenset(cmap.keys())
    except Exception as exc:  # fontTools 缺失、文件损坏等
        get_logger().debug(f"[FontStack] 读取字体 cmap 失败 ({path}#{index}): {exc}")
        return None


# ---------------------------------------------------------------------------
# 单个字体面
# ---------------------------------------------------------------------------


class FontFace:
    """封装单个字体文件，附带字形覆盖查询。"""

    __slots__ = ("path", "index", "size", "font", "_cmap", "label")

    def __init__(self, path: Union[str, Path], size: int, index: int = 0) -> None:
        self.path = str(path)
        self.index = int(index)
        self.size = int(size)
        self.label = Path(self.path).name
        self.font: ImageFont.FreeTypeFont = _load_freetype(self.path, self.size, self.index)
        self._cmap: Optional[frozenset[int]] = None

    @classmethod
    def from_pillow_font(
        cls, font: ImageFont.ImageFont, label: str = "PillowDefault"
    ) -> "FontFace":
        """用已有的 Pillow 字体对象构造 FontFace（不读取 cmap，视为全覆盖）。"""
        face = cls.__new__(cls)
        face.path = f"<{label}>"
        face.index = 0
        face.size = int(getattr(font, "size", 0)) or 0
        face.label = label
        face.font = font
        face._cmap = None
        return face

    @property
    def cmap(self) -> Optional[frozenset[int]]:
        if self._cmap is None:
            self._cmap = _load_cmap(self.path, self.index)
        return self._cmap

    def has_char(self, ch: str) -> bool:
        cmap = self.cmap
        if cmap is None:
            # 无法判定时保守认为支持，交由 Pillow 处理
            return True
        return ord(ch) in cmap

    # 委托 Pillow 常用接口
    @property
    def ascent(self) -> int:
        return self.font.getmetrics()[0]

    @property
    def descent(self) -> int:
        return self.font.getmetrics()[1]

    def getmetrics(self) -> Tuple[int, int]:
        return self.font.getmetrics()

    def getlength(self, text: str) -> float:
        return self.font.getlength(text)

    def getmask2(self, text: str, **kwargs):
        return self.font.getmask2(text, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"FontFace({self.label}#{self.index}, size={self.size})"


# ---------------------------------------------------------------------------
# 回退字体解析
# ---------------------------------------------------------------------------


def _candidate_system_fonts() -> Iterable[Path]:
    """返回当前系统上可能存在的、覆盖 CJK/符号的字体路径。"""
    if os.name == "nt":
        roots = [
            Path(os.environ.get("WINDIR", r"C:\\Windows")) / "Fonts",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts",
        ]
        names = (
            "msyh.ttc",
            "msyhbd.ttc",
            "msyhl.ttc",
            "simhei.ttf",
            "simsun.ttc",
            "simkai.ttf",
            "Deng.ttf",
            "seguisym.ttf",
            "seguiemj.ttf",
            "arialuni.ttf",
            "arial.ttf",
        )
        for root in roots:
            if not root.is_dir():
                continue
            for name in names:
                p = root / name
                if p.is_file():
                    yield p
    else:
        roots = [
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            Path.home() / ".local" / "share" / "fonts",
        ]
        names = (
            "NotoSansCJK-Regular.ttc",
            "NotoSansCJKsc-Regular.otf",
            "NotoSansSC-Regular.otf",
            "NotoColorEmoji.ttf",
            "DejaVuSans.ttf",
            "wqy-microhei.ttc",
            "wqy-zenhei.ttc",
        )
        for root in roots:
            if not root.is_dir():
                continue
            for name in names:
                for p in root.rglob(name):
                    if p.is_file():
                        yield p


def resolve_fallback_fonts(
    assets_dir: Union[str, Path, None] = None,
) -> List[Path]:
    """
    解析可用的回退字体列表（去重、保持顺序）。

    :param assets_dir: 当前主题的资源目录（例如 ``assets`` 或 ``assets/white``）
    """
    result: List[Path] = []
    seen: set[Path] = set()

    def _add(path: Union[str, Path, None]) -> None:
        if not path:
            return
        p = Path(path)
        try:
            p = p.resolve()
        except OSError:
            return
        if not p.is_file() or p in seen:
            return
        seen.add(p)
        result.append(p)

    # 1) 系统字体优先：内置 msyh_boot.ttf 仅 1270 字形，缺「雷」「雾」等常用字，
    #    若排在前面会让同一个中文短语里不同字符落到不同字体，观感不一致。
    #    因此先使用覆盖更完整的系统 CJK 字体。
    for p in _candidate_system_fonts():
        _add(p)

    # 2) 内置字体作为最后兜底（无系统 CJK 字体时仍可渲染大部分中文）
    try:
        import minesweepervariants as _mv

        pkg_root = Path(_mv.__path__[0])
        if assets_dir is not None:
            _add(pkg_root / Path(assets_dir) / "msyh_boot.ttf")
        _add(pkg_root / "assets" / "msyh_boot.ttf")
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# 字体栈
# ---------------------------------------------------------------------------


class FontStack:
    """按字符挑选字体的字体栈，接口对齐 ``PIL.ImageFont.FreeTypeFont`` 的子集。"""

    def __init__(self, faces: Sequence[FontFace]) -> None:
        if not faces:
            raise ValueError("FontStack 需要至少一个 FontFace")
        self.faces: List[FontFace] = list(faces)
        self.primary = self.faces[0]
        # 字符 -> 字体面 的缓存。长文本（如底部说明）逐字查 cmap 会明显变慢
        self._char_face_cache: dict[str, FontFace] = {}

    # ------------------------------------------------------------------ 构造
    @classmethod
    def load(
        cls,
        primary_path: Union[str, Path],
        size: int,
        fallback_paths: Sequence[Union[str, Path]] = (),
        *,
        primary_index: int = 0,
        fallback_index: int = 0,
    ) -> "FontStack":
        size = max(1, int(size))
        faces: List[FontFace] = [FontFace(primary_path, size, primary_index)]
        for p in fallback_paths:
            if not p:
                continue
            try:
                faces.append(FontFace(p, size, fallback_index))
            except Exception as exc:  # 字体文件损坏 / 不支持的格式
                get_logger().debug(f"[FontStack] 跳过回退字体 {p}: {exc}")
        return cls(faces)

    # ---------------------------------------------------------------- metrics
    @property
    def size(self) -> int:
        return self.primary.size

    @property
    def ascent(self) -> int:
        return self.primary.ascent

    @property
    def descent(self) -> int:
        return self.primary.descent

    def getmetrics(self) -> Tuple[int, int]:
        return self.primary.getmetrics()

    # ---------------------------------------------------------------- 字体分配
    def _face_for(self, ch: str) -> FontFace:
        face = self._char_face_cache.get(ch)
        if face is not None:
            return face
        for candidate in self.faces:
            if candidate.has_char(ch):
                self._char_face_cache[ch] = candidate
                return candidate
        self._char_face_cache[ch] = self.primary
        return self.primary

    def split(self, text: str) -> List[Tuple[FontFace, str]]:
        """把文本按“使用同一字体”切分成若干段。"""
        if not text:
            return []
        runs: List[Tuple[FontFace, str]] = []
        for ch in text:
            face = self._face_for(ch)
            if runs and runs[-1][0] is face:
                runs[-1] = (face, runs[-1][1] + ch)
            else:
                runs.append((face, ch))
        return runs

    # ---------------------------------------------------------------- 纯主字体快路径
    def _primary_only(self, text: str) -> bool:
        """判断文本是否完全可由主字体渲染（无需回退）。"""
        primary = self.primary
        for ch in text:
            if ch == "\n":
                continue
            if self._face_for(ch) is not primary:
                return False
        return True

    # ---------------------------------------------------------------- 长度/包围盒
    def getlength(self, text: str) -> float:
        if self._primary_only(text):
            return self.primary.getlength(text)
        total = 0.0
        for face, seg in self.split(text):
            total += face.getlength(seg)
        return total

    def _layout_line(self, text: str, stroke_width: int = 0):
        """
        计算单行文本的排版。

        返回 ``(items, min_x, min_y, max_x, max_y)``，其中 ``items`` 为
        ``[(left_rel, top_rel, mask)]``，坐标均相对 baseline 起点；
        ``min_*`` / ``max_*`` 为整行的 ink 包围盒（相对 baseline）。

        混排时 ``stroke_width`` 对每一段分别生效；由于不同字体段之间不会互相
        描边，与整段一次性描边存在极细微差异，属可接受的视觉折衷。
        """
        items = []
        min_x = math.inf
        min_y = math.inf
        max_x = -math.inf
        max_y = -math.inf
        cursor = 0.0
        for face, seg in self.split(text):
            try:
                mask, offset = face.getmask2(
                    seg, anchor="ls", stroke_width=stroke_width
                )
            except Exception:
                mask, offset = face.getmask2(seg)
                offset = (offset[0], offset[1] - face.ascent)
            w, h = mask.size
            if w > 0 and h > 0:
                left = cursor + offset[0]
                top = offset[1]
                items.append((left, top, mask))
                if left < min_x:
                    min_x = left
                if top < min_y:
                    min_y = top
                if left + w > max_x:
                    max_x = left + w
                if top + h > max_y:
                    max_y = top + h
            cursor += face.getlength(seg)
        if not items:
            return items, 0.0, 0.0, 0.0, 0.0
        return items, min_x, min_y, max_x, max_y

    def _anchor_offset(
        self,
        anchor: Optional[str],
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> Tuple[float, float]:
        """
        返回锚点相对于 baseline 起点的偏移 (h, v)。

        语义与 Pillow 的 ``anchor`` 保持一致：
          - a: ascender        - t: ink top        - m: (ascent-descent)/2
          - s: baseline        - b: ink bottom     - d: descender
        垂直方向向上为负。
        """
        anchor = anchor or "la"
        if len(anchor) != 2:
            raise ValueError(f"bad anchor specified: {anchor}")
        ha, va = anchor[0], anchor[1]
        if ha == "l":
            h = min_x
        elif ha == "m":
            h = (min_x + max_x) / 2.0
        elif ha == "r":
            h = max_x
        else:
            raise ValueError(f"bad anchor specified: {anchor}")

        ascent, descent = self.getmetrics()
        if va == "a":  # ascender
            v = -float(ascent)
        elif va == "t":  # text top
            v = min_y
        elif va == "m":  # middle
            v = -float(ascent - descent) / 2.0
        elif va == "s":  # baseline
            v = 0.0
        elif va == "b":  # bottom
            v = max_y
        elif va == "d":  # descender
            v = float(descent)
        else:
            raise ValueError(f"bad anchor specified: {anchor}")
        return h, v

    def getbbox(
        self,
        text: str,
        anchor: Optional[str] = None,
        stroke_width: int = 0,
    ) -> Tuple[float, float, float, float]:
        """等价于 ``Font.getbbox(text, anchor=anchor)``。"""
        if self._primary_only(text):
            # 纯主字体：直接使用 Pillow 的结果，保证与历史渲染完全一致
            return self.primary.font.getbbox(
                text, anchor=anchor or "la", stroke_width=stroke_width
            )
        if "\n" in text:
            return self._multiline_bbox(text, anchor, stroke_width)
        items, min_x, min_y, max_x, max_y = self._layout_line(text, stroke_width)
        if not items:
            return (0.0, 0.0, 0.0, 0.0)
        h, v = self._anchor_offset(anchor, min_x, min_y, max_x, max_y)
        return (min_x - h, min_y - v, max_x - h, max_y - v)

    def _multiline_bbox(
        self,
        text: str,
        anchor: Optional[str],
        stroke_width: int = 0,
    ) -> Tuple[float, float, float, float]:
        anchor = anchor or "la"
        if len(anchor) != 2:
            raise ValueError(f"bad anchor specified: {anchor}")
        if anchor[1] in "tb":
            raise ValueError("anchor not supported for multiline text")

        lines = text.split("\n")
        line_spacing = self.getbbox("A", "la", stroke_width)[3] + stroke_width + 4
        widths = [self.getlength(line) for line in lines]
        max_width = max(widths) if widths else 0.0

        # 以第一行起点为基准，逐行累计 bbox
        line_boxes = []
        for line in lines:
            items, min_x, min_y, max_x, max_y = self._layout_line(line, stroke_width)
            h, v = self._anchor_offset(anchor, min_x, min_y, max_x, max_y)
            line_boxes.append((min_x - h, min_y - v, max_x - h, max_y - v))

        top = 0.0
        if anchor[1] == "m":
            top -= (len(lines) - 1) * line_spacing / 2.0
        elif anchor[1] == "d":
            top -= (len(lines) - 1) * line_spacing

        min_x = math.inf
        min_y = math.inf
        max_x = -math.inf
        max_y = -math.inf
        for idx, (bx0, by0, bx1, by1) in enumerate(line_boxes):
            width_diff = max_width - widths[idx]
            left = 0.0
            if anchor[0] == "m":
                left -= width_diff / 2.0
            elif anchor[0] == "r":
                left -= width_diff
            lx0 = bx0 + left
            lx1 = bx1 + left
            ly0 = by0 + top
            ly1 = by1 + top
            min_x = min(min_x, lx0)
            min_y = min(min_y, ly0)
            max_x = max(max_x, lx1)
            max_y = max(max_y, ly1)
            top += line_spacing
        if math.isinf(min_x):
            return (0.0, 0.0, 0.0, 0.0)
        return (min_x, min_y, max_x, max_y)

    # ---------------------------------------------------------------- 绘制
    def draw_on(
        self,
        draw: ImageDraw.ImageDraw,
        xy: Tuple[float, float],
        text: str,
        fill=None,
        anchor: Optional[str] = None,
        stroke_width: int = 0,
        stroke_fill=None,
        spacing: int = 4,
    ) -> None:
        # 纯主字体：交给 Pillow 原生绘制，完整保留描边/字距等既有行为
        if self._primary_only(text):
            draw.text(
                xy,
                text,
                font=self.primary.font,
                fill=fill,
                anchor=anchor,
                spacing=spacing,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill,
            )
            return
        if "\n" in text:
            self._draw_multiline(
                draw, xy, text, fill, anchor, stroke_width, stroke_fill, spacing
            )
        else:
            self._draw_single(draw, xy, text, fill, anchor, stroke_width, stroke_fill)

    def _draw_single(
        self,
        draw: ImageDraw.ImageDraw,
        xy: Tuple[float, float],
        text: str,
        fill,
        anchor: Optional[str],
        stroke_width: int,
        stroke_fill,
    ) -> None:
        if not text:
            return
        # 有描边时：先用“带描边扩展”的 mask 铺描边色，再用原 mask 叠主体色，
        # 逐段模拟 Pillow 的描边效果。
        if stroke_width:
            outline = stroke_fill if stroke_fill is not None else fill
            self._paint(draw, xy, text, outline, anchor, stroke_width, use_stroke=True)
            self._paint(draw, xy, text, fill, anchor, 0, use_stroke=False)
            return
        self._paint(draw, xy, text, fill, anchor, 0, use_stroke=False)

    def _paint(
        self,
        draw: ImageDraw.ImageDraw,
        xy: Tuple[float, float],
        text: str,
        fill,
        anchor: Optional[str],
        stroke_width: int,
        *,
        use_stroke: bool,
    ) -> None:
        # use_stroke=True 时取带描边扩展的 mask（用于铺描边色）；
        # 否则取原字形 mask（用于叠主体色）。
        layout_stroke = stroke_width if use_stroke else 0
        items, min_x, min_y, max_x, max_y = self._layout_line(text, layout_stroke)
        if not items:
            return
        h, v = self._anchor_offset(anchor, min_x, min_y, max_x, max_y)
        base_x = int(xy[0] - h)
        base_y = int(xy[1] - v)

        for left, top, mask in items:
            mask_img = _mask_to_image(mask)
            x = base_x + int(round(left))
            y = base_y + int(round(top))
            try:
                draw.bitmap((x, y), mask_img, fill=fill)
            except ValueError:
                # RGBA mask（如彩色 emoji）时回退到 alpha_composite
                if draw.im is None:
                    continue
                if mask_img.mode == "RGBA":
                    draw.im.alpha_composite(mask_img, (x, y))

    def _draw_multiline(
        self,
        draw: ImageDraw.ImageDraw,
        xy: Tuple[float, float],
        text: str,
        fill,
        anchor: Optional[str],
        stroke_width: int,
        stroke_fill,
        spacing: int,
    ) -> None:
        anchor = anchor or "la"
        if len(anchor) != 2:
            raise ValueError(f"bad anchor specified: {anchor}")
        if anchor[1] in "tb":
            raise ValueError("anchor not supported for multiline text")

        lines = text.split("\n")
        line_spacing = self.getbbox("A", "la", stroke_width)[3] + stroke_width + spacing
        widths = [self.getlength(line) for line in lines]
        max_width = max(widths) if widths else 0.0

        top = xy[1]
        if anchor[1] == "m":
            top -= (len(lines) - 1) * line_spacing / 2.0
        elif anchor[1] == "d":
            top -= (len(lines) - 1) * line_spacing

        for idx, line in enumerate(lines):
            width_diff = max_width - widths[idx]
            left = xy[0]
            if anchor[0] == "m":
                left -= width_diff / 2.0
            elif anchor[0] == "r":
                left -= width_diff
            self._draw_single(
                draw, (left, top), line, fill, anchor, stroke_width, stroke_fill
            )
            top += line_spacing

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        names = ", ".join(f.label for f in self.faces)
        return f"FontStack(size={self.size}, faces=[{names}])"


# ---------------------------------------------------------------------------
# 底层工具
# ---------------------------------------------------------------------------


def _mask_to_image(mask) -> Image.Image:
    """把 Pillow 的 ``ImagingCore`` 掩码转换成 ``Image``。"""
    mode = mask.mode if mask.mode in ("L", "RGBA") else "L"
    return Image.frombytes(mode, mask.size, bytes(mask))


# ---------------------------------------------------------------------------
# 便捷函数（替代 ``ImageDraw.text``）
# ---------------------------------------------------------------------------


def draw_text(
    draw: ImageDraw.ImageDraw,
    font: FontStack,
    xy: Tuple[float, float],
    text: str,
    fill=None,
    anchor: Optional[str] = None,
    stroke_width: int = 0,
    stroke_fill=None,
    spacing: int = 4,
) -> None:
    """使用 :class:`FontStack` 绘制文本（替代 ``ImageDraw.text``）。"""
    font.draw_on(
        draw,
        xy,
        text,
        fill=fill,
        anchor=anchor,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill,
        spacing=spacing,
    )


def draw_multiline_text(
    draw: ImageDraw.ImageDraw,
    font: FontStack,
    xy: Tuple[float, float],
    text: str,
    fill=None,
    anchor: Optional[str] = None,
    spacing: int = 4,
    stroke_width: int = 0,
    stroke_fill=None,
) -> None:
    """使用 :class:`FontStack` 绘制多行文本。"""
    font._draw_multiline(
        draw, xy, text, fill, anchor, stroke_width, stroke_fill, spacing
    )
