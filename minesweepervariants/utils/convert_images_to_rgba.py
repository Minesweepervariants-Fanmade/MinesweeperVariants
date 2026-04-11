from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image

try:
    from .tool import get_logger
except Exception:
    get_logger = None


# =========================
# 输入参数（直接改变量）
# =========================
INPUT_A_PATH = Path(r"z:\A.png")
INPUT_B_PATH = Path(r"z:\B.png")
OUTPUT_DIR = Path(r"z:\test_out")

WHITE_PCT = 20.0  # 对应 HTML 的 compressPct（1..100）
BLACK_PCT = 0.0   # 对应 HTML 的 compressBlackPct（0..100）
INVERT_B = False

AUTO_TUNE_TOP = True    # 自动白点：令 max(B-A) 初始接近 0
AUTO_TUNE_BLACK = True  # 自动黑点：令 99 分位饱和度损失 < 0.1%


def _safe_info(msg: str):
    if get_logger is None:
        return
    try:
        get_logger().info(msg)
    except Exception:
        return


@dataclass
class ProcessResult:
    sat_loss_avg: float
    sat_loss_top1: float
    max_diff: float
    max_diff_after: float
    stats_text: str
    out_rgba: np.ndarray
    white_comp: np.ndarray
    black_comp: np.ndarray
    alpha_view: np.ndarray
    b_gray_view: np.ndarray
    a_view: np.ndarray


def clamp(v: np.ndarray | float, lo: float, hi: float):
    return np.clip(v, lo, hi)


def luma(rgb: np.ndarray) -> np.ndarray:
    # rgb: [..., 3], float
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def hsv_saturation(rgb: np.ndarray) -> np.ndarray:
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    s = np.zeros_like(mx, dtype=np.float32)
    mask = mx > 1e-6
    s[mask] = (mx[mask] - mn[mask]) / mx[mask]
    return s


def percentile_from_hist(hist: np.ndarray, total: int, p: float) -> int:
    # 与 JS 版本保持一致：累计计数达到 total*p 的首个 bin
    target = total * p
    acc = 0
    for i in range(256):
        acc += int(hist[i])
        if acc >= target:
            return i
    return 255


def hist_from_gray(gray: np.ndarray) -> np.ndarray:
    # gray: float32 [0,255]
    bins = np.rint(gray).astype(np.int32)
    bins = np.clip(bins, 0, 255)
    return np.bincount(bins.ravel(), minlength=256).astype(np.int64)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_rgb(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def save_rgba(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8), mode="RGBA").save(path)


def letterbox_b_to_a_frame(b_rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    # 与 JS 一致：黑底 + 等比缩放 + 居中
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    b_img = Image.fromarray(b_rgb, mode="RGB")

    scale = min(w / b_rgb.shape[1], h / b_rgb.shape[0])
    dw = int(round(b_rgb.shape[1] * scale))
    dh = int(round(b_rgb.shape[0] * scale))
    dx = (w - dw) // 2
    dy = (h - dh) // 2

    resized = b_img.resize((dw, dh), resample=Image.Resampling.BILINEAR)
    canvas.paste(resized, (dx, dy))
    return np.array(canvas, dtype=np.uint8)


def process(
    a_rgb_u8: np.ndarray,
    b_rgb_u8: np.ndarray,
    white_pct: float,
    black_pct: float,
    invert_b: bool,
) -> ProcessResult:
    h, w = a_rgb_u8.shape[0], a_rgb_u8.shape[1]

    # 1) A
    a_view = a_rgb_u8.copy()
    a_rgb = a_view.astype(np.float32)

    # 2) B letterbox 到 A 尺寸
    b_fit = letterbox_b_to_a_frame(b_rgb_u8, w, h)
    b_rgb = b_fit.astype(np.float32)

    # 3) B 灰度 + 直方图
    b_gray = luma(b_rgb)
    if invert_b:
        b_gray = 255.0 - b_gray
    hist_b = hist_from_gray(b_gray)
    b_gray_view = np.clip(np.rint(b_gray), 0, 255).astype(np.uint8)

    # 4) B 顶底分位一次线性映射
    total = w * h
    p99_b = percentile_from_hist(hist_b, total, 0.99)
    p1_b = percentile_from_hist(hist_b, total, 0.01)

    pct = white_pct / 100.0
    b_pct = black_pct / 100.0
    target_bottom = 255.0 * b_pct
    target_top = max(target_bottom, 255.0 * pct)
    den_b = max(1e-6, float(p99_b - p1_b))
    slope_b = (target_top - target_bottom) / den_b

    y_mid = np.clip(b_gray, p1_b, p99_b)
    b_gray_mapped = target_bottom + (y_mid - p1_b) * slope_b
    b_gray_mapped = np.clip(b_gray_mapped, target_bottom, target_top)
    b_gray_mapped = np.clip(b_gray_mapped, 0.0, 255.0).astype(np.float32)

    # 5) A 低切到 1% 最低亮度
    a_y = luma(a_rgb).astype(np.float32)
    hist_a = hist_from_gray(a_y)
    p1_a = percentile_from_hist(hist_a, total, 0.01)
    a_y = np.maximum(a_y, float(p1_a))

    # 6) max(B-A)，再增益 A
    diff = b_gray_mapped - a_y
    max_diff = float(np.max(diff))

    gain_a = 1.0
    a_max = float(np.max(a_y))
    range_a = max(1e-6, a_max - float(p1_a))
    if max_diff > 0:
        gain_a = 1.0 + max_diff / range_a

    a_y = float(p1_a) + gain_a * (a_y - float(p1_a))
    a_y = np.clip(a_y, 0.0, 255.0)

    # 再检查并强制 A>=B
    diff2 = b_gray_mapped - a_y
    max_diff_after = float(np.max(diff2))
    a_y = np.maximum(a_y, b_gray_mapped)

    # 7) 反解 RGBA
    ya = a_y
    yb = b_gray_mapped
    alpha = 1.0 - (ya - yb) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)

    a_r = a_rgb[..., 0]
    a_g = a_rgb[..., 1]
    a_b = a_rgb[..., 2]

    eps = 1e-6
    alpha_safe = np.where(alpha > eps, alpha, 1.0)

    fr = (a_r - (1.0 - alpha) * 255.0) / alpha_safe
    fg = (a_g - (1.0 - alpha) * 255.0) / alpha_safe
    fb = (a_b - (1.0 - alpha) * 255.0) / alpha_safe

    fr = np.where(alpha > eps, fr, 0.0)
    fg = np.where(alpha > eps, fg, 0.0)
    fb = np.where(alpha > eps, fb, 0.0)

    fr = np.clip(fr, 0.0, 255.0)
    fg = np.clip(fg, 0.0, 255.0)
    fb = np.clip(fb, 0.0, 255.0)

    out_rgba = np.stack(
        [
            np.rint(fr),
            np.rint(fg),
            np.rint(fb),
            np.rint(alpha * 255.0),
        ],
        axis=-1,
    ).astype(np.uint8)

    # 预览：白底/黑底/alpha
    wr = alpha * fr + (1.0 - alpha) * 255.0
    wg = alpha * fg + (1.0 - alpha) * 255.0
    wb = alpha * fb + (1.0 - alpha) * 255.0
    white_comp = np.stack([np.rint(np.clip(wr, 0, 255)), np.rint(np.clip(wg, 0, 255)), np.rint(np.clip(wb, 0, 255))], axis=-1).astype(np.uint8)

    br = alpha * fr
    bg = alpha * fg
    bb = alpha * fb
    black_comp = np.stack([np.rint(np.clip(br, 0, 255)), np.rint(np.clip(bg, 0, 255)), np.rint(np.clip(bb, 0, 255))], axis=-1).astype(np.uint8)

    av = np.rint(np.clip(alpha, 0, 1) * 255.0).astype(np.uint8)
    alpha_view = np.stack([av, av, av], axis=-1).astype(np.uint8)

    # 饱和度损失统计（对白底）
    s_a = hsv_saturation(a_rgb)
    s_w = hsv_saturation(white_comp.astype(np.float32))
    mask = s_a > 1e-6
    if np.any(mask):
        loss = np.clip((s_a[mask] - s_w[mask]) / s_a[mask], 0.0, 1.0)
        sat_loss_avg = float(np.mean(loss))
        # 与 JS 保持一致的 99 分位索引
        loss_sorted = np.sort(loss)
        idx = min(loss_sorted.shape[0] - 1, int(np.floor(0.99 * loss_sorted.shape[0])))
        sat_loss_top1 = float(loss_sorted[idx])
    else:
        sat_loss_avg = 0.0
        sat_loss_top1 = 0.0

    alpha_min = int(av.min())
    alpha_max = int(av.max())

    stats_text = (
        f"A尺寸: {w}x{h}\n"
        f"B反色: {'开' if invert_b else '关'}\n"
        f"B灰度1分位(底部1%阈值): {float(p1_b):.2f}\n"
        f"B灰度99分位(顶部1%阈值): {float(p99_b):.2f}\n"
        f"B压缩目标白点: {target_top:.2f}\n"
        f"B压缩目标黑点: {target_bottom:.2f}\n"
        f"B单步映射: y' = {target_bottom:.2f} + (clamp(y, {float(p1_b):.2f}, {float(p99_b):.2f})-{float(p1_b):.2f})*{slope_b:.6f}\n"
        f"A亮度1分位(底部1%阈值): {float(p1_a):.2f}\n"
        f"A动态范围增益: {gain_a:.4f}\n"
        f"max(B-A)初始: {max_diff:.4f}\n"
        f"max(B-A)处理后: {max_diff_after:.4f} (<=0越好)\n"
        f"白底饱和度平均损失: {sat_loss_avg * 100.0:.3f}%\n"
        f"白底饱和度损失99分位(最高1%): {sat_loss_top1 * 100.0:.3f}%\n"
        f"Alpha范围[8bit]: [{alpha_min}, {alpha_max}]"
    )

    return ProcessResult(
        sat_loss_avg=sat_loss_avg,
        sat_loss_top1=sat_loss_top1,
        max_diff=max_diff,
        max_diff_after=max_diff_after,
        stats_text=stats_text,
        out_rgba=out_rgba,
        white_comp=white_comp,
        black_comp=black_comp,
        alpha_view=alpha_view,
        b_gray_view=np.stack([b_gray_view, b_gray_view, b_gray_view], axis=-1),
        a_view=a_view,
    )


def auto_tune_top_pct_for_zero_max_diff(
    a_rgb_u8: np.ndarray,
    b_rgb_u8: np.ndarray,
    black_pct: float,
    invert_b: bool,
) -> float:
    # 与 HTML 当前逻辑一致：解析求白点，使 max(B-A) 初始接近 0
    h, w = a_rgb_u8.shape[0], a_rgb_u8.shape[1]

    a_rgb = a_rgb_u8.astype(np.float32)
    b_fit = letterbox_b_to_a_frame(b_rgb_u8, w, h).astype(np.float32)

    b_gray_raw = luma(b_fit)
    if invert_b:
        b_gray_raw = 255.0 - b_gray_raw

    hist_b = hist_from_gray(b_gray_raw)
    p99_b = percentile_from_hist(hist_b, w * h, 0.99)
    p1_b = percentile_from_hist(hist_b, w * h, 0.01)

    a_y_low = luma(a_rgb).astype(np.float32)
    hist_a = hist_from_gray(a_y_low)
    p1_a = percentile_from_hist(hist_a, w * h, 0.01)
    a_y_low = np.maximum(a_y_low, float(p1_a))

    target_bottom = 255.0 * (black_pct / 100.0)
    den_b = max(1e-6, float(p99_b - p1_b))
    eps = 1e-9

    y_mid = np.clip(b_gray_raw, p1_b, p99_b)
    m = (y_mid - p1_b) / den_b

    # 对 m>eps: targetTop <= targetBottom + (a-targetBottom)/m
    # 对 m<=eps: 要求 targetBottom<=a，否则不可行
    infeasible_by_bottom = np.max(target_bottom - a_y_low[m <= eps]) if np.any(m <= eps) else -np.inf
    if infeasible_by_bottom > 0:
        return float(100.0 * target_bottom / 255.0)

    if np.any(m > eps):
        ub = target_bottom + (a_y_low[m > eps] - target_bottom) / m[m > eps]
        top_upper = float(np.min(ub))
    else:
        top_upper = target_bottom

    target_top = float(np.clip(top_upper, target_bottom, 255.0))
    return float(100.0 * target_top / 255.0)


def auto_tune_black_pct_for_sat_loss_top1(
    a_rgb_u8: np.ndarray,
    b_rgb_u8: np.ndarray,
    white_pct: float,
    invert_b: bool,
    target_loss: float = 0.001,
) -> float:
    # 与 HTML 当前逻辑一致：二分找最小 black_pct，使 satLossTop1 <= target_loss
    max_black_pct = max(0.0, white_pct)

    l0 = process(a_rgb_u8, b_rgb_u8, white_pct=white_pct, black_pct=0.0, invert_b=invert_b).sat_loss_top1
    if l0 <= target_loss:
        return 0.0

    l1 = process(
        a_rgb_u8,
        b_rgb_u8,
        white_pct=white_pct,
        black_pct=max_black_pct,
        invert_b=invert_b,
    ).sat_loss_top1
    if l1 > target_loss:
        return float(max_black_pct)

    lo, hi = 0.0, float(max_black_pct)
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        lm = process(a_rgb_u8, b_rgb_u8, white_pct=white_pct, black_pct=mid, invert_b=invert_b).sat_loss_top1
        if lm <= target_loss:
            hi = mid
        else:
            lo = mid
    return hi


def run_pipeline() -> Tuple[ProcessResult, Dict[str, float]]:
    a = load_rgb(INPUT_A_PATH)
    b = load_rgb(INPUT_B_PATH)

    white_pct = WHITE_PCT
    black_pct = BLACK_PCT

    if AUTO_TUNE_TOP:
        white_pct = auto_tune_top_pct_for_zero_max_diff(a, b, black_pct=black_pct, invert_b=INVERT_B)

    if AUTO_TUNE_BLACK:
        black_pct = auto_tune_black_pct_for_sat_loss_top1(
            a,
            b,
            white_pct=white_pct,
            invert_b=INVERT_B,
            target_loss=0.001,
        )

    result = process(a, b, white_pct=white_pct, black_pct=black_pct, invert_b=INVERT_B)
    params = {
        "white_pct": white_pct,
        "black_pct": black_pct,
        "invert_b": float(INVERT_B),
    }
    return result, params


def convert_images_to_rgba(
    image_a: Image.Image,
    image_b: Image.Image,
    white_pct: float = WHITE_PCT,
    black_pct: float = BLACK_PCT,
    invert_b: bool = INVERT_B,
    auto_tune_top: bool = AUTO_TUNE_TOP,
    auto_tune_black: bool = AUTO_TUNE_BLACK,
) -> Image.Image:
    """
    输入两个 PIL Image，输出与 HTML 工具等价处理后的 RGBA PIL Image。

    参数单位与页面一致：
    - white_pct: 顶部 1% 压缩白点百分比（0~100）
    - black_pct: 底部 1% 压缩黑点百分比（0~100）
    """
    a = np.array(image_a.convert("RGB"), dtype=np.uint8)
    b = np.array(image_b.convert("RGB"), dtype=np.uint8)

    _safe_info(
        "convert_images_to_rgba start: "
        f"A={a.shape[1]}x{a.shape[0]}, B={b.shape[1]}x{b.shape[0]}, "
        f"white_pct={white_pct}, black_pct={black_pct}, invert_b={invert_b}, "
        f"auto_tune_top={auto_tune_top}, auto_tune_black={auto_tune_black}"
    )

    w_pct = float(white_pct)
    b_pct = float(black_pct)

    if auto_tune_top:
        w_pct = auto_tune_top_pct_for_zero_max_diff(a, b, black_pct=b_pct, invert_b=invert_b)
        _safe_info(f"convert_images_to_rgba auto_tune_top applied: white_pct={w_pct:.4f}")

    if auto_tune_black:
        b_pct = auto_tune_black_pct_for_sat_loss_top1(
            a,
            b,
            white_pct=w_pct,
            invert_b=invert_b,
            target_loss=0.001,
        )
        _safe_info(f"convert_images_to_rgba auto_tune_black applied: black_pct={b_pct:.4f}")

    result = process(a, b, white_pct=w_pct, black_pct=b_pct, invert_b=invert_b)
    _safe_info(
        "convert_images_to_rgba done: "
        f"max_diff={result.max_diff:.4f}, max_diff_after={result.max_diff_after:.4f}, "
        f"sat_loss_top1={result.sat_loss_top1 * 100.0:.4f}%"
    )
    return Image.fromarray(result.out_rgba, mode="RGBA")


def save_outputs(result: ProcessResult, params: Dict[str, float]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_rgb(OUTPUT_DIR / "A_view.png", result.a_view)
    save_rgb(OUTPUT_DIR / "B_gray_view.png", result.b_gray_view)
    save_rgba(OUTPUT_DIR / "result_rgba.png", result.out_rgba)
    save_rgb(OUTPUT_DIR / "white_comp.png", result.white_comp)
    save_rgb(OUTPUT_DIR / "black_comp.png", result.black_comp)
    save_rgb(OUTPUT_DIR / "alpha_view.png", result.alpha_view)

    stats_path = OUTPUT_DIR / "stats.txt"
    stats_path.write_text(
        result.stats_text
        + "\n\n"
        + f"最终参数:\nwhite_pct={params['white_pct']:.4f}\n"
        + f"black_pct={params['black_pct']:.4f}\n"
        + f"invert_b={'True' if params['invert_b'] > 0.5 else 'False'}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    res, used_params = run_pipeline()
    save_outputs(res, used_params)
    print(res.stats_text)
    print("\n输出目录:", OUTPUT_DIR)
