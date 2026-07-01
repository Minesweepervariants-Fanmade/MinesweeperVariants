#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2026/07/01 10:33
# @Author  : Wu_RH
# @FileName: detect.py
from typing import Tuple, List, Dict

import cv2
import numpy as np

from minesweepervariants.size import Size
from minesweepervariants.utils.tool import get_logger


# -------------------- 核心函数 --------------------
def preprocess_binary(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """读取图片，确保网格线为白色(255)，背景为黑色(0)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_ratio = np.sum(thresh == 255) / thresh.size
    if white_ratio > 0.5:
        binary = cv2.bitwise_not(thresh)
    else:
        binary = thresh

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary, img


def extract_peak_positions(proj: np.ndarray) -> List[int]:
    """动态阈值提取峰值（质心法）"""
    if np.max(proj) == 0:
        return []

    mean_val = np.mean(proj)
    std_val = np.std(proj)
    thresh_stat = mean_val + 1.8 * std_val
    thresh_min = np.max(proj) * 0.15
    threshold = max(thresh_stat, thresh_min)

    peaks = []
    in_segment = False
    start = 0

    for i, val in enumerate(proj):
        if val > threshold and not in_segment:
            in_segment = True
            start = i
        elif val <= threshold and in_segment:
            in_segment = False
            end = i
            segment = proj[start:end]
            if len(segment) > 0:
                indices = np.arange(len(segment))
                centroid = start + np.average(indices, weights=segment)
                peaks.append(int(centroid))

    if in_segment:
        segment = proj[start:]
        indices = np.arange(len(segment))
        centroid = start + np.average(indices, weights=segment)
        peaks.append(int(centroid))

    return peaks


def refine_peaks(peaks: List[int], axis_limit: int) -> List[int]:
    """
    基于等间距特性滤除噪点，并用众数间距补齐遗漏线条。
    补齐范围限制在原始检测到的最小和最大线条附近，不铺满全图。
    """
    if len(peaks) < 2:
        return peaks

    peaks = sorted(set(peaks))

    # 1. 计算候选间距，找出众数间距（网格基本步长）
    diffs = []
    for i in range(min(len(peaks), 30)):
        for j in range(i + 1, min(i + 10, len(peaks))):
            d = peaks[j] - peaks[i]
            if d > 3:
                diffs.append(d)

    if not diffs:
        return peaks

    max_d = max(diffs)
    bins = np.arange(0, max_d + 3, 2)  # 2像素宽桶
    hist, bin_edges = np.histogram(diffs, bins=bins)
    if len(hist) == 0:
        return peaks
    best_idx = np.argmax(hist)
    grid_step = int(round((bin_edges[best_idx] + bin_edges[best_idx + 1]) / 2))
    if grid_step == 0:
        return peaks

    # 2. 根据众数步长筛选候选峰（保留符合间距的，丢弃噪点）
    refined = []
    for p in peaks:
        if not refined:
            refined.append(p)
            continue

        dist = p - refined[0]
        n = round(dist / grid_step)
        if n <= 0:
            continue
        ideal_pos = refined[0] + n * grid_step
        if abs(p - ideal_pos) <= grid_step * 0.2:
            if not refined or abs(p - refined[-1]) > grid_step * 0.5:
                refined.append(p)

    if len(refined) < 2:
        return peaks  # 保底

    # 3. 补齐：仅补在原始检测到的范围附近（最多外扩一个步长）
    orig_min = min(peaks)
    orig_max = max(peaks)

    # 向前补齐（不能小于 orig_min - grid_step）
    while refined[0] - grid_step > orig_min - grid_step:
        refined.insert(0, refined[0] - grid_step)
    # 向后补齐（不能大于 orig_max + grid_step）
    while refined[-1] + grid_step < orig_max + grid_step:
        refined.append(refined[-1] + grid_step)

    # 最后再过滤掉超出图片边界的（确保安全）
    refined = [x for x in refined if 0 <= x < axis_limit]

    return refined


def detect_and_crop_grid(img: np.ndarray) -> Tuple[Dict[Tuple[int, int], np.ndarray], Size]:
    """主函数：识别网格并裁剪每个格子，返回字典 key=(行,列) value=格子图像，索引连续"""
    logger = get_logger()
    binary, original = preprocess_binary(img)
    h, w = binary.shape

    vert_proj = np.sum(binary, axis=0)
    horz_proj = np.sum(binary, axis=1)

    raw_x = extract_peak_positions(vert_proj)
    raw_y = extract_peak_positions(horz_proj)

    x_lines = refine_peaks(raw_x, w)
    y_lines = refine_peaks(raw_y, h)

    # 过滤掉紧贴边缘的线（防止误检测）
    x_lines = [x for x in x_lines if 0 < x < w - 1]
    y_lines = [y for y in y_lines if 0 < y < h - 1]

    if len(x_lines) < 2 or len(y_lines) < 2:
        logger.error("未能检测到有效的网格结构，请检查图片是否包含清晰的网格线。")
        return {}, Size(0, 0)

    x_step = int(round(np.median(np.diff(x_lines))))
    y_step = int(round(np.median(np.diff(y_lines))))

    logger.debug(f"检测到垂直网格线 {len(x_lines)} 条，水平网格线 {len(y_lines)} 条")
    logger.debug(f"网格步长 (像素): 宽={x_step}, 高={y_step}")

    valid_cells = []  # 存储 (orig_row, orig_col, cell_image)

    for i in range(len(y_lines) - 1):
        y1, y2 = y_lines[i], y_lines[i + 1]
        if abs((y2 - y1) - y_step) > y_step * 0.3:
            continue
        for j in range(len(x_lines) - 1):
            x1, x2 = x_lines[j], x_lines[j + 1]
            if abs((x2 - x1) - x_step) > x_step * 0.3:
                continue
            cell = original[y1:y2, x1:x2]
            if cell.size == 0:
                continue
            valid_cells.append((i, j, cell))

    if not valid_cells:
        logger.warning("未裁剪出任何有效格子")
        return {}, Size(0, 0)

    # 提取所有出现过的原始行、列并排序，建立连续映射
    orig_rows = sorted({item[0] for item in valid_cells})
    orig_cols = sorted({item[1] for item in valid_cells})
    row_mapping = {orig: new for new, orig in enumerate(orig_rows)}
    col_mapping = {orig: new for new, orig in enumerate(orig_cols)}

    result = {}
    for orig_i, orig_j, cell_img in valid_cells:
        new_i = row_mapping[orig_i]
        new_j = col_mapping[orig_j]
        result[(new_j, new_i)] = cell_img

    logger.debug(f"成功裁剪出 {len(result)} 个有效格子（行数={len(orig_rows)}，列数={len(orig_cols)}）")
    return result, Size(len(orig_rows), len(orig_cols))
