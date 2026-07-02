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

def detect_grid_cells(img_path) -> Tuple[Dict[Tuple[int, int], np.ndarray], Size]:
    logger = get_logger()

    # ---------- 1. 读取与预处理 ----------
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"无法读取: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # if np.mean(gray) < 127:
    #     gray = cv2.bitwise_not(gray)
    #     img_processed = cv2.bitwise_not(img)
    # else:
    img_processed = img.copy()
    orig_h, orig_w = img.shape[:2]

    # ---------- 2. 二值化 ----------
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY_INV, 15, 2)
    # kernel = np.ones((3, 3), np.uint8)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                cv2.THRESH_BINARY_INV, 51, -2)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.sum(binary == 255) > binary.size * 0.5:
        binary = cv2.bitwise_not(binary)
    # show(binary)

    # ---------- 3. 投影 ----------
    row_sum = np.sum(binary == 255, axis=1).astype(np.float32)
    col_sum = np.sum(binary == 255, axis=0).astype(np.float32)
    row_centered = row_sum - np.mean(row_sum)
    col_centered = col_sum - np.mean(col_sum)

    # ---------- 4. 自相关 ----------
    def autocorr(x):
        n = len(x)
        corr = np.correlate(x, x, mode='full')
        corr = corr[n - 1:]  # 从0 lag开始
        corr = corr / corr[0]
        return corr

    row_autocorr = autocorr(row_centered)
    col_autocorr = autocorr(col_centered)

    # ---------- 5. 联合自相关：求和，取前10个局部峰值（不平滑） ----------
    max_lag = min(len(row_autocorr), len(col_autocorr)) - 1
    combined = row_autocorr[1:max_lag + 1] + col_autocorr[1:max_lag + 1]

    # 忽略 lag < 5（网格不可能小于5像素）
    min_period = 5
    search_start = min_period - 1
    if search_start >= len(combined):
        raise ValueError("图片太小或网格太密")

    # 找所有局部极大值
    peaks = []
    for i in range(search_start + 1, len(combined) - 1):
        if combined[i] > combined[i - 1] and combined[i] > combined[i + 1]:
            peaks.append((i, combined[i]))  # i 是 lag-1 的索引

    # 按强度降序，取前10个
    peaks.sort(key=lambda x: -x[1])
    top_peaks = peaks[:10]
    candidate_periods = [idx + 1 for idx, val in top_peaks]  # 转回实际lag

    for rank, (idx, strength) in enumerate(top_peaks):
        period = idx + 1
        logger.debug(f"#{rank + 1}: 周期={period}, 强度={strength:.4f}")
    # 可选画图（注释掉，只打印数字）
    # import matplotlib.pyplot as plt
    # plt.plot(range(1, max_lag+1), combined)
    # for p in candidate_periods:
    #     plt.axvline(x=p, color='r', linestyle='--', alpha=0.5)
    # plt.title('Combined Autocorrelation (Row + Col)')
    # plt.show()

    # 注意：这里仅输出候选，不自动选择最佳，后续需手动指定 step
    # 您可从此处选择其中一个作为 step，继续执行
    # 例如：step = float(candidate_periods[0])  # 取最强的一个
    step = float(candidate_periods[0])  # 临时取第一个，您可以改成任意候选

    # ---------- 6. 定位网格线 ----------
    def locate_lines(projection, step):
        import numpy as np

        raw = np.array(projection, dtype=np.float32)
        diff = np.diff(raw)
        diff_positive = np.maximum(diff, 0).astype(np.int64)
        # diff_positive = np.convolve(diff_positive, np.ones(5)/5, mode='same')

        # ===== 计算 score_positive =====
        score_positive = []
        for i in range(len(diff_positive) - step - 2):
            score_positive.append([])
            for real_step in range(-2, 3):
                real_step += step
                line_diff = abs(diff_positive[i] - diff_positive[i + real_step])
                line_min = min(diff_positive[i], diff_positive[i + real_step])
                s = line_min - line_diff
                s -= 2 * np.log2(diff_positive[i + 1: i + real_step].sum() + 1)
                score_positive[-1].append(s)
            score_positive[-1] = max(score_positive[-1])
        score_positive = np.array(score_positive, dtype=np.int64)
        score_min, score_max = 20, 200
        score_positive = np.maximum(score_positive, -500).astype(np.int64)
        score_positive_cut = np.maximum(score_positive, score_min).astype(np.int64)
        score_positive_cut = np.minimum(score_positive_cut, score_max).astype(np.int64)

        # score_positive = score_positive / score_positive.max()

        # ===== 定义 show，包含 5 个子图 =====
        def show():
            from matplotlib import pyplot as plt
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9))
            ax1.plot(raw, label='Raw Projection')
            ax1.set_title('Raw Projection')
            ax1.legend()

            ax2.plot(diff_positive, label='Positive Diff', color='green')
            ax2.set_title('Positive Diff (negative removed)')
            ax2.legend()

            ax3.plot(score_positive, label='Score (max - score)', color='red')
            ax3.axhline(y=score_min, color='black', linestyle='--', label='y=0')
            ax3.axhline(y=score_max, color='gray', linestyle='--', label='y=200')
            ax3.set_title('Score Positive')
            ax3.legend()

            plt.tight_layout()
            plt.show()

        # show()

        final = []
        _tmp_final = []
        for i in range(score_positive_cut.shape[0] - step - 2):
            while True:
                def check(__pos_i):
                    if __pos_i + 1 >= score_positive_cut.shape[0]:
                        return False
                    if score_positive_cut[__pos_i + 1] <= score_min:
                        return False
                    if score_positive_cut[__pos_i] > score_min + 10:
                        return False
                    if score_positive_cut[__pos_i + 2] > score_min + 10:
                        return False
                    return True
                flag = False
                j = i
                for j in range(-5, 6):
                    j += i
                    if check(j):
                        flag = True
                        break
                if flag:
                    i = j
                    shift = 1
                    if not any(i + j + shift in _tmp_final for j in range(-5, 6)):
                        _tmp_final.append(i + shift)
                    i += step
                    if i + shift < raw.shape[0]:
                        _tmp_final.append(i + shift)
                    if i >= score_positive.shape[0]:
                        break
                    continue
                break
            if len(_tmp_final) > len(final):
                logger.debug(f"获取网格像素位置: {_tmp_final}")
                final = _tmp_final[:]
            _tmp_final = []
        logger.debug(f"最终网格像素位置: {final}")
        return final

    logger.debug(f"开始处理垂直线")
    v_lines = locate_lines(col_sum, int(step))
    logger.debug(f"开始处理水平线")
    h_lines = locate_lines(row_sum, int(step))

    if len(h_lines) < 2 or len(v_lines) < 2:
        raise ValueError(f"定位网格线失败: H={len(h_lines)}, V={len(v_lines)}")

    # ---------- 8. 裁剪格子 ----------
    cells = {}
    for i in range(len(h_lines) - 1):
        for j in range(len(v_lines) - 1):
            x1, y1 = v_lines[j], h_lines[i]
            x2, y2 = v_lines[j + 1], h_lines[i + 1]
            if x2 > x1 and y2 > y1:
                cell = img_processed[y1:y2, x1:x2]
                cells[(i, j)] = cell

    # ---------- 9. 可视化 ----------

    # vis = img_processed.copy()
    # for x in v_lines:
    #     cv2.line(vis, (x, 0), (x, orig_h), (0, 255, 0), 2)
    # for y in h_lines:
    #     cv2.line(vis, (0, y), (orig_w, y), (0, 255, 0), 2)
    # cv2.imwrite("debug_grid_lines.jpg", vis)
    # # cv2.imshow('Grid (Combined Autocorr + LR)', vis)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    logger.info(f"检测到步长: {step:.2f} 像素")
    logger.info(f"列数: {len(v_lines) - 1}, 行数: {len(h_lines) - 1}")

    return cells, Size(len(v_lines) - 1, len(h_lines) - 1)
