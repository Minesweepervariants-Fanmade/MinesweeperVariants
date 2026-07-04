#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2026/07/01 11:52
# @Author  : Wu_RH
# @FileName: ocr_assets.py
from pathlib import Path

import cv2
import numpy as np

from minesweepervariants.utils.tool import SELF_PATH
from minesweepervariants.utils.ocr.detect import show

ASSETS_DIR = SELF_PATH + "\\minesweepervariants\\assets"


class TemplateMatcher:
    def __init__(self):
        self.templates = {}
        for p in Path(ASSETS_DIR).glob("*.png"):
            p_name = p.stem
            if p_name.endswith("_black"):
                continue
            img = cv2.imread(str(p))
            if img is not None:
                tmpl_cropped = self.crop_foreground(img)
                self.templates[p_name] = tmpl_cropped

    def ocr_img(self, img):
        # 1. 转灰度，大津法二值化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

        # # 2. 确保前景（文字/物体）为白色，背景为黑色
        # white_pixels = cv2.countNonZero(binary)
        # total_pixels = gray.shape[0] * gray.shape[1]
        # if white_pixels > total_pixels // 2:
        #     binary = cv2.bitwise_not(binary)
        #
        # # 3. 定位前景的外接矩形
        # coords = cv2.findNonZero(binary)
        # if coords is None:
        #     # 没找到前景，返回全黑彩色图（与原图同尺寸）
        #     return np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        # x, y, w, h = cv2.boundingRect(coords)
        #
        # # 4. 在原图上裁剪该矩形区域
        # result = binary[y:y + h, x:x + w]
        # result = np.bitwise_not(result)
        #
        # return result

    def crop_foreground(self, img):
        # 1. 转灰度，大津法二值化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 2. 确保前景（文字/物体）为白色，背景为黑色
        white_pixels = cv2.countNonZero(binary)
        total_pixels = gray.shape[0] * gray.shape[1]
        if white_pixels > total_pixels // 2:
            binary = cv2.bitwise_not(binary)

        # 3. 定位前景的外接矩形
        coords = cv2.findNonZero(binary)
        if coords is None:
            # 没找到前景，返回全黑彩色图（与原图同尺寸）
            return np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        x, y, w, h = cv2.boundingRect(coords)

        # 4. 在原图上裁剪该矩形区域
        roi_color = img[y:y + h, x:x + w]
        # 同步裁剪二值掩码
        roi_mask = binary[y:y + h, x:x + w]

        # 5. 用掩码保留前景彩色，背景全部变成黑色
        result = cv2.bitwise_and(roi_color, roi_color, mask=roi_mask)
        return result

    def match(self, cell_img, threshold=0.7):
        cell_cropped = self.crop_foreground(cell_img)
        best_char = None
        best_score = -1
        for char, tmpl_cropped in self.templates.items():
            if tmpl_cropped.shape[0] > cell_cropped.shape[0] or tmpl_cropped.shape[1] > cell_cropped.shape[1]:
                tmpl_cropped = cv2.resize(tmpl_cropped, (cell_cropped.shape[1], cell_cropped.shape[0]))
            result = cv2.matchTemplate(cell_cropped, tmpl_cropped, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_char = char
        if best_score >= threshold:
            return best_char, best_score
        return None, -1
