#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2026/07/01 11:52
# @Author  : Wu_RH
# @FileName: ocr_assets.py
from pathlib import Path

import cv2

from minesweepervariants.utils.tool import SELF_PATH

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
                self.templates[p_name] = img

    def crop_foreground(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(binary)
        if coords is None:
            return img
        x, y, w, h = cv2.boundingRect(coords)
        return img[y:y + h, x:x + w]

    def match(self, cell_img, threshold=0.7):
        cell_cropped = self.crop_foreground(cell_img)
        best_char = None
        best_score = -1
        for char, tmpl in self.templates.items():
            tmpl_cropped = self.crop_foreground(tmpl)
            if tmpl_cropped.shape[0] > cell_cropped.shape[0] or tmpl_cropped.shape[1] > cell_cropped.shape[1]:
                tmpl_cropped = cv2.resize(tmpl_cropped, (cell_cropped.shape[1], cell_cropped.shape[0]))
            result = cv2.matchTemplate(cell_cropped, tmpl_cropped, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_char = char
        if best_score >= threshold:
            return best_char, best_score
        return None, None
