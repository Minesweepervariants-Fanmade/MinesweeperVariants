#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# @Time    : 2026/07/01 12:50
# @Author  : Wu_RH
# @FileName: ocr.py

import os
from typing import TypedDict, Dict, Tuple

import cv2
import numpy as np

from rapidocr_onnxruntime import RapidOCR

from minesweepervariants.position import Position
from minesweepervariants.size import Size
from minesweepervariants.utils.ocr.detect import detect_and_crop_grid
from minesweepervariants.utils.ocr.ocr_assets import TemplateMatcher
from minesweepervariants.utils.tool import get_logger


def show(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_color(
    cell_img, hsv_lower=(20, 100, 100),
    hsv_upper=(30, 255, 255),
    area_percent=0.05
):
    """
    检测图像中是否存在指定颜色区域（按面积百分比）
    :param cell_img: BGR 图像
    :param hsv_lower: HSV 下界 (H, S, V)
    :param hsv_upper: HSV 上界 (H, S, V)
    :param area_percent: 黄色像素占总像素的最小比例（例如 0.05 表示 5%）
    :return: True 表示存在黄色
    """
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    yellow_pixels = cv2.countNonZero(mask)
    total_pixels = cell_img.shape[0] * cell_img.shape[1]
    ratio = yellow_pixels / total_pixels
    return ratio > area_percent


class CellData(TypedDict):
    is_mines: bool
    texts: list[str]
    imgs: list[str]


class OCRResult(TypedDict):
    cell_data: Dict[Tuple[int, int], CellData]
    size_data: Size


def ocr_board(img_path: str) -> OCRResult:
    logger = get_logger()
    if not os.path.exists(img_path):
        logger.error(f"不存在的文件: {img_path}")
        raise FileNotFoundError(img_path)
    img = cv2.imread(img_path)
    pos_cell, size = detect_and_crop_grid(img)
    ocr = RapidOCR(
        box_thresh=0.3,      # 默认约0.6，调低此值让检测模型更“敏感”，更容易检出小文字[reference:6][reference:7]
        text_score=0.2,  # 默认约0.5，调低此值放宽识别结果的置信度要求[reference:8][reference:9]
        unclip_ratio=2.0  # 默认约1.5，调大此值可适当扩大检测框，有助于完整包含小文字[reference:10][reference:11]
    )
    tmpl = TemplateMatcher()
    pos_ocr_result = {}
    for pos_key, cell_img in pos_cell.items():
        cell_img = cell_img
        is_flag = get_color(cell_img)

        ocr_result, _ = ocr(cell_img)
        img_result = tmpl.match(cell_img[4:-4, 4:-4])

        logger.debug(f"POS[{pos_key}]: OCR:[{ocr_result}] img:[{img_result}]")
        # show(cell_img)
        if not (img_result[0] or ocr_result or is_flag):
            continue
        pos_ocr_result[pos_key] = CellData(
                texts=[r[1] for r in ocr_result] if ocr_result else [],
                imgs=[img_result[0]] if img_result[0] else [],
                is_mines=is_flag,
            )

    logger.debug(pos_ocr_result)
    return OCRResult(cell_data=pos_ocr_result, size_data=size)
