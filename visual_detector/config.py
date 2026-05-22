#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉问题判断模块配置
TODO: 根据实际业务场景修改标签映射
"""

# 二分类示例：是否是视觉相关问题
LABEL2ID = {
    "visual": 0,
    "non_visual": 1,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

MODEL_NAME = "hfl/chinese-macbert-base"
