#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图识别分流模块配置
TODO: 根据实际业务场景修改标签映射和默认超参
"""

# 示例标签映射 — 请替换为真实意图类别
LABEL2ID = {
    "weather": 0,
    "music": 1,
    "device_control": 2,
    "schedule": 3,
    "navigation": 4,
    "other": 5,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

MODEL_NAME = "hfl/chinese-macbert-base"
