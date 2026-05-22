#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拒识模块配置
"""

LABEL2ID = {
    "unsafe": 0,
    "non_dialogue": 1,
    "waiting": 2,
    "ready": 3,
    "confused": 4,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

MODEL_NAME = "./models/chinese-macbert-base"
