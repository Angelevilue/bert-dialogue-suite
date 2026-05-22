#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拒识模块训练脚本
薄封装：定义标签映射，调用 core.trainer.train() 执行通用训练流程。
"""

import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from core.cli import build_argparser
from core.trainer import train
from config import LABEL2ID, ID2LABEL, MODEL_NAME


def main():
    parser = build_argparser(description="BERT 拒识模块微调")
    args = parser.parse_args()

    # 验证数据文件
    for path in [args.train_data, args.val_data]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在: {path}")

    # 使用拒识模块的默认超参（若 CLI 未覆盖则生效）
    model_name = args.model_name if args.model_name != "hfl/chinese-macbert-base" else MODEL_NAME

    train(
        args=args,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()
