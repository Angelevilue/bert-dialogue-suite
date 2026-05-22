#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 CLI 参数解析
"""

import argparse


def build_argparser(description: str = "BERT 微调") -> argparse.ArgumentParser:
    """构建通用的 argparse ArgumentParser"""
    parser = argparse.ArgumentParser(description=description)

    # 数据
    parser.add_argument("--train_data", type=str, required=True, help="训练集 JSONL 路径")
    parser.add_argument("--val_data", type=str, required=True, help="验证集 JSONL 路径")
    parser.add_argument("--test_data", type=str, default=None, help="测试集 JSONL 路径 (可选)")

    # 模型
    parser.add_argument(
        "--model_name",
        type=str,
        default="hfl/chinese-macbert-base",
        help="预训练模型名称 (默认: hfl/chinese-macbert-base)",
    )

    # 训练超参
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup比例")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--patience", type=int, default=3, help="早停耐心值")

    # 其他
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--fp16", action="store_true", help="开启混合精度训练")
    parser.add_argument("--save_every_epoch", action="store_true", help="每个epoch都保存模型")

    return parser
