#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉问题判断模块 — 数据生成脚本（骨架）
TODO:
  1. 定义视觉相关/非视觉相关的文本模板
  2. 实现 generate_visual() 和 generate_non_visual() 函数
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import random
import argparse
from collections import Counter

from core.data_utils import fill_template, split_dataset, save_jsonl, add_context


def generate_visual(n: int) -> list:
    """生成视觉相关问题样本 TODO: 实现模板填充"""
    return [{"text": "这是什么颜色", "label": "visual"}] * n


def generate_non_visual(n: int) -> list:
    """生成非视觉问题样本 TODO: 实现模板填充"""
    return [{"text": "今天天气怎么样", "label": "non_visual"}] * n


def generate_dataset(
    output_dir: str = "./dataset_visual",
    context_ratio: float = 0.35,
    seed: int = 42,
):
    """生成视觉问题判断数据集 TODO: 补充各类别样本数参数"""
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("视觉问题判断数据集生成（骨架）")
    print("=" * 60)

    dataset = []
    # TODO: 调用生成函数并汇总
    # dataset.extend(generate_visual(n=1000))
    # dataset.extend(generate_non_visual(n=1000))

    print(f"\n基础样本总计: {len(dataset)}条")

    train_data, val_data, test_data = split_dataset(dataset)
    save_jsonl(train_data, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(val_data, os.path.join(output_dir, "val.jsonl"))
    save_jsonl(test_data, os.path.join(output_dir, "test.jsonl"))
    save_jsonl(dataset, os.path.join(output_dir, "dataset_all.jsonl"))

    print("\n✅ 数据集生成完成！")
    return train_data, val_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成视觉问题判断数据集")
    parser.add_argument("--output", "-o", type=str, default="./dataset_visual", help="输出目录")
    parser.add_argument("--context-ratio", type=float, default=0.35, help="多轮上下文比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    generate_dataset(
        output_dir=args.output,
        context_ratio=args.context_ratio,
        seed=args.seed,
    )
