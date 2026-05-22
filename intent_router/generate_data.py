#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图识别分流模块 — 数据生成脚本（骨架）
TODO:
  1. 定义各类意图的文本模板和填充词
  2. 实现 generate_<intent>() 函数
  3. 在 generate_dataset() 中汇总各意图样本
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import random
import argparse
from collections import Counter

from core.data_utils import fill_template, split_dataset, save_jsonl, add_context


# ============================================================
# TODO: 定义意图模板和填充词
# ============================================================
# EXAMPLE_TEMPLATES = [
#     "今天{location}天气怎么样",
#     "播放{artist}的歌",
# ]
# EXAMPLE_FILLERS = {
#     "location": ["北京", "上海", "广州"],
#     "artist": ["周杰伦", "林俊杰"],
# }


def generate_weather(n: int) -> list:
    """生成天气查询意图样本 TODO: 实现模板填充"""
    return [{"text": "今天天气怎么样", "label": "weather"}] * n


def generate_music(n: int) -> list:
    """生成音乐播放意图样本 TODO: 实现模板填充"""
    return [{"text": "播放一首歌", "label": "music"}] * n


def generate_dataset(
    output_dir: str = "./dataset_intent",
    context_ratio: float = 0.35,
    seed: int = 42,
):
    """生成意图识别分流数据集 TODO: 补充各意图样本数参数"""
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("意图识别分流数据集生成（骨架）")
    print("=" * 60)

    # TODO: 调用各意图生成函数并汇总
    dataset = []
    # dataset.extend(generate_weather(n=1000))
    # dataset.extend(generate_music(n=1000))
    # ...

    print(f"\n基础样本总计: {len(dataset)}条")

    # 划分数据集
    train_data, val_data, test_data = split_dataset(dataset)
    save_jsonl(train_data, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(val_data, os.path.join(output_dir, "val.jsonl"))
    save_jsonl(test_data, os.path.join(output_dir, "test.jsonl"))
    save_jsonl(dataset, os.path.join(output_dir, "dataset_all.jsonl"))

    print("\n✅ 数据集生成完成！")
    return train_data, val_data, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成意图识别分流数据集")
    parser.add_argument("--output", "-o", type=str, default="./dataset_intent", help="输出目录")
    parser.add_argument("--context-ratio", type=float, default=0.35, help="多轮上下文比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()
    generate_dataset(
        output_dir=args.output,
        context_ratio=args.context_ratio,
        seed=args.seed,
    )
