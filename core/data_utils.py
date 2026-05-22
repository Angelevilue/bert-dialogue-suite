#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯数据工具函数（零重依赖）
包含数据生成、划分、保存等不需要 torch/sklearn 的工具。
"""

import json
import os
import random
import re


def fill_template(template: str, fillers: dict) -> str:
    """填充模板中的占位符"""
    result = template
    for key in re.findall(r'\{(\w+)\}', template):
        if key in fillers:
            result = result.replace('{' + key + '}', random.choice(fillers[key]), 1)
    return result


def split_dataset(samples: list, train_ratio=0.8, val_ratio=0.1):
    """按 8:1:1 分层划分训练/验证/测试集，确保每类标签在各集合中都有分布"""
    by_label = {}
    for s in samples:
        label = s["label"]
        by_label.setdefault(label, []).append(s)

    train_data, val_data, test_data = [], [], []
    for label, label_samples in by_label.items():
        random.shuffle(label_samples)
        total = len(label_samples)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        train_data.extend(label_samples[:train_end])
        val_data.extend(label_samples[train_end:val_end])
        test_data.extend(label_samples[val_end:])

    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    return train_data, val_data, test_data


def add_context(samples: list, context_ratio: float = 0.35) -> list:
    """为部分样本添加多轮对话上下文（默认模板）"""
    context_templates = {
        "ready": [
            ("今天天气不错", "明天呢"),
            ("播放周杰伦的歌", "换一首"),
            ("打开空调", "温度调低一点"),
        ],
        "unsafe": [
            ("你好", "我想问个问题"),
        ],
        "waiting": [
            ("帮我查一下", "那个..."),
        ],
        "confused": [
            ("你好", "嗯..."),
        ],
        "non_dialogue": [
            ("测试", "一二三四"),
        ],
    }

    by_label = {}
    for s in samples:
        label = s["label"]
        by_label.setdefault(label, []).append(s)

    result = []
    for label, label_samples in by_label.items():
        n_contextual = int(len(label_samples) * context_ratio)
        indices = list(range(len(label_samples)))
        random.shuffle(indices)
        context_indices = set(indices[:n_contextual])
        templates = context_templates.get(label, [])

        for i, s in enumerate(label_samples):
            if i not in context_indices or not templates:
                result.append(s)
                continue
            template = random.choice(templates)
            n_turns = min(random.randint(2, 4), len(template))
            if n_turns == 1:
                turns = [s["text"]]
            else:
                turns = list(template[:n_turns - 1]) + [s["text"]]
            text = " [SEP] ".join(turns)
            result.append({"text": text, "label": label})

    random.shuffle(result)
    return result


def save_jsonl(samples: list, filepath: str):
    """保存 JSONL 文件"""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
