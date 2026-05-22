#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用工具函数
包含模型保存、评估报告、数据加载、数据生成辅助函数等。
"""

import json
import logging
import os
from typing import Dict, List

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


# ============================================================
# 训练/评估工具
# ============================================================

def save_model(
    model,
    tokenizer,
    output_dir: str,
    label2id: Dict = None,
    id2label: Dict = None,
):
    """保存模型和 Tokenizer"""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if label2id and id2label:
        with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}},
                f,
                ensure_ascii=False,
                indent=2,
            )

    logger.info(f"模型已保存到: {output_dir}")


def print_report(labels: np.ndarray, preds: np.ndarray, id2label: dict, title: str = "验证集"):
    """打印分类报告和混淆矩阵"""
    target_names = [id2label.get(i, str(i)) for i in range(len(id2label))]

    sep_width = max(50, len(target_names) * 12)
    logger.info("=" * sep_width)
    logger.info(f"{title} 分类报告")
    logger.info("=" * sep_width)
    report = classification_report(
        labels, preds,
        target_names=target_names,
        digits=4,
    )
    for line in report.splitlines():
        logger.info(line)

    cm = confusion_matrix(labels, preds)
    logger.info(f"{title} 混淆矩阵:")
    header = f"{'':>15}" + "".join(f"{name[:8]:>10}" for name in target_names)
    logger.info(header)
    for i, name in enumerate(target_names):
        row = f"{name[:14]:>15}" + "".join(f"{cm[i][j]:>10}" for j in range(len(target_names)))
        logger.info(row)
    logger.info("=" * sep_width)


def save_inference_script(output_dir: str, class_name: str = "BertPredictor"):
    """保存推理示例脚本"""
    model_dir = os.path.join(output_dir, "checkpoints", "best")
    script = f'''#!/usr/bin/env python3
"""推理示例脚本"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from core.predictor import {class_name}

# 加载模型
predictor = {class_name}("{model_dir}")

# 单条预测
texts = [
    "今天天气怎么样",
    "帮我播放周杰伦的歌 [SEP] 换一首",
]

for text in texts:
    result = predictor.predict(text)
    print(f"文本: {{text[:50]}}")
    print(f"预测: {{result['label']}} (置信度: {{result['confidence']:.4f}})")
    print(f"分数: {{result['scores']}}")
    print()
'''.replace("{model_dir}", model_dir.replace("\\", "/"))

    script_path = os.path.join(output_dir, "inference_example.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)
