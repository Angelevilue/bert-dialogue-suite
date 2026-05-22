#!/usr/bin/env python3
"""推理示例脚本"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from core.predictor import BertPredictor

# 加载模型
predictor = BertPredictor("./data/rejection_output/chinese-macbert-base_0521_0330/checkpoints/best")

# 单条预测
texts = [
    "今天天气怎么样",
    "帮我播放周杰伦的歌 [SEP] 换一首",
]

for text in texts:
    result = predictor.predict(text)
    print(f"文本: {text[:50]}")
    print(f"预测: {result['label']} (置信度: {result['confidence']:.4f})")
    print(f"分数: {result['scores']}")
    print()
