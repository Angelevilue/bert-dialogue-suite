#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拒识模块推理测试脚本
加载最佳模型，支持单条/多轮对话推理，输出预测结果和推理时间
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.predictor import BertPredictor

MODEL_PATH = "/Users/zephyrmuse/Work/Bert-Fine_tuning/bert-dialogue-suite/data/rejection_output/chinese-macbert-base_0521_0330/checkpoints/best"


def main():
    print("=" * 60)
    print("拒识模块推理测试")
    print("=" * 60)
    print(f"模型路径: {MODEL_PATH}")
    print()

    print("加载模型中...")
    predictor = BertPredictor(MODEL_PATH)
    print("模型加载完成！\n")

    print("输入文本测试，支持 [SEP] 分隔多轮对话")
    print("输入 q 或 quit 退出\n")

    while True:
        try:
            text = input(">>> ").strip()
        except EOFError:
            break

        if text.lower() in ["q", "quit", "exit"]:
            print("退出")
            break

        if not text:
            continue

        # 推理并计时
        start = time.perf_counter()
        result = predictor.predict(text)
        elapsed = (time.perf_counter() - start) * 1000  # ms

        # 输出结果
        label = result["label"]
        confidence = result["confidence"]
        scores = result["scores"]

        print(f"  标签: {label}")
        print(f"  置信度: {confidence:.4f}")
        print(f"  推理耗时: {elapsed:.2f} ms")
        print(f"  各标签分数: unsafe={scores[0]:.4f} non_dialogue={scores[1]:.4f} waiting={scores[2]:.4f} ready={scores[3]:.4f} confused={scores[4]:.4f}")
        print()


if __name__ == "__main__":
    main()
