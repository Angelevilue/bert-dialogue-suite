#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
拒识模块推理脚本
用法:
    python inference.py                    # 交互模式
    python inference.py "你的文本"          # 单条预测
"""

import sys
import os
import time
import json

# 将项目根目录加入路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from deploy.rejector import BertPredictor

# 模型路径（相对于本文件目录）
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint")


def main():
    if len(sys.argv) > 1:
        # 命令行模式：预测单条
        text = " ".join(sys.argv[1:])
        predictor = BertPredictor(MODEL_PATH)
        start = time.perf_counter()
        result = predictor.predict(text)
        elapsed = (time.perf_counter() - start) * 1000

        print(json.dumps({
            "text": text,
            "label": result["label"],
            "confidence": round(result["confidence"], 4),
            "inference_time_ms": round(elapsed, 2),
        }, ensure_ascii=False, indent=2))
    else:
        # 交互模式
        print("=" * 60)
        print("拒识模块推理测试")
        print("=" * 60)
        print(f"模型路径: {MODEL_PATH}\n")

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

            start = time.perf_counter()
            result = predictor.predict(text)
            elapsed = (time.perf_counter() - start) * 1000

            print(f"  标签: {result['label']}")
            print(f"  置信度: {result['confidence']:.4f}")
            print(f"  推理耗时: {elapsed:.2f} ms")
            print(f"  各标签分数: unsafe={result['scores'][0]:.4f} "
                  f"non_dialogue={result['scores'][1]:.4f} "
                  f"waiting={result['scores'][2]:.4f} "
                  f"ready={result['scores'][3]:.4f} "
                  f"confused={result['scores'][4]:.4f}")
            print()


if __name__ == "__main__":
    main()
