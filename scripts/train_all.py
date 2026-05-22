#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键训练所有模块
遍历各任务目录，自动检测并执行 train.py。
注意：骨架模块（intent_router / visual_detector）需先填充数据生成模板并生成数据集。
"""

import os
import subprocess
import sys

MODULES = [
    "rejection",
    "intent_router",
    "visual_detector",
]

# 各模块默认数据路径（相对于项目根目录）
MODULE_DATA_PATHS = {
    "rejection": {
        "train_data": "dataset/train.jsonl",
        "val_data": "dataset/val.jsonl",
        "test_data": "dataset/test.jsonl",
    },
    "intent_router": {
        "train_data": "dataset_intent/train.jsonl",
        "val_data": "dataset_intent/val.jsonl",
        "test_data": "dataset_intent/test.jsonl",
    },
    "visual_detector": {
        "train_data": "dataset_visual/train.jsonl",
        "val_data": "dataset_visual/val.jsonl",
        "test_data": "dataset_visual/test.jsonl",
    },
}


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    for module in MODULES:
        train_script = os.path.join(module, "train.py")
        if not os.path.exists(train_script):
            print(f"[跳过] {module}/train.py 不存在")
            continue

        data_paths = MODULE_DATA_PATHS.get(module, {})
        train_data = data_paths.get("train_data")
        val_data = data_paths.get("val_data")
        test_data = data_paths.get("test_data")

        # 检查数据文件是否存在
        missing = []
        for key in ["train_data", "val_data"]:
            path = data_paths.get(key)
            if path and not os.path.exists(path):
                missing.append(path)
        if missing:
            print(f"\n[跳过] {module}: 数据文件不存在，请先生成数据集")
            for m in missing:
                print(f"  - {m}")
            continue

        print(f"\n{'='*60}")
        print(f"开始训练模块: {module}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable,
            train_script,
            "--train_data", train_data,
            "--val_data", val_data,
            "--output_dir", "./output",
        ]
        if test_data and os.path.exists(test_data):
            cmd.extend(["--test_data", test_data])

        result = subprocess.run(cmd, cwd=project_root)

        if result.returncode != 0:
            print(f"[警告] {module} 训练脚本异常退出 (code: {result.returncode})")

    print("\n所有模块训练流程结束。")


if __name__ == "__main__":
    main()
