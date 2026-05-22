#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT 模型导出为 ONNX 格式
用于量化部署

用法:
    python export_onnx.py                          # 导出 FP16 (默认)
    python export_onnx.py --precision fp32        # 导出 FP32
    python export_onnx.py --precision fp16        # 导出 FP16
    python export_onnx.py --checkpoint ./checkpoint
    python export_onnx.py --dynamic-batch         # 支持动态 batch
    python export_onnx.py --no-optimize           # 不导出 INT8
"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def export_onnx(
    checkpoint_path: str,
    output_dir: str = None,
    dynamic_batch: bool = False,
    optimize: bool = True,
    precision: str = "fp32",
):
    """
    将 BERT 模型导出为 ONNX 格式

    Args:
        checkpoint_path: 模型检查点目录
        output_dir: 输出目录，默认 <checkpoint_path>/onnx/
        dynamic_batch: 是否支持动态 batch（第一维可变）
        optimize: 是否导出 INT8 量化版本
    """
    if output_dir is None:
        output_dir = os.path.join(checkpoint_path, "onnx")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("BERT Rejector ONNX 导出")
    print("=" * 60)
    print(f"模型路径: {checkpoint_path}")
    print(f"输出路径: {output_dir}")
    print(f"动态Batch: {dynamic_batch}")
    print(f"优化: {optimize}")
    print(f"精度: {precision.upper()}")
    print("=" * 60)

    # 加载模型
    print("\n[1/3] 加载模型...")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    model.eval()

    # 加载 tokenizer（用于验证）
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # 准备示例输入
    print("\n[2/3] 生成示例输入...")
    sample_text = "今天天气怎么样"
    encoding = tokenizer(
        sample_text,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  attention_mask shape: {attention_mask.shape}")

    # 转换精度 - 只转换模型，不转换输入
    if precision == "fp16":
        print("\n[转换] 转换为 FP16...")
        model = model.half()
        # input_ids 和 attention_mask 保持 FP32（embedding 需要整数索引）

    # 导出 ONNX
    print("\n[导出] 导出 ONNX...")

    if dynamic_batch:
        # 动态 batch + 动态序列长度
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "attention_mask": {0: "batch_size", 1: "seq_length"},
            "logits": {0: "batch_size"},
        }
    else:
        # 固定 batch，动态序列长度
        dynamic_axes = {
            "input_ids": {1: "seq_length"},
            "attention_mask": {1: "seq_length"},
            "logits": {0: "batch_size"},
        }

    output_path = os.path.join(output_dir, f"bert_rejector_{precision}.onnx")

    torch.onnx.export(
        model,
        args=(input_ids, attention_mask),
        f=output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes if dynamic_batch else None,
        opset_version=14,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ 导出成功: {output_path}")
    print(f"   文件大小: {file_size:.2f} MB")

    # 验证导出
    print("\n[验证] 检查 ONNX 模型...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("   ONNX 模型检查通过")

    # 量化
    if optimize:
        print("\n[量化] INT8 量化...")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quantized_path = os.path.join(output_dir, "bert_rejector_int8.onnx")
            quantize_dynamic(
                output_path,
                quantized_path,
                weight_type=QuantType.QInt8,
            )
            q_size = os.path.getsize(quantized_path) / (1024 * 1024)
            print(f"   量化成功: {quantized_path}")
            print(f"   文件大小: {q_size:.2f} MB (压缩率: {file_size/q_size:.1f}x)")
        except ImportError:
            print("   onnxruntime 不支持量化，跳过")

    print("\n" + "=" * 60)
    print("导出完成")
    print("=" * 60)
    print(f"\n导出的模型:")
    print(f"  FP32: {os.path.join(output_dir, 'bert_rejector_fp32.onnx')}")
    print(f"  FP16: {os.path.join(output_dir, 'bert_rejector_fp16.onnx')}")
    if optimize:
        print(f"  INT8: {os.path.join(output_dir, 'bert_rejector_int8.onnx')}")
    print("\n使用方法（Python）:")
    print("""
import onnxruntime as ort

# FP32 推理
sess = ort.InferenceSession("bert_rejector.onnx")
outputs = sess.run(None, {
    "input_ids": input_ids.numpy(),
    "attention_mask": attention_mask.numpy()
})

# INT8 推理（更快）
sess_int8 = ort.InferenceSession("bert_rejector_int8.onnx")
""")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="BERT Rejector ONNX 导出")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="模型路径 (默认: ./checkpoint)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="支持动态 batch",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="禁用 INT8 量化",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16"],
        default="fp16",
        help="导出精度 (默认: fp16)",
    )
    args = parser.parse_args()

    if args.checkpoint is None:
        args.checkpoint = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint")

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        dynamic_batch=args.dynamic_batch,
        optimize=not args.no_optimize,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()