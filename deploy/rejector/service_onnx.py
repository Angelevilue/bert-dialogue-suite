#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bert 拒识模块 ONNX 推理服务
使用 ONNX Runtime 替代 PyTorch，部署更轻量

用法:
    python service_onnx.py --model onnx/bert_rejector_int8.onnx
    python service_onnx.py --model onnx/bert_rejector.onnx --port 8001
    nohup python service_onnx.py > service.log 2>&1 &
"""

import argparse
import os
import sys
import time
import json
import logging

import numpy as np
from transformers import AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import onnxruntime as ort


# ============== 配置 ==============
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8089
DEFAULT_MODEL_PATH = None  # 默认: checkpoint/onnx/bert_rejector_fp16.onnx (FP16)
# ============== 配置 ==============

logger = logging.getLogger("rejector")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # 模型在 main() 中已加载


app = FastAPI(
    title="Bert Rejector ONNX Service",
    description="拒识模块 ONNX 推理服务（INT8 量化版）",
    version="1.0.0",
    lifespan=lifespan,
)

# 全局变量
ort_session = None
tokenizer = None
id2label = None
label2id = None


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    scores: list
    id: int
    inference_time_ms: float


class BatchPredictRequest(BaseModel):
    texts: list[str]


class BatchPredictResponse(BaseModel):
    results: list
    total_time_ms: float
    avg_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_path: str
    providers: list


def softmax(x, axis=-1):
    """numpy 实现 softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def load_model(model_path: str):
    """加载 ONNX 模型和 tokenizer"""
    global ort_session, tokenizer, id2label, label2id

    # 获取 checkpoint 目录（用于加载 tokenizer）
    checkpoint_dir = os.path.dirname(os.path.dirname(os.path.abspath(model_path)))

    # 加载 ONNX 模型
    logger.info(f"加载 ONNX 模型: {model_path}")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # 自动选择最优 providers（优先 GPU 加速）
    available_providers = ort.get_available_providers()
    providers = []
    if "CoreMLExecutionProvider" in available_providers:
        providers.append("CoreMLExecutionProvider")
    elif "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available_providers:
        providers.append("CPUExecutionProvider")
    providers = providers or available_providers

    ort_session = ort.InferenceSession(model_path, sess_options, providers=providers)

    # 加速方式说明
    used_provider = ort_session.get_providers()[0]
    if used_provider == "CoreMLExecutionProvider":
        accel = "Apple Neural Engine / GPU"
    elif used_provider == "CUDAExecutionProvider":
        accel = "NVIDIA GPU (CUDA)"
    elif used_provider == "CPUExecutionProvider":
        accel = "CPU"
    else:
        accel = used_provider
    logger.info(f"可用推理引擎: {available_providers}")
    logger.info(f"使用推理引擎: {ort_session.get_providers()}, 加速方式: {accel}")

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    # 加载标签映射
    label_map_path = os.path.join(checkpoint_dir, "label_map.json")
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    id2label = label_map["id2label"]
    label2id = label_map["label2id"]

    logger.info(f"模型加载完成，标签数: {len(id2label)}")


def preprocess(text: str) -> dict:
    """文本预处理"""
    # 处理 [SEP] 多轮对话
    if " [SEP] " in text:
        parts = text.split(" [SEP] ")
        if len(parts) >= 2:
            text_a, text_b = parts[0], " [SEP] ".join(parts[1:])
            encoding = tokenizer(
                text_a,
                text_b,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors="np",
            )
        else:
            encoding = tokenizer(
                text,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors="np",
            )
    else:
        encoding = tokenizer(
            text,
            max_length=128,
            padding=True,
            truncation=True,
            return_tensors="np",
        )

    return {
        "input_ids": encoding["input_ids"].astype(np.int64),
        "attention_mask": encoding["attention_mask"].astype(np.int64),
    }


def predict_text(text: str) -> dict:
    """预测单条文本"""
    global ort_session, tokenizer, id2label

    inputs = preprocess(text)

    # ONNX 推理
    outputs = ort_session.run(
        None,
        {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
    )

    logits = outputs[0]
    probs = softmax(logits, axis=-1)[0]
    pred_id = int(np.argmax(probs))
    confidence = float(probs[pred_id])
    scores = probs.tolist()

    return {
        "label": id2label[str(pred_id)],
        "confidence": confidence,
        "scores": scores,
        "id": pred_id,
    }


@app.get("/", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy",
        model_path=MODEL_PATH or "",
        providers=ort_session.get_providers() if ort_session else [],
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """单条预测"""
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    result = predict_text(request.text)
    elapsed = (time.perf_counter() - start) * 1000

    logger.info(f"[{result['label']}] conf={result['confidence']:.4f}  time={elapsed:.1f}ms  text=\"{request.text[:50]}{'...' if len(request.text) > 50 else ''}\"")

    return PredictResponse(
        label=result["label"],
        confidence=result["confidence"],
        scores=result["scores"],
        id=result["id"],
        inference_time_ms=round(elapsed, 2),
    )


@app.post("/predict_batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """批量预测"""
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    results = [predict_text(text) for text in request.texts]
    total_elapsed = (time.perf_counter() - start) * 1000

    for i, (text, result) in enumerate(zip(request.texts, results)):
        logger.info(f"[{i+1}/{len(results)}] [{result['label']}] conf={result['confidence']:.4f}  text=\"{text[:50]}{'...' if len(text) > 50 else ''}\"")
    logger.info(f"Batch total: {total_elapsed:.1f}ms, avg: {total_elapsed/len(results):.1f}ms")

    return BatchPredictResponse(
        results=results,
        total_time_ms=round(total_elapsed, 2),
        avg_time_ms=round(total_elapsed / len(request.texts), 2) if request.texts else 0,
    )


def main():
    global MODEL_PATH

    parser = argparse.ArgumentParser(description="Bert Rejector ONNX Service")
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"监听地址 (默认: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"监听端口 (默认: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"ONNX 模型路径",
    )
    args = parser.parse_args()

    # 确定模型路径
    if args.model:
        MODEL_PATH = args.model
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        MODEL_PATH = os.path.join(script_dir, "checkpoint", "onnx", "bert_rejector_fp16.onnx")

    # 加载模型
    load_model(MODEL_PATH)

    logger.info("=" * 60)
    logger.info("Bert Rejector ONNX Service")
    logger.info("=" * 60)
    logger.info(f"模型路径: {MODEL_PATH}")
    logger.info(f"服务地址: http://{args.host}:{args.port}")
    logger.info("=" * 60)
    logger.info("启动服务中...")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
