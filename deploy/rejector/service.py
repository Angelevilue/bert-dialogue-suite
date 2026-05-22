#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bert 拒识模块推理服务
提供 HTTP API 供外部调用

用法:
    python service.py                    # 前台运行
    python service.py --host 0.0.0.0 --port 8089
    python service.py --device cuda     # 指定 GPU 设备
    nohup python service.py > service.log 2>&1 &  # 后台运行
"""

import argparse
import sys
import os
import time
import logging

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from contextlib import asynccontextmanager

from predictor import BertPredictor


# ============== 配置 ==============
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8089
DEFAULT_DEVICE = None  # None 表示自动选择 cuda > mps > cpu
# ============== 配置 ==============

logger = logging.getLogger("rejector")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# 全局预测器实例（启动时加载，常驻内存）
predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = BertPredictor(CHECKPOINT_PATH, device=DEVICE)
    logger.info(f"模型加载完成，设备: {predictor.device}")
    yield
    predictor = None


app = FastAPI(
    title="Bert Rejector Service",
    description="拒识模块推理服务 - 5分类 (unsafe/non_dialogue/waiting/ready/confused)",
    version="1.0.0",
    lifespan=lifespan,
)


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
    device: str
    model_loaded: bool


@app.get("/", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        device=str(predictor.device) if predictor else "not_loaded",
        model_loaded=predictor is not None,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    单条文本预测

    请求:
        {"text": "今天天气怎么样"}

    响应:
        {
            "label": "ready",
            "confidence": 0.99,
            "scores": [0.001, 0.002, 0.003, 0.984, 0.01],
            "id": 3,
            "inference_time_ms": 10.5
        }
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    result = predictor.predict(request.text)
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
    """
    批量文本预测

    请求:
        {"texts": ["文本1", "文本2", "文本3"]}

    响应:
        {
            "results": [...],
            "total_time_ms": 35.5,
            "avg_time_ms": 11.8
        }
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    results = predictor.predict_batch(request.texts)
    total_elapsed = (time.perf_counter() - start) * 1000

    for i, (text, result) in enumerate(zip(request.texts, results)):
        logger.info(f"[{i+1}/{len(results)}] [{result['label']}] conf={result['confidence']:.4f}  text=\"{text[:50]}{'...' if len(text) > 50 else ''}\"")
    logger.info(f"Batch total: {total_elapsed:.1f}ms, avg: {total_elapsed/len(results):.1f}ms")

    return BatchPredictResponse(
        results=results,
        total_time_ms=round(total_elapsed, 2),
        avg_time_ms=round(total_elapsed / len(request.texts), 2) if request.texts else 0,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Bert Rejector Service")
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"服务监听地址 (默认: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"服务监听端口 (默认: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "mps", "auto"],
        default=DEFAULT_DEVICE,
        help="推理设备 (默认: auto)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="模型路径 (默认: ./checkpoint)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 确定设备
    global DEVICE
    DEVICE = args.device if args.device != "auto" else None

    # 确定模型路径
    global CHECKPOINT_PATH
    if args.checkpoint:
        CHECKPOINT_PATH = args.checkpoint
    else:
        CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint")

    logger.info("=" * 60)
    logger.info("Bert Rejector Service")
    logger.info("=" * 60)
    logger.info(f"模型路径: {CHECKPOINT_PATH}")
    logger.info(f"设备: {DEVICE or 'auto'}")
    logger.info(f"服务地址: http://{args.host}:{args.port}")
    logger.info("=" * 60)
    logger.info("启动服务中...")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
    )
