#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 BERT 分类训练循环
支持早停、Warmup、fp16、自动保存最佳模型。
"""

import json
import logging
import os
import random
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score

from .dataset import create_dataloader
from .utils import save_model, print_report, save_inference_script

logger = logging.getLogger(__name__)


def get_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    """创建带 Warmup 的线性学习率调度器"""
    return get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
    use_fp16: bool = False,
    epoch: int = 0,
) -> Dict[str, float]:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    scaler = torch.cuda.amp.GradScaler() if use_fp16 and torch.cuda.is_available() else None

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(outputs.logits, dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{total_correct / total_samples:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return {"loss": avg_loss, "accuracy": avg_acc}


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int = 0,
    desc: str = "Val",
) -> Dict[str, float]:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"{desc} Epoch {epoch}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            total_loss += loss.item() * labels.size(0)

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
    }


def train(
    args,
    label2id: dict,
    id2label: dict,
    model_name: str = None,
):
    """
    主训练流程

    Args:
        args: argparse Namespace，包含所有训练超参
        label2id: 标签到 ID 的映射
        id2label: ID 到标签的映射
        model_name: 预训练模型名称，默认使用 args.model_name
    """
    model_name = model_name or args.model_name
    num_labels = len(label2id)

    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设备：优先 CUDA，其次 MPS (Apple Silicon)，最后 CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"使用设备: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # 创建输出目录
    timestamp = datetime.now().strftime("%m%d_%H%M")
    output_dir = os.path.join(args.output_dir, f"{model_name.split('/')[-1]}_{timestamp}")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # 保存配置
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        config = vars(args).copy()
        config["label2id"] = label2id
        config["id2label"] = id2label
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 加载 Tokenizer 和模型
    logger.info(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    logger.info(f"可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")

    # 创建数据加载器
    logger.info("加载数据集...")
    train_loader = create_dataloader(
        args.train_data, tokenizer, args.batch_size, args.max_length,
        label2id=label2id, shuffle=True,
    )
    val_loader = create_dataloader(
        args.val_data, tokenizer, args.batch_size, args.max_length,
        label2id=label2id, shuffle=False,
    )

    test_loader = None
    if args.test_data:
        if os.path.exists(args.test_data):
            test_loader = create_dataloader(
                args.test_data, tokenizer, args.batch_size, args.max_length,
                label2id=label2id, shuffle=False,
            )
        else:
            logger.warning(f"测试集不存在: {args.test_data}，跳过测试集评估")

    # 优化器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

    # 学习率调度
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_scheduler(optimizer, warmup_steps, total_steps)

    logger.info(f"总训练步数: {total_steps}")
    logger.info(f"Warmup步数: {warmup_steps}")
    logger.info(f"学习率: {args.lr}")
    logger.info(f"批大小: {args.batch_size}")
    logger.info(f"最大长度: {args.max_length}")
    logger.info(f"混合精度: {args.fp16}")

    # 基线测试：评估未微调的预训练模型
    logger.info("\n" + "=" * 60)
    logger.info("基线测试（预训练模型，未微调）")
    logger.info("=" * 60)
    baseline_metrics = evaluate(model, val_loader, device, desc="Baseline")
    logger.info(f"基线 Val Loss:    {baseline_metrics['loss']:.4f}")
    logger.info(f"基线 Val Acc:     {baseline_metrics['accuracy']:.4f}")
    logger.info(f"基线 Val Macro-F1: {baseline_metrics['macro_f1']:.4f}")
    logger.info(f"基线 Val Weighted-F1: {baseline_metrics['weighted_f1']:.4f}")
    print_report(baseline_metrics["labels"], baseline_metrics["preds"], id2label, "基线验证集")
    logger.info("=" * 60 + "\n")

    # 训练循环
    best_f1 = baseline_metrics["macro_f1"]  # 基线作为初始参考
    best_epoch = 0
    patience_counter = 0
    history = []

    logger.info("\n" + "=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, use_fp16=args.fp16, epoch=epoch,
        )
        val_metrics = evaluate(model, val_loader, device, epoch=epoch, desc="Val")

        record = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "train_acc": round(train_metrics["accuracy"], 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
            "val_weighted_f1": round(val_metrics["weighted_f1"], 4),
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(record)

        logger.info(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        # 保存最佳模型
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0
            best_dir = os.path.join(checkpoint_dir, "best")
            save_model(model, tokenizer, best_dir, label2id, id2label)
            logger.info(f"  >>> 最佳模型更新 (Macro-F1: {best_f1:.4f})")
        else:
            patience_counter += 1

        # 每个 epoch 都保存
        if args.save_every_epoch:
            epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch}")
            save_model(model, tokenizer, epoch_dir, label2id, id2label)

        # 早停
        if patience_counter >= args.patience:
            logger.info(f"\n早停触发！连续 {args.patience} 个 epoch 未提升")
            break

    # 训练结束
    logger.info("\n" + "=" * 60)
    logger.info("训练完成!")
    logger.info(f"最佳模型: Epoch {best_epoch}, Macro-F1: {best_f1:.4f}")
    logger.info("=" * 60)

    # 加载最佳模型进行测试
    logger.info("\n加载最佳模型进行评估...")
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(checkpoint_dir, "best"))
    model.to(device)

    val_metrics = evaluate(model, val_loader, device, desc="Val")
    print_report(val_metrics["labels"], val_metrics["preds"], id2label, "验证集")

    if test_loader:
        test_metrics = evaluate(model, test_loader, device, desc="Test")
        logger.info(f"\n测试集结果:")
        logger.info(f"  Loss:      {test_metrics['loss']:.4f}")
        logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        logger.info(f"  Macro-F1:  {test_metrics['macro_f1']:.4f}")
        logger.info(f"  Weighted-F1: {test_metrics['weighted_f1']:.4f}")
        print_report(test_metrics["labels"], test_metrics["preds"], id2label, "测试集")

    # 保存训练历史
    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # 保存推理示例脚本
    save_inference_script(output_dir)

    logger.info(f"\n所有文件已保存到: {output_dir}")
    return output_dir
