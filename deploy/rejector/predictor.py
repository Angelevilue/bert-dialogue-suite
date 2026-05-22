#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BertPredictor - BERT 模型推理类
从 checkpoint 目录加载模型和标签映射，支持单条/多轮对话推理
"""

import os
import json
from typing import Dict, List, Union

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BertPredictor:
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        初始化预测器

        Args:
            checkpoint_path: 模型检查点目录，包含：
                          - model.safetensors (模型权重)
                          - label_map.json (标签映射)
                          - tokenizer.json / vocab.txt 等 (分词器文件)
            device: 推理设备，"cuda" / "cpu" / "mps"，默认自动选择
        """
        self.checkpoint_path = checkpoint_path

        # 自动选择设备
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # 加载标签映射
        label_map_path = os.path.join(checkpoint_path, "label_map.json")
        with open(label_map_path, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)
        self.id2label = self.label_map["id2label"]
        self.label2id = self.label_map["label2id"]

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        # 加载模型
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> Dict:
        """
        预测单条文本

        Args:
            text: 输入文本，支持 [SEP] 分隔的多轮对话

        Returns:
            {
                "label": 预测标签 (str),
                "confidence": 最高概率 (float),
                "scores": 各标签分数 list[float],
                "id": 预测标签的 id (int)
            }
        """
        # 处理 [SEP] 多轮对话
        if " [SEP] " in text:
            parts = text.split(" [SEP] ")
            if len(parts) >= 2:
                # 第一部分作为 text，后续作为 text_pair
                text_a, text_b = parts[0], " [SEP] ".join(parts[1:])
                encoding = self.tokenizer(
                    text_a,
                    text_b,
                    max_length=128,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
            else:
                encoding = self.tokenizer(
                    text,
                    max_length=128,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
        else:
            encoding = self.tokenizer(
                text,
                max_length=128,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred_id = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_id].item()
            scores = probs[0].cpu().tolist()

        return {
            "label": self.id2label[str(pred_id)],
            "confidence": confidence,
            "scores": scores,
            "id": pred_id,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        批量预测

        Args:
            texts: 输入文本列表

        Returns:
            预测结果列表，每个元素同 predict() 返回值
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results
