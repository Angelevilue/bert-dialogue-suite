#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 BERT 分类推理器
从保存的模型目录加载标签映射，支持单条/批量预测。
"""

import os
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BertPredictor:
    """通用 BERT 分类推理器"""

    def __init__(self, model_dir: str, device: str = None):
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # 加载标签映射
        label_map_path = os.path.join(model_dir, "label_map.json")
        if os.path.exists(label_map_path):
            import json
            with open(label_map_path, "r", encoding="utf-8") as f:
                maps = json.load(f)
                self.id2label = {int(k): v for k, v in maps["id2label"].items()}
        else:
            self.id2label = {}

    @torch.no_grad()
    def predict(self, text: str, max_length: int = 128) -> Dict:
        """
        单条文本预测

        返回:
            {
                "label": str,
                "confidence": float,
                "scores": List[float]
            }
        """
        if " [SEP] " in text:
            parts = text.split(" [SEP] ")
            encoding = self.tokenizer(
                parts[0],
                text_pair=" [SEP] ".join(parts[1:]) if len(parts) > 1 else None,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()

        return {
            "label": self.id2label.get(pred_id, str(pred_id)),
            "confidence": round(confidence, 4),
            "scores": [round(p, 4) for p in probs[0].cpu().tolist()],
        }

    def predict_batch(self, texts: List[str], max_length: int = 128, batch_size: int = 32) -> List[Dict]:
        """批量预测（真正的 batch forward）"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            batch_encodings = {"input_ids": [], "attention_mask": []}
            for text in batch_texts:
                if " [SEP] " in text:
                    parts = text.split(" [SEP] ")
                    enc = self.tokenizer(
                        parts[0],
                        text_pair=" [SEP] ".join(parts[1:]) if len(parts) > 1 else None,
                        truncation=True,
                        max_length=max_length,
                        padding="max_length",
                        return_tensors="pt",
                    )
                else:
                    enc = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=max_length,
                        padding="max_length",
                        return_tensors="pt",
                    )
                batch_encodings["input_ids"].append(enc["input_ids"])
                batch_encodings["attention_mask"].append(enc["attention_mask"])

            input_ids = torch.cat(batch_encodings["input_ids"], dim=0).to(self.device)
            attention_mask = torch.cat(batch_encodings["attention_mask"], dim=0).to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            for pred, prob in zip(preds, probs):
                label_id = pred.item()
                confidence = prob[label_id].item()
                results.append({
                    "label": self.id2label.get(label_id, str(label_id)),
                    "confidence": round(confidence, 4),
                    "scores": [round(p, 4) for p in prob.cpu().tolist()],
                })
        return results
