#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用对话数据集类
支持 [SEP] 分隔的多轮对话文本，支持通过 label2id 注入标签映射。
"""

import json
import logging
from typing import Dict, List

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DialogueDataset(Dataset):
    """
    通用对话数据集
    支持 [SEP] 分隔的多轮对话文本
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 128,
        label2id: dict = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id or {}
        self.samples = self._load_data(data_path)

    def _load_data(self, path: str) -> List[Dict]:
        """加载 JSONL 文件"""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "").strip()
                    label = obj.get("label", "")
                    if not text:
                        logger.warning(f"第{line_num}行: 空文本，已跳过")
                        continue
                    if self.label2id and label not in self.label2id:
                        logger.warning(f"第{line_num}行: 未知标签 '{label}'，已跳过")
                        continue
                    samples.append({
                        "text": text,
                        "label": self.label2id.get(label, label),
                        "label_name": label,
                    })
                except (json.JSONDecodeError, AttributeError, TypeError) as e:
                    logger.warning(f"第{line_num}行: 解析失败 ({e})，已跳过")
        logger.info(f"从 {path} 加载了 {len(samples)} 条样本")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        text = item["text"]
        label = item["label"]

        # 处理 [SEP] 分隔的多轮对话
        if " [SEP] " in text:
            parts = text.split(" [SEP] ")
            encoding = self.tokenizer(
                parts[0],
                text_pair=" [SEP] ".join(parts[1:]) if len(parts) > 1 else None,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
        else:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def create_dataloader(
    data_path: str,
    tokenizer,
    batch_size: int,
    max_length: int,
    label2id: dict = None,
    shuffle: bool = True,
    num_workers: int = 0,
):
    """创建 DataLoader"""
    from torch.utils.data import DataLoader
    dataset = DialogueDataset(data_path, tokenizer, max_length, label2id=label2id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )
