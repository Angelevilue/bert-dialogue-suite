# Rejection Module (拒识模块)

## Overview

前置过滤器，将用户输入分类为 5 类之一，拦截不安全、不完整、非对话或意图不明的输入。

## Labels

| Label | Meaning |
|-------|---------|
| `unsafe` | 不安全内容 |
| `non_dialogue` | 非对话输入 |
| `waiting` | 不完整输入 |
| `ready` | 正常输入 |
| `confused` | 意图不明 |

## Quick Start

### 1. Generate Dataset

```bash
python rejector/generate_data.py --output ./dataset
```

### 2. Train

```bash
python rejector/train.py \
    --train_data dataset/train.jsonl \
    --val_data dataset/val.jsonl \
    --test_data dataset/test.jsonl \
    --output_dir ./output
```

### 3. Inference

```python
import sys; sys.path.insert(0, ".")
from core.predictor import BertPredictor

p = BertPredictor("./output/.../checkpoints/best")
print(p.predict("今天天气怎么样"))
```

## Files

- `config.py` — 标签映射与模型配置
- `train.py` — 训练入口
- `generate_data.py` — 合成数据生成
- `docs/` — 详细手册
