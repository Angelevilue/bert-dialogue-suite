# Visual Detector (视觉问题判断)

## Overview

TODO: 判断用户输入是否属于视觉相关问题（如颜色识别、物体识别、场景描述等）。

## Labels

| Label | Meaning |
|-------|---------|
| `visual` | 视觉相关问题 |
| `non_visual` | 非视觉问题 |

## Files

- `config.py` — 标签映射与模型配置
- `train.py` — 训练入口
- `generate_data.py` — 数据生成（TODO: 填充真实模板）
