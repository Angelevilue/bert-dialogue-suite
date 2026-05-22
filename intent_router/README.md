# Intent Router (意图识别分流)

## Overview

TODO: 将用户输入分类到不同意图类别，用于后续分流到不同处理模块。

## Labels

TODO: 定义真实意图类别

| Label | Meaning |
|-------|---------|
| `weather` | 天气查询 |
| `music` | 音乐播放 |
| `device_control` | 设备控制 |
| ... | ... |

## Files

- `config.py` — 标签映射与模型配置
- `train.py` — 训练入口
- `generate_data.py` — 数据生成（TODO: 填充真实模板）
