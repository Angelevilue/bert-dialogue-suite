# bert-dialogue-suite

BERT 对话系统多任务微调套件，支持多分类文本分类，未来计划支持 NER。

## 介绍

**bert-dialogue-suite** 是一个生产级别的对话系统模块微调框架。采用 monorepo 结构：共享的训练/推理工具位于 `core/`，各任务模块拥有独立的数据生成模板、标签映射和训练配置。

目前 **拒识模块** 已投入生产，部署在家庭机器人（VLA_Robot）上，在用户输入进入下游对话引擎前，过滤不安全、不完整或意图不明的输入。框架设计为可扩展，方便新增分类任务。

## 模块列表

| 模块 | 状态 | 说明 |
|------|------|------|
| `rejector/` | 已完成 | 5 分类拒识模块（`unsafe`、`non_dialogue`、`waiting`、`ready`、`confused`） |
| `intent_router/` | 骨架 | 意图识别与分流（TODO：填充模板） |
| `visual_detector/` | 骨架 | 视觉问题判断（TODO：填充模板） |

## 项目结构

```
.
├── core/                           # 通用训练/推理/数据基础设施
│   ├── dataset.py                  # DialogueDataset（支持 [SEP] 多轮对话）
│   ├── data_utils.py               # 分层划分、模板填充、save_jsonl（零重依赖）
│   ├── trainer.py                  # 通用训练循环（含早停）
│   ├── predictor.py                # BertPredictor（单条/批量推理）
│   ├── utils.py                    # 模型保存、评估报告、推理脚本生成
│   └── cli.py                      # 通用 CLI 参数解析
├── rejector/                      # 拒识模块
│   ├── config.py                   # 标签映射与模型配置
│   ├── train.py                    # 训练入口
│   ├── generate_data.py            # 合成数据生成器
│   └── docs/                       # 详细手册（中文）
│       ├── BERT模型微调手册.md      # 微调指南
│       └── 数据构造和生成手册.md     # 数据构造指南
├── deploy/rejector/                # 部署包（ONNX/HTTP/ROS2）
│   ├── checkpoint/                  # 模型权重和 tokenizer（已 gitignore）
│   ├── checkpoint/onnx/             # ONNX 模型（FP32/FP16/INT8）
│   ├── service.py                   # HTTP API（PyTorch 版）
│   ├── service_onnx.py             # HTTP API（ONNX Runtime 版）
│   ├── export_onnx.py              # ONNX 导出脚本
│   ├── client_ros2.py              # ROS2 客户端
│   └── start_service.sh            # 统一启动脚本
├── intent_router/                  # 意图识别分流（骨架）
├── visual_detector/                # 视觉问题判断（骨架）
├── models/                         # 预训练模型（已加入 .gitignore）
├── scripts/
│   └── train_all.py                # 一键顺序训练所有模块
└── docs/
    └── architecture.md             # 架构说明
```

## 快速开始

### 安装依赖

```bash
conda create -n bert_fine_tuning python=3.12 -y
conda activate bert_fine_tuning
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 下载预训练模型

训练前先将预训练模型下载到 `./models/` 目录：

```bash
modelscope download --model hfl/chinese-macbert-base \
    pytorch_model.bin config.json vocab.txt \
    tokenizer.json tokenizer_config.json special_tokens_map.json \
    --local_dir ./models/chinese-macbert-base
```

训练时指定本地路径：

```bash
python rejector/train.py \
    --model_name ./models/chinese-macbert-base \
    ...
```

### 拒识模块

```bash
# 1. 生成数据集
python rejector/generate_data.py --output ./dataset

# 2. 训练
python rejector/train.py \
    --train_data dataset/train.jsonl \
    --val_data dataset/val.jsonl \
    --test_data dataset/test.jsonl \
    --output_dir ./output

# 3. 推理
python -c "
import sys; sys.path.insert(0, '.')
from core.predictor import BertPredictor
p = BertPredictor('./output/.../checkpoints/best')
print(p.predict('今天天气怎么样'))
"
```

## 硬件支持

训练和推理脚本会自动检测并使用最优加速器：

| 硬件 | 后端 | 混合精度 (`--fp16`) |
|------|------|---------------------|
| NVIDIA 显卡 | `cuda` | 支持 |
| Apple Silicon (M1/M2/M3/M5) | `mps` | 不支持（自动禁用） |
| CPU | `cpu` | 不支持 |

```bash
# 查看可用的后端
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

## 部署

训练完成后，导出并部署为 HTTP 服务：

```bash
# 导出 ONNX 模型
python deploy/rejector/export_onnx.py --precision fp16

# 启动服务（PyTorch 或 ONNX）
./deploy/rejector/start_service.sh --onnx --port 8089
```

详细部署说明见 [`deploy/rejector/README.md`](deploy/rejector/README.md)。

### 性能对比（MacBook Pro M3，单条请求）

| 后端 | 设备 | 精度 | 置信度 | 推理耗时 |
|------|------|------|--------|----------|
| PyTorch | MPS (Metal) | FP32 | ~99% | 42~96ms (平均 ~71ms) |
| ONNX Runtime | CoreML (Apple GPU/NE) | FP32 | 98.6% | 12~17ms |
| ONNX Runtime | CoreML (Apple GPU/NE) | **FP16** | 98.6% | 6~19ms |
| ONNX Runtime | CoreML (Apple GPU/NE) | INT8 | 71~80% | 3~6ms |

**推荐 ONNX FP16**：精度不损失，模型减半，推理速度最快。

## 添加新模块

1. 复制 `intent_router/` 到新目录，如 `ner/`
2. 修改 `config.py` 中的标签映射和模型名称
3. 在 `generate_data.py` 中实现数据生成模板
4. 运行 `python ner/train.py --train_data ... --val_data ...`

## License

MIT
