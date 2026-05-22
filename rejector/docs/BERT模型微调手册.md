# VLA_Robot 拒识模块 - BERT 模型微调手册

> 文档版本: 1.0  
> 关联系统: VLA_Robot 对话系统 - 拒识模块  
> 最后更新: 2026-05-18

---

## 目录

1. [概述](#1-概述)
2. [模型选型](#2-模型选型)
3. [环境准备](#3-环境准备)
4. [微调脚本使用](#4-微调脚本使用)
5. [超参数调优指南](#5-超参数调优指南)
6. [训练过程监控](#6-训练过程监控)
7. [模型评估](#7-模型评估)
8. [部署与集成](#8-部署与集成)
9. [性能优化](#9-性能优化)
10. [常见问题排查](#10-常见问题排查)

---

## 1. 概述

### 1.1 文档目的

本手册指导完成拒识模块 BERT 模型的微调、评估和部署集成。涵盖从环境搭建到生产集成的完整流程。

### 1.2 脚本文件

| 文件 | 说明 |
|------|------|
| `rejector/train.py` | 微调主脚本（含训练、评估、推理） |

### 1.3 任务定义

- **任务类型**: 5 分类文本分类
- **标签**: `unsafe` | `non_dialogue` | `waiting` | `ready` | `confused`
- **输入**: 用户文本（支持 `[SEP]` 多轮对话格式）
- **输出**: 分类标签 + 置信度分数

---

## 2. 模型选型

### 2.1 推荐模型

| 优先级 | 模型名称 | 参数量 | 特点 | 适用场景 |
|--------|----------|--------|------|----------|
| **首选** | `hfl/chinese-macbert-base` | 102M | MacBERT 纠正预训练-微调偏差，中文全词掩码 | 标准部署 |
| **备选** | `hfl/chinese-bert-wwm-ext` | 102M | 全词掩码 BERT，经典强基线 | 稳定性优先 |
| **轻量** | `hfl/chinese-macbert-small` | ~30M | 推理速度快 3 倍 | 低延迟场景 |

### 2.2 为何选择 MacBERT

MacBERT 相比原生 BERT 的改进：

| 改进点 | 说明 | 效果 |
|--------|------|------|
| 纠正 [MASK] 偏差 | 用相似词替换替代 `[MASK]` | 消除预训练与微调不一致 |
| 全词掩码 (WWM) | 对整个词进行掩码 | 更好的中文语义理解 |
| 句子顺序预测 | 替代 NSP 任务 | 更好的句间关系理解 |

### 2.3 模型对比基准

在中文分类任务上，MacBERT 通常比原生 BERT 高 1-3 个百分点准确率，推理速度完全相同。

---

## 3. 环境准备

### 3.1 硬件要求

| 硬件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | 无 (CPU训练) | NVIDIA GPU >= 8GB 显存 |
| 内存 | 8GB | 16GB |
| 存储 | 2GB | 5GB (含模型文件) |

### 3.2 依赖安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install transformers torch tqdm scikit-learn numpy

# 如需 GPU 加速 (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"
```

### 3.3 数据准备

确保已生成数据集（参见《数据构造和生成手册》）：

```
project/
├── train.jsonl
├── val.jsonl
├── test.jsonl
└── rejector/train.py
```

---

## 4. 微调脚本使用

### 4.1 快速开始

```bash
python rejector/train.py \
    --train_data train.jsonl \
    --val_data val.jsonl \
    --test_data test.jsonl \
    --output_dir ./output
```

### 4.2 完整参数

```bash
python rejector/train.py \
    --train_data train.jsonl \
    --val_data val.jsonl \
    --test_data test.jsonl \
    --model_name hfl/chinese-macbert-base \
    --output_dir ./output \
    --batch_size 32 \
    --epochs 10 \
    --lr 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --max_length 128 \
    --patience 3 \
    --seed 42 \
    --fp16 \
    --save_every_epoch
```

### 4.3 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_data` | **必填** | 训练集 JSONL 路径 |
| `--val_data` | **必填** | 验证集 JSONL 路径 |
| `--test_data` | None | 测试集 JSONL 路径（可选） |
| `--model_name` | `hfl/chinese-macbert-base` | 预训练模型名称 |
| `--batch_size` | 32 | 批大小 |
| `--epochs` | 10 | 最大训练轮数 |
| `--lr` | 2e-5 | 学习率 |
| `--weight_decay` | 0.01 | 权重衰减（L2正则化） |
| `--warmup_ratio` | 0.1 | Warmup 步数占比 |
| `--max_length` | 128 | 最大序列长度 |
| `--patience` | 3 | 早停耐心（连续N轮无提升则停止） |
| `--output_dir` | `./output` | 输出目录 |
| `--seed` | 42 | 随机种子 |
| `--fp16` | False | 开启混合精度训练（需GPU） |
| `--save_every_epoch` | False | 每轮都保存检查点 |

### 4.4 输出结构

```
output/
└── chinese-macbert-base_0518_1430/     # 时间戳命名的实验目录
    ├── config.json                      # 训练配置快照
    ├── history.json                     # 每轮训练指标
    ├── inference_example.py             # 推理示例脚本
    └── checkpoints/
        ├── best/                        # 最佳模型（按 Val Macro-F1）
        │   ├── pytorch_model.bin        # 模型权重
        │   ├── config.json              # 模型配置
        │   ├── tokenizer.json           # 分词器
        │   ├── tokenizer_config.json
        │   ├── vocab.txt
        │   └── label_map.json           # 标签映射
        ├── epoch_1/                     # 第1轮检查点
        ├── epoch_2/                     # 第2轮检查点
        ...
```

### 4.5 首次运行示例输出

```
============================================================
BERT 拒识模块微调
============================================================
使用设备: cuda
GPU: NVIDIA GeForce RTX 3090
输出目录: ./output/chinese-macbert-base_0518_1430
加载模型: hfl/chinese-macbert-base
总参数量: 102,268,805 (102.27M)
可训练参数量: 102,268,805 (102.27M)
从 train.jsonl 加载了 4468 条样本
从 val.jsonl 加载了 558 条样本
从 test.jsonl 加载了 559 条样本
总训练步数: 1400
Warmup步数: 140
学习率: 2e-05
批大小: 32
最大长度: 128
混合精度: True

============================================================
开始训练
============================================================
Epoch  1/10 | Train Loss: 0.6234 | Val Loss: 0.3121 | Val Acc: 0.8925 | Val Macro-F1: 0.8712 | LR: 1.82e-05
  >>> 最佳模型更新 (Macro-F1: 0.8712)
Epoch  2/10 | Train Loss: 0.2456 | Val Loss: 0.1892 | Val Acc: 0.9410 | Val Macro-F1: 0.9323 | LR: 1.20e-05
  >>> 最佳模型更新 (Macro-F1: 0.9323)
Epoch  3/10 | Train Loss: 0.1234 | Val Loss: 0.1567 | Val Acc: 0.9523 | Val Macro-F1: 0.9456 | LR: 6.00e-06
  >>> 最佳模型更新 (Macro-F1: 0.9456)
Epoch  4/10 | Train Loss: 0.0789 | Val Loss: 0.1432 | Val Acc: 0.9587 | Val Macro-F1: 0.9512 | LR: 2.00e-06
  >>> 最佳模型更新 (Macro-F1: 0.9512)
Epoch  5/10 | Train Loss: 0.0567 | Val Loss: 0.1489 | Val Acc: 0.9562 | Val Macro-F1: 0.9487 | LR: 0.00e+00

============================================================
训练完成!
最佳模型: Epoch 4, Macro-F1: 0.9512
============================================================
```

---

## 5. 超参数调优指南

### 5.1 学习率 (Learning Rate)

```bash
# 尝试范围: 1e-5 ~ 5e-5
python rejector/train.py --lr 1e-5  # 更稳定
python rejector/train.py --lr 3e-5  # 较快收敛
python rejector/train.py --lr 5e-5  # 快速但可能震荡
```

| 场景 | 推荐值 | 原因 |
|------|--------|------|
| 数据量 < 5000 | 2e-5 | 保守收敛 |
| 数据量 > 10000 | 3e-5 | 可以适当激进 |
| 模型不收敛 | 1e-5 | 降低学习率 |
| 验证集震荡 | 1e-5 + 增大 batch_size | 稳定训练 |

### 5.2 批大小 (Batch Size)

| GPU显存 | 最大 batch_size | 效果 |
|---------|----------------|------|
| 8GB | 16 | 标准 |
| 16GB | 32 | 推荐 |
| 24GB+ | 64 | 更快收敛 |

### 5.3 最大序列长度 (Max Length)

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 单轮短文本为主 | 64 | 速度提升 40% |
| 含多轮上下文 | 128 | 标准配置 |
| 长对话场景 | 256 | 覆盖长历史 |

> 拒识场景以短文本为主，建议保持 128 或降至 64 提速。

### 5.4 早停策略

```bash
# 宽松早停（数据量大时）
python rejector/train.py --patience 5

# 严格早停（防止过拟合）
python rejector/train.py --patience 2 --epochs 10
```

### 5.5 快速调参流程

```bash
# Step 1: 基线实验
python rejector/train.py --lr 2e-5 --batch_size 32 --epochs 10

# Step 2: 调整学习率
python rejector/train.py --lr 3e-5  # 若欠拟合
python rejector/train.py --lr 1e-5  # 若过拟合

# Step 3: 调整批大小
python rejector/train.py --batch_size 64  # 若 GPU 允许

# Step 4: 确定最佳组合后完整训练
python rejector/train.py --lr 2e-5 --batch_size 64 --epochs 15 --patience 5
```

---

## 6. 训练过程监控

### 6.1 关键指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| Val Acc | 验证集准确率 | > 93% |
| Val Macro-F1 | 宏平均 F1 | > 92% |
| Val Loss | 验证集损失 | 持续下降后平稳 |
| Train/Val Gap | 训练与验证差距 | < 5%（防止过拟合） |

### 6.2 正常训练曲线

```
Loss
  |    Train  Val
  |     \\   /
  |      \\ /
  |       X
  |      / \\
  |     /   \\
  +-------------------> Epoch
        1 2 3 4 5
```

- 训练损失持续下降
- 验证损失先降后平稳
- 最佳模型通常在验证损失最低点

### 6.3 异常诊断

| 现象 | 原因 | 解决方案 |
|------|------|----------|
| 训练 loss 不降 | 学习率过低 | 提高到 3e-5 或 5e-5 |
| 验证 loss 持续上升 | 过拟合 | 增大 weight_decay，降低 lr |
| 训练/验证差距大 | 过拟合 | 增加数据，降低模型复杂度 |
| Macro-F1 远低于 Acc | 类别不平衡 | 检查各类别样本数，考虑加权损失 |
| 某类别 F1 特别低 | 样本不足或质量差 | 补充该类数据 |

### 6.4 训练历史查看

```python
import json

with open("output/.../history.json", "r") as f:
    history = json.load(f)

for record in history:
    print(f"Epoch {record['epoch']}: "
          f"Val Acc={record['val_acc']:.4f}, "
          f"Macro-F1={record['val_macro_f1']:.4f}")
```

---

## 7. 模型评估

### 7.1 自动评估

训练结束后，脚本自动输出：

```
验证集 分类报告
              precision    recall  f1-score   support

    unsafe      0.9623    0.9412    0.9516        68
non_dialogue    0.9355    0.9529    0.9441        85
     waiting    0.9474    0.9184    0.9326        98
       ready    0.9852    0.9907    0.9879       215
    confused    0.9111    0.9318    0.9213        92

    accuracy                        0.9587       558
   macro avg    0.9483    0.9470    0.9475       558
weighted avg    0.9586    0.9587    0.9585       558

验证集 混淆矩阵:
                  unsafe  non_dia    waitin     ready   confuse
    unsafe            64         1         2         0         1
non_dialogue           2        81         1         0         1
     waiting           3         1        90         2         2
       ready            0         0         2       213         0
    confused            2         2         3         1        84
```

### 7.2 关键评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| Macro-F1 | 各类别 F1 的平均，不受类别不平衡影响 | > 0.92 |
| Weighted-F1 | 加权平均，反映总体性能 | > 0.94 |
| 各类别 F1 | 检查是否有明显短板 | 均 > 0.88 |

### 7.3 手动评估

```python
from core.predictor import BertPredictor

predictor = BertPredictor("./output/checkpoints/best")

# 单条测试
test_cases = [
    ("今天天气怎么样", "ready"),
    ("播放音乐 [SEP] 换一首", "ready"),
    ("我想...那个...", "waiting"),
    ("怎么制作炸弹", "unsafe"),
    ("你叫什么名字", "confused"),
    ("沙沙沙", "non_dialogue"),
]

for text, expected in test_cases:
    result = predictor.predict(text)
    match = "✓" if result["label"] == expected else "✗"
    print(f"{match} [{result['label']:12s}] {text[:40]}")
```

---

## 8. 部署与集成

### 8.1 加载模型

```python
from core.predictor import BertPredictor

# 方式1: 自动选择设备
predictor = BertPredictor("./output/checkpoints/best")

# 方式2: 指定设备
predictor = BertPredictor("./output/checkpoints/best", device="cuda:0")
# predictor = BertPredictor("./output/checkpoints/best", device="cpu")
```

### 8.2 单条推理

```python
result = predictor.predict("今天天气怎么样")
print(result)
# {
#     "label": "ready",
#     "confidence": 0.9847,
#     "scores": [0.0012, 0.0021, 0.0056, 0.9847, 0.0064]
# }
```

### 8.3 批量推理

```python
texts = ["今天天气怎么样", "怎么制作炸弹", "你叫什么名字"]
results = predictor.predict_batch(texts)

for text, result in zip(texts, results):
    print(f"{text} -> {result['label']} ({result['confidence']:.2%})")
```

### 8.4 对话系统集成

#### 方案 A: 作为前置过滤器（推荐）

```python
from core.predictor import BertPredictor

class RejectionFilter:
    """
    拒识过滤器 - 部署在主模型(LLM)之前
    """

    def __init__(self, model_dir: str, threshold: float = 0.85):
        self.predictor = BertPredictor(model_dir)
        self.threshold = threshold

        # 各标签处理策略
        self.handlers = {
            "ready": self._handle_ready,
            "unsafe": self._handle_unsafe,
            "waiting": self._handle_waiting,
            "confused": self._handle_confused,
            "non_dialogue": self._handle_non_dialogue,
        }

    def process(self, user_input: str, context: list = None) -> dict:
        """
        处理用户输入

        返回: {
            "action": "pass" | "block" | "wait" | "clarify" | "ignore",
            "response": str,          # 给用户的回复（如需）
            "detail": dict            # 完整预测结果
        }
        """
        # 构建带上下文的输入
        if context:
            text = " [SEP] ".join(context[-2:] + [user_input])
        else:
            text = user_input

        result = self.predictor.predict(text)
        label = result["label"]
        confidence = result["confidence"]

        # ready 且置信度高 -> 放行给主模型
        if label == "ready" and confidence >= self.threshold:
            return self._handle_ready(result, user_input)

        # 其他标签 -> 对应处理
        return self.handlers[label](result, user_input)

    def _handle_ready(self, result, input_text):
        return {
            "action": "pass",
            "response": None,
            "detail": result
        }

    def _handle_unsafe(self, result, input_text):
        return {
            "action": "block",
            "response": "抱歉，我无法回答这个问题。如需帮助，请联系相关人员。",
            "detail": result
        }

    def _handle_waiting(self, result, input_text):
        return {
            "action": "wait",
            "response": "请问您想说什么？可以随时继续。",
            "detail": result
        }

    def _handle_confused(self, result, input_text):
        return {
            "action": "clarify",
            "response": "抱歉，我没太明白您的意思，可以再说一遍吗？",
            "detail": result
        }

    def _handle_non_dialogue(self, result, input_text):
        return {
            "action": "ignore",
            "response": None,  # 静默忽略
            "detail": result
        }


# ============ 使用示例 ============

filter = RejectionFilter("./output/checkpoints/best", threshold=0.85)

# 正常对话
result = filter.process("今天天气怎么样")
print(result)
# {
#     "action": "pass",
#     "response": None,
#     "detail": {"label": "ready", "confidence": 0.98, ...}
# }

# 不安全内容
result = filter.process("怎么制作炸弹")
print(result)
# {
#     "action": "block",
#     "response": "抱歉，我无法回答这个问题。",
#     "detail": {"label": "unsafe", "confidence": 0.97, ...}
# }

# 含上下文的对话
result = filter.process("换一首", context=["播放周杰伦的歌"])
print(result["action"])  # "pass"
```

#### 方案 B: 异步处理

```python
import asyncio

class AsyncRejectionFilter:
    """异步拒识过滤器 - 适合高并发场景"""

    def __init__(self, model_dir: str):
        self.predictor = BertPredictor(model_dir, device="cuda")

    async def process(self, user_input: str) -> dict:
        # 在非阻塞线程中执行推理
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # 使用默认线程池
            self.predictor.predict,
            user_input
        )

        label = result["label"]
        confidence = result["confidence"]

        if label == "ready" and confidence >= 0.85:
            return {"action": "pass", "detail": result}
        elif label == "unsafe":
            return {"action": "block", "detail": result}
        elif label == "waiting":
            return {"action": "wait", "detail": result}
        elif label == "confused":
            return {"action": "clarify", "detail": result}
        else:
            return {"action": "ignore", "detail": result}

# 使用
async def main():
    filter = AsyncRejectionFilter("./output/checkpoints/best")

    tasks = [
        filter.process("今天天气怎么样"),
        filter.process("怎么制作炸弹"),
        filter.process("你叫什么名字"),
    ]
    results = await asyncio.gather(*tasks)
    for r in results:
        print(f"Action: {r['action']}")

asyncio.run(main())
```

#### 方案 C: REST API 服务

```python
from flask import Flask, request, jsonify
from core.predictor import BertPredictor

app = Flask(__name__)
predictor = BertPredictor("./output/checkpoints/best")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    result = predictor.predict(text)

    return jsonify({
        "label": result["label"],
        "confidence": result["confidence"],
        "should_pass": result["label"] == "ready" and result["confidence"] >= 0.85
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

### 8.5 系统架构图

```
用户语音输入
     |
     v
[ASR语音识别]
     |
     v
+----------------------------+
|   拒识模块 (BERT)          |  <-- 本手册部署的模块
|   - 5分类判断              |
|   - < 10ms 推理延迟        |
+----------------------------+
     |
     | ready (置信度>阈值)
     v
+----------------------------+
|   主模型 (LLM)             |
|   - 意图识别               |
|   - 对话生成               |
+----------------------------+
     |
     v
[语音合成输出]
     |
     | unsafe / waiting / confused / non_dialogue
     v
+----------------------------+
|   拦截/等待/忽略处理       |
|   - 安全提示               |
|   - 等待补充               |
|   - 澄清追问               |
|   - 静默忽略               |
+----------------------------+
```

### 8.6 集成到 VLA_Robot 系统

```python
# vla_robot/core/rejection_module.py

from core.predictor import BertPredictor

class RejectionModule:
    """VLA_Robot 拒识模块"""

    def __init__(self, config):
        self.predictor = BertPredictor(
            model_dir=config["model_path"],
            device=config.get("device", "cuda")
        )
        self.threshold = config.get("threshold", 0.85)
        self.enable = config.get("enable", True)

    def check(self, asr_text: str, dialogue_history: list = None) -> dict:
        """
        检查 ASR 输出是否需要拒识

        Returns:
            {
                "blocked": bool,        # 是否被拦截
                "label": str,           # 分类标签
                "confidence": float,    # 置信度
                "reason": str,          # 拦截原因（如被拦截）
                "scores": list          # 各类别分数
            }
        """
        if not self.enable:
            return {"blocked": False, "label": "ready", "confidence": 1.0}

        # 构建带上下文的输入
        if dialogue_history:
            context = " [SEP] ".join(dialogue_history[-2:] + [asr_text])
        else:
            context = asr_text

        result = self.predictor.predict(context)

        label = result["label"]
        confidence = result["confidence"]

        # ready 且置信度高 -> 放行
        if label == "ready" and confidence >= self.threshold:
            return {
                "blocked": False,
                "label": label,
                "confidence": confidence,
                "scores": result["scores"]
            }

        # 其他类别 -> 拦截
        block_reasons = {
            "unsafe": "不安全内容",
            "waiting": "输入不完整",
            "confused": "意图不明确",
            "non_dialogue": "非对话输入"
        }

        return {
            "blocked": True,
            "label": label,
            "confidence": confidence,
            "reason": block_reasons.get(label, "未知原因"),
            "scores": result["scores"]
        }


# ============ 在对话主流程中使用 ============

# robot/dialogue_manager.py
class DialogueManager:
    def __init__(self, config):
        self.rejection = RejectionModule(config["rejection"])
        self.llm = LargeLanguageModel(config["llm"])
        self.asr = ASRModule(config["asr"])
        self.tts = TTSModule(config["tts"])

    def handle_user_input(self, audio_input):
        # 1. ASR 识别
        asr_text = self.asr.recognize(audio_input)

        # 2. 拒识检查
        check_result = self.rejection.check(
            asr_text,
            dialogue_history=self.history
        )

        if check_result["blocked"]:
            # 被拦截，不进入主模型
            response = self._get_rejection_response(check_result)
            return self.tts.synthesize(response)

        # 3. 进入主模型处理
        llm_response = self.llm.generate(asr_text, context=self.history)

        # 4. 更新对话历史
        self.history.append(asr_text)
        self.history.append(llm_response)

        # 5. 语音输出
        return self.tts.synthesize(llm_response)

    def _get_rejection_response(self, check_result):
        """根据拒识类型返回合适的回复"""
        responses = {
            "unsafe": "抱歉，这个问题我无法回答。",
            "waiting": "请继续，我在听。",
            "confused": "抱歉，我没听清楚，可以再说一遍吗？",
            "non_dialogue": ""  # 静默
        }
        return responses.get(check_result["label"], "请再说一次")
```

---

## 9. 性能优化

### 9.1 推理加速

| 方法 | 加速比 | 精度损失 | 实施难度 |
|------|--------|----------|----------|
| FP16 推理 | 1.5-2x | 极小 | 简单 |
| ONNX 导出 | 2-3x | 无 | 中等 |
| TensorRT | 3-5x | 极小 | 中等 |
| 模型量化 (INT8) | 2-4x | 小 | 中等 |
| 模型裁剪 (small) | 3x | 小 | 简单 |

### 9.2 导出 ONNX

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "./output/checkpoints/best"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# 导出 ONNX
dummy_input = tokenizer(
    "测试文本",
    return_tensors="pt",
    max_length=128,
    padding="max_length"
)

torch.onnx.export(
    model,
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    "rejection_model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"}
    },
    opset_version=14
)
print("ONNX 模型已导出")
```

### 9.3 模型量化

```python
from transformers import AutoModelForSequenceClassification

# 动态量化 (CPU 推理)
model = AutoModelForSequenceClassification.from_pretrained("./output/checkpoints/best")
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 保存量化模型
torch.save(quantized_model.state_dict(), "quantized_model.pt")
print("量化模型已保存")
```

### 9.4 批处理优化

```python
# 真正的批量推理（推荐用于生产）
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class BatchPredictor:
    def __init__(self, model_dir: str, batch_size: int = 32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size

    @torch.no_grad()
    def predict(self, texts: list) -> list:
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            # 批量编码
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt"
            )

            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            # 批量推理
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            for j, (pred, prob) in enumerate(zip(preds, probs)):
                label_id = pred.item()
                confidence = prob[label_id].item()
                results.append({
                    "label": ID2LABEL[label_id],
                    "confidence": round(confidence, 4),
                    "scores": [round(p, 4) for p in prob.cpu().tolist()]
                })

        return results
```

---

## 10. 常见问题排查

### Q1: 模型下载失败

```bash
# 设置镜像源（国内）
export HF_ENDPOINT=https://hf-mirror.com
python rejector/train.py ...

# 或手动下载后指定本地路径
python rejector/train.py --model_name ./local_models/chinese-macbert-base ...
```

### Q2: CUDA 显存不足 (OOM)

```bash
# 方案1: 减小 batch_size
python rejector/train.py --batch_size 16  # 或 8

# 方案2: 减小 max_length
python rejector/train.py --max_length 64

# 方案3: 开启梯度累积 (需修改脚本)
# 方案4: 使用 CPU 训练
python rejector/train.py --batch_size 8  # CPU 用更小的 batch
```

### Q3: 训练速度过慢

```bash
# 开启混合精度 (需 GPU)
python rejector/train.py --fp16

# 增大 batch_size (如显存允许)
python rejector/train.py --batch_size 64

# 减小 max_length
python rejector/train.py --max_length 64

# 使用 num_workers (DataLoader 多进程)
# 需在脚本中修改 create_dataloader 的 num_workers 参数
```

### Q4: 验证集指标异常

| 现象 | 排查方向 |
|------|----------|
| 所有样本预测同一类 | 检查标签映射；检查数据加载是否正确 |
| 准确率极低 (< 50%) | 检查标签是否反了；检查模型是否正确加载 |
| 某类别 F1 = 0 | 检查该类别在验证集中是否有样本 |
| 训练 loss 为 nan | 学习率过大；数据有 nan 值；开启 fp16 导致 |

### Q5: 模型推理结果不一致

```python
# 确保 eval 模式
model.eval()

# 确保没有梯度计算
with torch.no_grad():
    result = predictor.predict(text)

# 确保随机种子固定
torch.manual_seed(42)
```

### Q6: 如何增量训练

```python
# 加载已训练的模型作为起点
python rejector/train.py \
    --model_name ./output/checkpoints/best \  # 使用自己的模型作为基座
    --train_data new_data.jsonl \
    --val_data new_val.jsonl \
    --lr 1e-5          # 使用更低的学习率
    --epochs 5         # 更少的轮数
    --patience 2
```

### Q7: 如何多 GPU 训练

```python
# 使用 torchrun 启动
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 rejector/train.py ...

# 脚本内需添加 DistributedDataParallel 支持
# （当前版本为单 GPU/CPU 版本，如需多 GPU 请修改脚本）
```

---

## 附录

### A. 版本兼容性

| 组件 | 推荐版本 | 最低版本 |
|------|----------|----------|
| Python | 3.9+ | 3.7 |
| PyTorch | 2.0+ | 1.10 |
| transformers | 4.30+ | 4.20 |
| scikit-learn | 1.3+ | 1.0 |

### B. 训练配置文件示例

```json
{
  "model_name": "hfl/chinese-macbert-base",
  "train_data": "data/train.jsonl",
  "val_data": "data/val.jsonl",
  "test_data": "data/test.jsonl",
  "output_dir": "./output",
  "batch_size": 32,
  "epochs": 10,
  "lr": 2e-5,
  "weight_decay": 0.01,
  "warmup_ratio": 0.1,
  "max_length": 128,
  "patience": 3,
  "seed": 42,
  "fp16": true,
  "threshold": 0.85
}
```

### C. 性能基准参考

在 NVIDIA RTX 3090 上的测试结果：

| 指标 | 数值 |
|------|------|
| 模型参数量 | 102M |
| 模型大小 | ~400MB |
| 单条推理延迟 (GPU) | ~5ms |
| 单条推理延迟 (CPU) | ~50ms |
| 批处理 32 条 (GPU) | ~80ms |
| FP16 推理加速 | 1.8x |
| ONNX 推理加速 | 2.5x |
| 训练时间 (5600 样本, 10 epoch) | ~15 分钟 |
| 预期 Macro-F1 | 0.93 - 0.97 |
