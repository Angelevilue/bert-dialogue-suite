# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **bert-dialogue-suite**, a modular BERT fine-tuning suite for Chinese dialogue systems. It currently contains one production-ready module and two planned modules:

- **`rejector/`** — 5-class text classification rejection module (拒识模块) for a Chinese family robot dialogue system (VLA_Robot). Labels: `unsafe`, `non_dialogue`, `waiting`, `ready`, `confused`. Deployed on VLA_Robot family robots.
- **`intent_router/`** — Skeleton for intent classification and routing (in progress).
- **`visual_detector/`** — Skeleton for visual vs non-visual question detection (planned).

The repository follows a **monorepo** structure: shared training/inference/data utilities live in `core/`, while each task module owns its own data generation templates, label mappings, and training configuration.

## Common Commands

### Install Dependencies

No `requirements.txt` exists. Install manually:

```bash
pip install transformers torch tqdm scikit-learn numpy
```

For GPU training with CUDA:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Generate Dataset (Rejection Module)

```bash
python rejector/generate_data.py --output ./dataset
```

Defaults output to `./dataset/` with ~5600 samples and 35% multi-turn context ratio. Key flags:
- `--output`: output directory
- `--ready`, `--unsafe`, `--waiting`, `--non_dialogue`, `--confused`: per-class sample counts
- `--context-ratio`: proportion of samples with `[SEP]` multi-turn context (0.0–1.0)
- `--seed`: random seed for reproducibility

### Train Model (Rejection Module)

```bash
python rejector/train.py \
    --train_data dataset/train.jsonl \
    --val_data dataset/val.jsonl \
    --test_data dataset/test.jsonl \
    --output_dir ./output
```

Key training flags (defined in `core/cli.py`, overridable per module):
- `--model_name`: default is `hfl/chinese-macbert-base` (or module `config.py` default)
- `--batch_size`: default 32
- `--epochs`: default 10
- `--lr`: default 2e-5
- `--max_length`: default 128
- `--patience`: early stopping patience, default 3
- `--fp16`: enable mixed precision training (requires GPU)
- `--save_every_epoch`: save checkpoint after every epoch

Training artifacts are written to `./output/<model_name>_<timestamp>/`, including:
- `checkpoints/best/`: best model by validation Macro-F1
- `history.json`: per-epoch metrics
- `inference_example.py`: auto-generated inference script

### Run Inference

Import the predictor class from `core`:

```python
import sys; sys.path.insert(0, ".")
from core.predictor import BertPredictor

predictor = BertPredictor("./output/.../checkpoints/best")
result = predictor.predict("今天天气怎么样")
# Returns: {"label": "ready", "confidence": 0.98, "scores": [...]}
```

## High-Level Architecture

### Data Format

All datasets use **JSONL** with exactly two fields per line:

```jsonl
{"text": "播放音乐 [SEP] 换一首", "label": "ready"}
```

- `text`: user utterance. Multi-turn dialogue history is joined with ` [SEP] ` (space-bracket-SEP-bracket-space).
- `label`: task-specific string label (defined in each module's `config.py`).

### Core Package (`core/`)

- **`core/dataset.py`** — `DialogueDataset`: loads JSONL, handles `[SEP]` splitting by passing parts as `text` and `text_pair` to the tokenizer. Accepts `label2id` mapping as a constructor argument.
- **`core/trainer.py`** — Generic training loop: AdamW with weight decay, linear warmup + decay scheduler, gradient clipping, fp16 mixed precision, and early stopping based on validation Macro-F1. The `train()` function accepts `label2id`, `id2label`, and `model_name` as parameters instead of relying on globals.
- **`core/predictor.py`** — `BertPredictor`: inference class that loads a saved checkpoint and reads `label_map.json` for dynamic label mapping. Handles `[SEP]` tokenization identically to training.
- **`core/utils.py`** — `save_model`, `print_report`, `create_dataloader`, `split_dataset` (stratified), `fill_template`, `add_context`, `save_jsonl`, `save_inference_script`.
- **`core/cli.py`** — `build_argparser()`: shared CLI argument parser used by all task modules.

### Task Modules (`rejector/`, `intent_router/`, `visual_detector/`)

Each module contains:
- `config.py` — `LABEL2ID`, `ID2LABEL`, `MODEL_NAME`, default hyperparameters
- `train.py` — thin wrapper that imports `core.trainer.train` and passes module-specific config
- `generate_data.py` — task-specific synthetic data generation with templates and fillers
- `README.md` — module-level documentation

### Deployment Package (`deploy/rejector/`)

- `service.py` — HTTP API service (PyTorch backend)
- `service_onnx.py` — HTTP API service (ONNX Runtime backend, faster and lighter)
- `export_onnx.py` — Export trained model to ONNX format (FP32/FP16/INT8)
- `start_service.sh` — Unified launch script (PyTorch by default, `--onnx` for ONNX mode)
- `client_ros2.py` — ROS2 client for integrating with robot systems
- `checkpoint/` — Model weights and tokenizer (gitignored)
- `checkpoint/onnx/` — Exported ONNX models (gitignored)

### Reference Documentation

Detailed manuals exist in `rejector/docs/` and should be consulted for parameter tuning, data construction rules, and deployment integration patterns:
- `rejector/docs/Fine-tuning Guide.md`: training parameters, evaluation metrics, deployment examples (REST API, async, filter integration).
- `rejector/docs/Data Construction Guide.md`: label definitions, data format spec, per-category construction rules, quality checks, and extension guidelines.

## Critical Invariants

- **Label mappings are owned by each module's `config.py`**. `core` never hardcodes labels; it receives `label2id` / `id2label` as arguments. If a module's labels change, only that module's `config.py` needs updating.
- **`[SEP]` handling must match** between `core.dataset.DialogueDataset.__getitem__` and `core.predictor.BertPredictor.predict`. The splitting logic (`text.split(" [SEP] ")`) and tokenizer arguments (`text` + `text_pair`) are duplicated and must remain identical.
- **No formal package installation required**: `core/` is a plain directory of modules. Task scripts use `sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))` to import `core`. This preserves the "run directly" philosophy.
- **Each module is self-contained**: A module's `train.py` and `generate_data.py` must be runnable from the repo root without cross-importing other modules (except `core`).
- **Model weights and data files are gitignored**: All `*.safetensors`, `*.onnx`, `*.jsonl`, `dataset/`, `output/`, and `deploy/*/checkpoint/` are excluded from version control.
