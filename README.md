# bert-dialogue-suite

A modular BERT fine-tuning suite for Chinese dialogue systems, supporting multi-task text classification and NER (planned).

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| `rejector/` | Ready | 5-class rejection filter (`unsafe`, `non_dialogue`, `waiting`, `ready`, `confused`) |
| `intent_router/` | Skeleton | Intent classification and routing (TODO: fill templates) |
| `visual_detector/` | Skeleton | Visual vs non-visual question detection (TODO: fill templates) |

## Project Structure

```
.
├── core/                           # Shared training/inference/data utilities
│   ├── dataset.py                  # DialogueDataset (multi-turn [SEP] support)
│   ├── data_utils.py               # split_dataset, fill_template, save_jsonl (zero heavy deps)
│   ├── trainer.py                  # Generic training loop with early stopping
│   ├── predictor.py                # BertPredictor (single/batch inference)
│   ├── utils.py                    # save_model, print_report, save_inference_script
│   └── cli.py                      # Shared argparse builder
├── rejector/                      # Rejection module
│   ├── config.py                   # Label maps & model configuration
│   ├── train.py                    # Training entrypoint
│   ├── generate_data.py            # Synthetic data generator
│   └── docs/                       # Detailed manuals (Chinese)
│       ├── BERT模型微调手册.md      # Fine-tuning guide
│       └── 数据构造和生成手册.md     # Data construction guide
├── deploy/rejector/                # Deployment package (ONNX/HTTP/ROS2)
│   ├── checkpoint/onnx/             # ONNX models (FP32/FP16/INT8)
│   ├── service.py                   # HTTP API (PyTorch)
│   ├── service_onnx.py             # HTTP API (ONNX Runtime)
│   ├── export_onnx.py              # ONNX export script
│   ├── client_ros2.py              # ROS2 client
│   └── start_service.sh            # Unified launch script
├── intent_router/                  # Intent routing module (skeleton)
├── visual_detector/                # Visual question detector (skeleton)
├── models/                         # Pretrained models (gitignored)
├── scripts/
│   └── train_all.py                # Train all modules sequentially
└── docs/
    └── architecture.md             # Architecture guide
```

## Quick Start

### Install Dependencies

```bash
conda create -n bert_fine_tuning python=3.12 -y
conda activate bert_fine_tuning
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Download Pretrained Model

Download the pretrained model to `./models/` before training:

```bash
modelscope download --model hfl/chinese-macbert-base \
    pytorch_model.bin config.json vocab.txt \
    tokenizer.json tokenizer_config.json special_tokens_map.json \
    --local_dir ./models/chinese-macbert-base
```

Then use the local path in training:

```bash
python rejector/train.py \
    --model_name ./models/chinese-macbert-base \
    ...
```

### Rejection Module

```bash
# 1. Generate dataset
python rejector/generate_data.py --output ./dataset

# 2. Train
python rejector/train.py \
    --train_data dataset/train.jsonl \
    --val_data dataset/val.jsonl \
    --test_data dataset/test.jsonl \
    --output_dir ./output

# 3. Inference
python -c "
import sys; sys.path.insert(0, '.')
from core.predictor import BertPredictor
p = BertPredictor('./output/.../checkpoints/best')
print(p.predict('今天天气怎么样'))
"
```

## Hardware Support

The training and inference scripts automatically detect and use the best available accelerator:

| Hardware | Backend | Mixed Precision (`--fp16`) |
|----------|---------|---------------------------|
| NVIDIA GPU | `cuda` | Supported |
| Apple Silicon (M1/M2/M3/M5) | `mps` | Not supported (auto-disabled) |
| CPU | `cpu` | Not supported |

```bash
# Verify available backends
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

## Deployment

After training, export and deploy the model as an HTTP service:

```bash
# Export ONNX model
python deploy/rejector/export_onnx.py --precision fp16

# Start service (PyTorch or ONNX)
./deploy/rejector/start_service.sh --onnx --port 8089
```

For detailed deployment guide, see [`deploy/rejector/README.md`](deploy/rejector/README.md).

### Performance Comparison (MacBook Pro M3, single request)

| Backend | Device | Precision | Confidence | Inference Time |
|---------|--------|-----------|------------|----------------|
| PyTorch | MPS (Metal) | FP32 | ~99% | 42~96ms (avg ~71ms) |
| ONNX Runtime | CoreML (Apple GPU/NE) | FP32 | 98.6% | 12~17ms |
| ONNX Runtime | CoreML (Apple GPU/NE) | **FP16** | 98.6% | 6~19ms |
| ONNX Runtime | CoreML (Apple GPU/NE) | INT8 | 71~80% | 3~6ms |

**Recommendation:** ONNX FP16 offers the best balance — no accuracy loss, ~50% model size reduction, and fastest inference on Apple Silicon.

## Adding a New Module

1. Copy `intent_router/` to a new directory, e.g., `ner/`
2. Edit `config.py` with your label mappings and model name
3. Implement data generation templates in `generate_data.py`
4. Run `python ner/train.py --train_data ... --val_data ...`

## License

MIT
