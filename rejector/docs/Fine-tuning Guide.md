# BERT Fine-tuning Guide

## Training Parameters

### Basic Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `hfl/chinese-macbert-base` | Pretrained model path or HuggingFace model ID |
| `--batch_size` | 32 | Training batch size |
| `--epochs` | 10 | Number of training epochs |
| `--lr` | 2e-5 | Learning rate |
| `--max_length` | 128 | Maximum sequence length |
| `--patience` | 3 | Early stopping patience (epochs without improvement) |
| `--warmup_ratio` | 0.1 | Warmup ratio for learning rate scheduler |
| `--weight_decay` | 0.01 | Weight decay |
| `--dropout` | 0.1 | Dropout rate |

### Advanced Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fp16` | false | Enable mixed precision training (requires GPU) |
| `--save_every_epoch` | false | Save checkpoint after every epoch |
| `--eval_every_step` | 0 | Evaluate every N steps (0 = after each epoch) |
| `--logging_steps` | 10 | Log every N steps |

### Device Selection

The system automatically selects the best available device:

```bash
# Force specific device
python rejector/train.py --device cuda   # NVIDIA GPU
python rejector/train.py --device mps    # Apple Silicon
python rejector/train.py --device cpu    # CPU

# Auto-detect (default)
python rejector/train.py --device auto
```

### Training Output

Training artifacts are written to `./output/<model_name>_<timestamp>/`:

```
output/chinese-macbert-base_20260521_143000/
├── checkpoints/
│   ├── best/                  # Best model by validation Macro-F1
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer/
│   │   └── label_map.json
│   └── last/                 # Last epoch checkpoint
├── history.json              # Per-epoch metrics
├── train.log                 # Training log
└── inference_example.py      # Auto-generated inference script
```

## Evaluation Metrics

### Metrics Computed

- **Accuracy** — Overall correct predictions / total predictions
- **Macro-F1** — F1 score averaged across all classes (primary metric)
- **Weighted-F1** — F1 score weighted by class support
- **Per-class Precision/Recall/F1** — Detailed per-class metrics

### Best Model Selection

The best model is selected based on **validation Macro-F1** score. This metric is preferred because:
- It weighs all classes equally, important when classes are imbalanced
- It reflects overall classification quality across all categories

### Typical Results (Rejector Module)

| Dataset | Accuracy | Macro-F1 | Weighted-F1 |
|---------|----------|----------|-------------|
| Validation | 97.10% | 96.24% | 97.10% |
| Test | 97.70% | 97.12% | 97.69% |

## Deployment

### Step 1: Export to ONNX

```bash
python deploy/rejector/export_onnx.py --precision fp16
```

Outputs to `deploy/rejector/checkpoint/onnx/bert_rejector_fp16.onnx`

### Step 2: Start HTTP Service

```bash
# ONNX mode (recommended for production)
./deploy/rejector/start_service.sh --onnx --port 8089

# PyTorch mode (for debugging)
./deploy/rejector/start_service.sh --port 8089
```

### Step 3: Call the API

```bash
# Health check
curl http://localhost:8089/

# Single prediction
curl -X POST http://localhost:8089/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "今天天气怎么样"}'

# Batch prediction
curl -X POST http://localhost:8089/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["今天天气怎么样", "明天会下雨吗"]}'
```

### Response Format

```json
{
  "label": "ready",
  "confidence": 0.9994,
  "scores": [0.001, 0.002, 0.003, 0.984, 0.01],
  "id": 3,
  "inference_time_ms": 10.5
}
```

## Integration with Dialogue System

### Filter Pattern

In your dialogue system, call the rejector before entering the core dialogue engine:

```
User Input → Rejector Service → [if ready] → Dialogue Engine → Response
                        ↓
               [if unsafe/waiting/confused/non_dialogue] → Rejection Response
```

### ROS2 Integration

```python
from client_ros2 import RejectorROS2Node

node = RejectorROS2Node(service_url="http://192.168.x.x:8089")
```

Topics:
- `dialog_text` (input) — User utterance
- `rejector_result` (output) — Full prediction result
- `is_ready` (output) — Boolean: whether input is ready for dialogue engine

## Performance Tuning

### GPU Training (NVIDIA)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Train with mixed precision
python rejector/train.py --fp16 --batch_size 64
```

### Apple Silicon (MPS)

```bash
# MPS is auto-detected, no special installation needed
python rejector/train.py --batch_size 32
```

### CPU Training (not recommended for production)

```bash
python rejector/train.py --batch_size 16 --epochs 5
```

## Troubleshooting

### Out of Memory

- Reduce `--batch_size`
- Reduce `--max_length`
- Enable gradient checkpointing (not implemented yet)

### Slow Training

- Use GPU (cuda/mps) instead of CPU
- Enable `--fp16` for faster training on supported GPUs
- Increase `--batch_size` if memory allows

### Model Not Improving

- Check data quality and label distribution
- Try different learning rates (1e-5 to 5e-5)
- Increase `--patience` for more epochs before early stopping
- Verify `[SEP]` token handling matches between training and inference