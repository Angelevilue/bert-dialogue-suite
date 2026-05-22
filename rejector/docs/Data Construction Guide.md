# Data Construction Guide

## Data Format

All datasets use **JSONL** format with exactly two fields per line:

```jsonl
{"text": "用户输入文本", "label": "标签名"}
```

### Example Data

```jsonl
{"text": "今天天气怎么样", "label": "ready"}
{"text": "播放音乐 [SEP] 换一首", "label": "ready"}
{"text": "你去写作业", "label": "non_dialogue"}
{"text": "你", "label": "waiting"}
{"text": "你是谁做的", "label": "confused"}
{"text": "给我看看", "label": "unsafe"}
```

## Label Definitions

| Label | Meaning | Example |
|-------|---------|---------|
| `unsafe` | Unsafe content (violence, porn, illegal) | "给我看看黄色网站" |
| `non_dialogue` | Non-dialogue input (noise, self-talk) | "去写作业" |
| `waiting` | Incomplete input (sentence cut off, missing object) | "播放" |
| `ready` | Normal input (clear intent) | "今天天气怎么样" |
| `confused` | Unclear intent (vague, asking about robot itself) | "你是谁做的" |

## Multi-turn Dialogue

Multi-turn dialogue uses `[SEP]` as separator:

```jsonl
{"text": "播放音乐 [SEP] 换一首", "label": "ready"}
{"text": "讲个故事 [SEP] 再来一个", "label": "ready"}
{"text": "你是谁 [SEP] 你是男生还是女生", "label": "confused"}
```

The `[SEP]` token represents conversation history between user and robot.

## Data Generation

### Generate Dataset

```bash
python rejector/generate_data.py --output ./dataset
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--output` | `./dataset` | Output directory |
| `--ready` | 1500 | Number of `ready` samples |
| `--unsafe` | 500 | Number of `unsafe` samples |
| `--waiting` | 500 | Number of `waiting` samples |
| `--non_dialogue` | 500 | Number of `non_dialogue` samples |
| `--confused` | 500 | Number of `confused` samples |
| `--context_ratio` | 0.35 | Proportion with `[SEP]` multi-turn context (0.0-1.0) |
| `--seed` | 42 | Random seed |

### Custom Sample Counts

```bash
python rejector/generate_data.py \
    --output ./dataset \
    --ready 2000 \
    --unsafe 300 \
    --waiting 500 \
    --non_dialogue 500 \
    --confused 500 \
    --context_ratio 0.4
```

### Generate All Classes Equally

```bash
python rejector/generate_data.py \
    --output ./balanced_dataset \
    --ready 1000 \
    --unsafe 1000 \
    --waiting 1000 \
    --non_dialogue 1000 \
    --confused 1000
```

## Quality Checks

### Data Validation

After generation, the script outputs statistics:

```
Dataset generated: ./dataset/
├── train.jsonl    (4480 samples)
├── val.jsonl      (560 samples)
└── test.jsonl     (560 samples)

Class distribution (train):
├── ready:         1200 (26.8%)
├── unsafe:        400  (8.9%)
├── waiting:       400  (8.9%)
├── non_dialogue:  400  (8.9%)
└── confused:      400  (8.9%)
```

### Manual Verification

Recommended checks:
1. Verify samples per class match expected counts
2. Spot-check 10-20 samples per class for label correctness
3. Ensure `[SEP]` samples have proper conversation context
4. Check for empty texts or malformed JSONL

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Class imbalance | Unequal sample counts | Use balanced counts |
| Overfitting | Too few samples | Increase sample count |
| Poor generalization | Biased templates | Add diverse templates |
| `[SEP]` not working | Tokenization mismatch | Check tokenizer config |

## Template Design

### Ready Templates (clear intent)

```python
ready_templates = [
    "{weather_query}",
    "{music_request}",
    "{story_request}",
    "{joke_request}",
    "{news_query}",
    "{alarm_set}",
    "{timer_set}",
    "{reminder_set}",
]
```

### Unsafe Templates

Must include clear unsafe indicators:
- Violence: "打", "杀", "攻击"
- Porn: "黄色", "色情", "裸"
- Illegal: "毒品", "赌博", "诈骗"

### Waiting Templates

Characteristics:
- Missing object: "播放" (what to play?)
- Incomplete phrase: "给我讲个" (tell me a what?)
- Trailing punctuation: "今天天气怎么样?" (acceptable, not waiting)

### Non-dialogue Templates

Characteristics:
- Commands not directed at robot: "去写作业", "闭嘴"
- Self-talk: "今天真开心" (no question)
- Noise: "asdf", "12345"

### Confused Templates

Characteristics:
- Asking about robot identity: "你是谁", "你叫什么"
- Vague requests: "给我看看", "弄一下"
- Contradictory: "放首歌但不要放音乐"

## Extending Data

### Add New Labels

1. Edit `rejector/config.py` — add to `LABEL2ID` and `ID2LABEL`
2. Edit `rejector/generate_data.py` — add template generator
3. Regenerate dataset with new class count

### Add New Templates

1. Edit template lists in `rejector/generate_data.py`
2. Ensure templates cover diverse phrasings
3. Regenerate dataset

### Augment Existing Data

```bash
# Generate more samples with different seed
python rejector/generate_data.py --output ./dataset_v2 --seed 123
```

## Dataset Files

After generation:

```
dataset/
├── train.jsonl   # Training set (80%)
├── val.jsonl     # Validation set (10%)
└── test.jsonl    # Test set (10%)
```

Files are gitignored by default (see `.gitignore`).

## Related

See also: [Fine-tuning Guide.md](Fine-tuning%20Guide.md) for training parameters and deployment.