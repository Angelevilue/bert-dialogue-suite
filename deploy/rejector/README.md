# 拒识模块部署包

> BERT 5分类模型：unsafe / non_dialogue / waiting / ready / confused

## 目录结构

```
deploy/rejector/
├── checkpoint/              # PyTorch 模型和 tokenizer
│   ├── model.safetensors    # PyTorch 模型权重 (~400MB)
│   ├── config.json
│   ├── label_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── vocab.txt
├── checkpoint/onnx/        # ONNX 模型文件
│   ├── bert_rejector.onnx        # FP32 版本
│   ├── bert_rejector_fp16.onnx  # FP16 版本 (推荐)
│   └── bert_rejector_int8.onnx  # INT8 量化版本
├── predictor.py             # 推理核心类
├── __init__.py
├── inference.py             # 本地推理脚本（交互/单条）
├── service.py               # HTTP API 服务（PyTorch 版）
├── service_onnx.py          # HTTP API 服务（ONNX Runtime 版，更轻量）
├── export_onnx.py           # ONNX 导出脚本（支持 FP32/FP16/INT8）
├── client_ros2.py           # ROS2 客户端示例
├── start_service.sh         # 服务启动脚本
└── requirements.txt         # Python 依赖
```

## 相关文档

训练和微调详细说明见 [`rejector/docs/`](https://github.com/Angelevilue/bert-dialogue-suite/tree/main/rejector/docs)：
- `Fine-tuning Guide.md` — 微调参数、评估指标、部署示例
- `Data Construction Guide.md` — 数据格式规范、各类别构造规则、质量检查

## 依赖安装

### 1. 确认 CUDA 版本

```bash
nvcc --version
```

### 2. 安装 Python 依赖

**GPU 版本（NVIDIA Orin NX）：**

```bash
# 卸载 CPU 版本（如果有）
pip uninstall torch torchvision

# 安装 PyTorch GPU 版本（来自官方索引）
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118

# JetPack 6.x (CUDA 12.1):
# pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**验证 GPU 可用：**
```bash
python -c "import torch; print(torch.cuda.is_available())"
# 输出 True 表示安装成功
```

## 部署方式

### 方式一：本地推理（开发/测试）

```bash
# 交互模式
python inference.py

# 单条预测
python inference.py "今天天气怎么样"
```

### 方式二：HTTP API 服务（生产部署 - PyTorch）

**默认启动（不加 --onnx 参数则启动 PyTorch 版本）：**

```bash
# 前台运行
./start_service.sh

# 后台运行
./start_service.sh --background

# 指定参数
./start_service.sh --port 8089 --device cuda
```

### 方式三：HTTP API 服务（生产部署 - ONNX）

**优势：** 部署更轻量，推理速度快（~15ms/条），置信度高

两种启动方式（等效）：

```bash
# 方式A：使用统一启动脚本
./start_service.sh --onnx

# 方式B：直接调用 Python
python service_onnx.py
```

**指定参数：**
```bash
# 指定端口
./start_service.sh --onnx --port 8089
python service_onnx.py --port 8089

# 指定模型（默认 checkpoint/onnx/bert_rejector_fp16.onnx）
./start_service.sh --onnx --model checkpoint/onnx/bert_rejector_fp16.onnx
python service_onnx.py --model checkpoint/onnx/bert_rejector_fp16.onnx
```

**后台运行：**
```bash
./start_service.sh --onnx --background
nohup python service_onnx.py > service_onnx.log 2>&1 &
```

**ONNX 模型文件：**

| 文件 | 精度 | 大小 | 推荐度 |
|------|------|------|--------|
| `bert_rejector_fp16.onnx` | FP16 | 205 MB | ⭐ **推荐** |
| `bert_rejector.onnx` | FP32 | 409 MB | 高精度场景 |
| `bert_rejector_int8.onnx` | INT8 | 103 MB | 模型最小，但精度下降 |

**各模型性能对比（MacBook CPU 测试）：**

| 模型 | 置信度 | 推理耗时 |
|------|--------|----------|
| FP32 | 98.6% | 12~17ms |
| **FP16** | 98.6% | 6~19ms |
| INT8 | 71~80% | 3~6ms |

**推荐 FP16**：精度不损失，推理快，模型减半。

**ONNX 模型推理依赖（比 PyTorch 轻量很多）：**
```bash
pip install onnxruntime>=1.17.0 transformers>=4.30.0
```

**API 接口：**

| 接口 | 方法 | 说明 |
|------|------|------|
| `GET /` | 健康检查 | 返回服务状态 |
| `POST /predict` | 单条预测 | 请求: `{"text": "..."}` |
| `POST /predict_batch` | 批量预测 | 请求: `{"texts": ["...", "..."]}` |

**调用示例：**

```bash
# 健康检查
curl http://localhost:8089/

# 单条预测
curl -X POST http://localhost:8089/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "今天天气怎么样"}'

# 批量预测
curl -X POST http://localhost:8089/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["今天天气怎么样", "明天会下雨吗"]}'
```

## ROS2 集成

### 在 Docker 里的 ROS2 项目中调用服务

```python
# ros2_rejector_client.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from client_ros2 import RejectorROS2Node

def main(args=None):
    rclpy.init(args=args)
    # service_url 替换为 Orin NX 的 IP
    node = RejectorROS2Node(service_url="http://192.168.x.x:8089")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

### ROS2 Topic 说明

| Topic | 类型 | 说明 |
|-------|------|------|
| `dialog_text` (默认) | `std_msgs/String` | 输入文本 |
| `rejector_result` (默认) | `std_msgs/String` | 预测结果 |
| `is_ready` (默认) | `std_msgs/Bool` | 是否 ready（可用于后续流程过滤）|

### 直接使用（不依赖 ROS2）

```bash
# 测试服务连通性
python client_ros2.py --host 192.168.x.x --port 8089
```

## 推理结果说明

```json
{
  "label": "ready",
  "confidence": 0.9994,
  "scores": [0.001, 0.002, 0.003, 0.984, 0.01],
  "id": 3,
  "inference_time_ms": 10.5
}
```

**标签 id 对应关系：**

| id | 标签 | 含义 |
|----|------|------|
| 0 | unsafe | 不安全内容（暴力、色情、违法等）|
| 1 | non_dialogue | 非对话输入（噪音、自言自语等）|
| 2 | waiting | 不完整输入（句子中断、缺少宾语等）|
| 3 | ready | 正常输入（意图明确，可进入后续流程）|
| 4 | confused | 意图不明确（模糊、询问机器人自身等）|

## 性能指标

| 指标 | 验证集 | 测试集 |
|------|--------|--------|
| Accuracy | 97.10% | 97.70% |
| Macro-F1 | 96.24% | 97.12% |
| Weighted-F1 | 97.10% | 97.69% |

**推理速度：** ~10ms/条（GPU），~50ms/条（CPU）

## 注意事项

1. **模型只加载一次**，服务启动时加载，之后常驻内存
2. **部署到 Orin NX**：直接拷贝整个 `rejector/` 目录
3. **多轮对话**：`[SEP]` 是特殊分隔符，用于表示多轮对话历史
4. **GPU 显存**：模型约 400MB，推理时占用 ~1GB，Orin NX 16GB 完全够用
5. **服务端口**：默认 8089，确保防火墙开放

## 资源占用（Orin NX 16GB）

| 组件 | 占用内存 | 说明 |
|------|----------|------|
| CUDA 上下文 | ~1-2GB | GPU 驱动 |
| BERT 模型 | ~400MB | 权重 |
| 运行时显存 | ~1GB | 推理 |
| Python/FastAPI | ~500MB | 服务进程 |
| **总计** | ~3-4GB | 剩余 ~12GB 给主程序 |
