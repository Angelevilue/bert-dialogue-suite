#!/bin/bash
# =============================================================================
# Bert Rejector Service 启动脚本
# 支持 PyTorch 和 ONNX 两种推理引擎
# =============================================================================
# 用法:
#   ./start_service.sh                    # PyTorch 前台运行
#   ./start_service.sh --onnx             # ONNX 前台运行
#   ./start_service.sh --onnx --background # ONNX 后台运行
#   ./start_service.sh --port 8001        # 指定端口
#   ./start_service.sh --onnx --port 8001 # ONNX + 指定端口
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 自动激活 conda 环境（如果未激活）
if [[ "$CONDA_DEFAULT_ENV" != "rejector" ]]; then
    if command -v conda &> /dev/null && conda env list | grep -q "rejector"; then
        eval "$(conda shell.bash hook)"
        conda activate rejector
    fi
fi

# 默认值
HOST="0.0.0.0"
PORT=8089
DEVICE="auto"
BACKGROUND=0
ENGINE="pytorch"  # pytorch 或 onnx
MODEL_FILE=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --background|-b)
            BACKGROUND=1
            shift
            ;;
        --onnx)
            ENGINE="onnx"
            shift
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --model)
            MODEL_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --background, -b    后台运行"
            echo "  --onnx             使用 ONNX 推理引擎（默认使用 PyTorch）"
            echo "  --host <IP>        监听地址 (默认: 0.0.0.0)"
            echo "  --port <PORT>      监听端口 (默认: 8089)"
            echo "  --device <DEVICE>  设备: cuda/cpu/mps/auto (仅 PyTorch 模式)"
            echo "  --model <FILE>     指定模型文件 (仅 ONNX 模式)"
            echo ""
            echo "示例:"
            echo "  ./start_service.sh                      # PyTorch 模式"
            echo "  ./start_service.sh --onnx               # ONNX 模式"
            echo "  ./start_service.sh --onnx --background   # ONNX 后台运行"
            echo "  ./start_service.sh --onnx --model checkpoint/onnx/bert_rejector_fp16.onnx"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 构建命令
if [[ "$ENGINE" == "onnx" ]]; then
    SCRIPT="service_onnx.py"
    CMD="python3 $SCRIPT --host $HOST --port $PORT"
    if [[ -n "$MODEL_FILE" ]]; then
        CMD="$CMD --model $MODEL_FILE"
    fi
    ENGINE_NAME="ONNX (FP32)"
    LOG_FILE="service_onnx.log"
else
    SCRIPT="service.py"
    CMD="python3 $SCRIPT --host $HOST --port $PORT --device $DEVICE"
    ENGINE_NAME="PyTorch"
    LOG_FILE="service.log"
fi

echo "=========================================="
echo "Bert Rejector Service"
echo "=========================================="
echo "推理引擎: $ENGINE_NAME"
echo "服务脚本: $SCRIPT"
echo "服务地址: http://$HOST:$PORT"
if [[ "$ENGINE" == "onnx" ]]; then
    if [[ -n "$MODEL_FILE" ]]; then
        echo "模型文件: $MODEL_FILE"
    else
        echo "模型文件: checkpoint/onnx/bert_rejector_fp16.onnx (默认 FP16)"
    fi
else
    echo "模型路径: checkpoint/"
    echo "设备: $DEVICE"
fi
echo "日志文件: $SCRIPT_DIR/$LOG_FILE"
echo "=========================================="

if [[ $BACKGROUND -eq 1 ]]; then
    nohup $CMD > $LOG_FILE 2>&1 &
    echo "服务已后台启动, PID: $!"
    echo "查看日志: tail -f $SCRIPT_DIR/$LOG_FILE"
    echo "停止服务: kill $!"
else
    echo "启动服务中..."
    $CMD
fi
