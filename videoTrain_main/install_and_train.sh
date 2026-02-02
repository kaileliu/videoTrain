#!/bin/bash
# 完整安装依赖并开始训练

set -e

echo "=========================================="
echo "  自动安装依赖并开始训练"
echo "=========================================="

# 检查pip3
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3未找到，尝试使用pip..."
    if ! command -v pip &> /dev/null; then
        echo "❌ 未找到pip，请先安装pip"
        exit 1
    fi
    PIP_CMD="pip"
else
    PIP_CMD="pip3"
fi

echo ""
echo "📦 安装依赖（可能需要几分钟）..."
echo ""

# 安装PyTorch（CUDA 12.1版本，适合RTX 5090）
echo "安装PyTorch（CUDA 12.1）..."
$PIP_CMD install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet || {
    echo "⚠️  CUDA版本安装失败，尝试安装CPU版本..."
    $PIP_CMD install torch torchvision --quiet
}

# 安装其他依赖
echo "安装其他依赖..."
$PIP_CMD install opencv-python pandas pyarrow tqdm numpy --quiet

echo ""
echo "✅ 依赖安装完成"
echo ""

# 再次检查环境
echo "=========================================="
echo "  环境验证"
echo "=========================================="
python3 check_ready.py

echo ""
echo "=========================================="
read -p "按Enter开始训练，或Ctrl+C取消..." 

echo ""
echo "🚀 开始训练..."
echo ""

# 选择训练版本
if [ -f "train_5090_optimized.py" ]; then
    echo "使用优化版训练（ResNet18 + Transformer）"
    python3 train_5090_optimized.py
else
    echo "使用简化版训练"
    python3 train_direct.py
fi

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
echo ""
echo "模型保存在: checkpoints/"
ls -lh checkpoints/*.pt 2>/dev/null || echo "未找到保存的模型"