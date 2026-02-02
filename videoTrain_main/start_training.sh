#!/bin/bash
# 一键启动训练 - RTX 5090

echo "=========================================="
echo "  RTX 5090 一键训练"
echo "=========================================="
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装"
    exit 1
fi

# 检查数据
if [ ! -d "data/mp4" ] || [ ! -d "data/parquet" ]; then
    echo "❌ 数据目录不存在"
    echo "   请确保data/mp4/和data/parquet/目录存在"
    exit 1
fi

echo "✓ 环境检查通过"
echo ""

# 选择训练版本
echo "选择训练版本:"
echo "  1. 简化版 (轻量模型，适合快速验证)"
echo "  2. 优化版 (ResNet18+Transformer，充分利用5090)"
echo ""
read -p "输入选项 [1/2，默认2]: " choice
choice=${choice:-2}

if [ "$choice" = "1" ]; then
    echo ""
    echo "🚀 启动简化版训练..."
    python3 train_direct.py
else
    echo ""
    echo "🚀 启动优化版训练（推荐）..."
    python3 train_5090_optimized.py
fi

echo ""
echo "=========================================="
echo "✅ 完成！"
echo "=========================================="