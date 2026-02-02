#!/bin/bash
# 视频训练快速开始脚本

set -e

echo "🎬 视频训练快速开始"
echo "=========================================="

# 1. 检查Python
echo ""
echo "📦 步骤 1/4: 检查环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装"
    exit 1
fi
echo "✓ Python版本: $(python3 --version)"

# 2. 安装依赖
echo ""
echo "📦 步骤 2/4: 检查依赖..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "⚠️  PyTorch未安装"
    read -p "是否安装依赖包？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pip install -r requirements.txt
    fi
else
    echo "✓ 依赖已安装"
fi

# 3. 生成测试数据
echo ""
echo "📊 步骤 3/4: 准备数据..."
if [ ! -d "data/video_sample" ]; then
    read -p "生成测试数据？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 scripts/generate_video_data.py \
            --output-dir data/video_sample \
            --num-samples 20
    else
        echo "请确保数据目录存在"
        exit 1
    fi
else
    echo "✓ 数据目录已存在"
fi

# 4. 训练
echo ""
echo "🚀 步骤 4/4: 训练模型..."
read -p "开始训练？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 train_video.py \
        --data-root data/video_sample \
        --device cpu
else
    echo ""
    echo "✅ 准备完成！"
    echo ""
    echo "手动启动训练："
    echo "  python3 train_video.py --data-root data/video_sample"
fi

echo ""
echo "=========================================="
echo "✅ 完成！"