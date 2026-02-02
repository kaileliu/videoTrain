#!/bin/bash
# 快速开始脚本 - 用于测试完整流程

set -e  # 遇到错误立即退出

echo "🚀 Sim-to-Real 校准项目 - 快速开始"
echo "=========================================="

# 1. 检查Python环境
echo ""
echo "📦 步骤 1/5: 检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装，请先安装Python3"
    exit 1
fi
echo "✓ Python版本: $(python3 --version)"

# 2. 安装依赖
echo ""
echo "📦 步骤 2/5: 安装依赖..."
read -p "是否安装依赖包？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install -r requirements.txt
    echo "✓ 依赖安装完成"
else
    echo "⚠️  跳过依赖安装"
fi

# 3. 生成测试数据
echo ""
echo "📊 步骤 3/5: 生成测试数据..."
if [ ! -d "data/real" ] || [ ! -d "data/sim" ]; then
    read -p "是否生成模拟数据用于测试？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 scripts/generate_dummy_data.py
        echo "✓ 测试数据生成完成"
    else
        echo "⚠️  请确保 data/real 和 data/sim 目录存在"
        exit 1
    fi
else
    echo "✓ 数据目录已存在"
fi

# 4. 验证数据
echo ""
echo "🔍 步骤 4/5: 验证数据..."
python3 scripts/validate_my_data.py

# 5. 询问是否开始训练
echo ""
echo "🎯 步骤 5/5: 准备训练..."
read -p "数据验证通过！是否开始训练？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 开始训练..."
    echo "提示: 使用 Ctrl+C 可以随时停止训练"
    echo ""
    sleep 2
    python3 train.py --config configs/default.yaml
else
    echo ""
    echo "✅ 准备工作完成！"
    echo ""
    echo "手动启动训练："
    echo "  python3 train.py --config configs/default.yaml"
    echo ""
    echo "评估模型："
    echo "  python3 evaluate.py --checkpoint checkpoints/best_model.pt --visualize"
fi