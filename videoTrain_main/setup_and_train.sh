#!/bin/bash
# 一键安装依赖并开始训练（RTX 5090优化）

set -e

echo "=========================================="
echo "  RTX 5090 训练环境设置"
echo "=========================================="

# 1. 安装Python依赖
echo ""
echo "📦 步骤 1/3: 安装依赖..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121 -q || pip3 install torch torchvision -q
pip3 install opencv-python pandas pyarrow tqdm numpy -q

# 2. 验证CUDA
echo ""
echo "🔧 步骤 2/3: 验证GPU环境..."
python3 << 'EOF'
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  警告: CUDA不可用，将使用CPU训练")
EOF

# 3. 检查数据
echo ""
echo "📊 步骤 3/3: 检查数据..."
python3 << 'EOF'
from pathlib import Path

data_root = Path('data')
mp4_files = list((data_root / 'mp4').glob('*.mp4'))
parquet_files = list((data_root / 'parquet').glob('*.parquet'))

print(f"视频文件: {len(mp4_files)}")
print(f"Parquet文件: {len(parquet_files)}")

if len(mp4_files) == 0 or len(parquet_files) == 0:
    print("❌ 错误: data/目录下没有找到数据文件")
    print("   请确保data/mp4/和data/parquet/目录下有文件")
    exit(1)

if len(mp4_files) != len(parquet_files):
    print("⚠️  警告: 视频和parquet文件数量不匹配")
else:
    print("✓ 数据检查通过")
EOF

# 4. 开始训练
echo ""
echo "=========================================="
echo "🚀 开始训练..."
echo "=========================================="
echo ""

python3 train_direct.py

echo ""
echo "=========================================="
echo "✅ 训练完成！"
echo "=========================================="
echo ""
echo "最佳模型保存在: checkpoints/best_model.pt"
