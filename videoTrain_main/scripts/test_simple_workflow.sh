#!/bin/bash
# 测试简化工作流程

set -e

echo "🧪 测试简化版训练流程"
echo "========================================"

# 1. 创建示例数据
echo ""
echo "📊 步骤1: 创建示例数据..."
python scripts/create_sample_data.py \
    --output-dir data/test_sample \
    --num-samples 50

# 2. 验证数据格式
echo ""
echo "✅ 步骤2: 验证数据格式..."
python -c "
import pandas as pd
import numpy as np

print('读取数据文件...')
df1 = pd.read_parquet('data/test_sample/video_stream_1.parquet')
df2 = pd.read_parquet('data/test_sample/video_stream_2.parquet')
dfl = pd.read_parquet('data/test_sample/labels.parquet')

print(f'✓ Video1: {len(df1)} 样本')
print(f'✓ Video2: {len(df2)} 样本')
print(f'✓ Labels: {len(dfl)} 样本')

assert len(df1) == len(df2) == len(dfl), '样本数量不匹配！'

print(f'✓ 第一个样本帧数: {len(df1.iloc[0][\"frames\"])}')
print(f'✓ Delta示例: {dfl.iloc[0][\"delta\"]}')

print('\\n✅ 数据格式验证通过！')
"

# 3. 快速训练测试（只训练2个epoch）
echo ""
echo "🚀 步骤3: 快速训练测试（2 epochs）..."
python train_simple.py \
    --video1 data/test_sample/video_stream_1.parquet \
    --video2 data/test_sample/video_stream_2.parquet \
    --labels data/test_sample/labels.parquet \
    --config configs/default.yaml \
    --device cpu \
    2>&1 | head -n 50

echo ""
echo "========================================"
echo "✅ 测试完成！"
echo ""
echo "如果上面的测试都通过了，说明代码工作正常。"
echo ""
echo "下一步使用你的真实数据:"
echo "  python train_simple.py \\"
echo "    --video1 your_video1.parquet \\"
echo "    --video2 your_video2.parquet \\"
echo "    --labels your_labels.parquet"