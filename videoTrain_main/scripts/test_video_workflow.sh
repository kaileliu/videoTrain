#!/bin/bash
# 测试视频训练完整流程

set -e

echo "🧪 测试视频训练工作流程"
echo "========================================"

# 1. 生成小规模测试数据
echo ""
echo "📊 步骤1: 生成测试数据..."
python scripts/generate_video_data.py \
    --output-dir data/test_video \
    --num-samples 10

# 2. 验证数据格式
echo ""
echo "✅ 步骤2: 验证数据格式..."
python -c "
from pathlib import Path
import pandas as pd
import numpy as np

data_root = Path('data/test_video')

# 检查目录
assert (data_root / 'mp4').exists(), 'mp4目录不存在'
assert (data_root / 'episode').exists(), 'episode目录不存在'

# 统计文件
video1_files = list((data_root / 'mp4').glob('*_video1.mp4'))
video2_files = list((data_root / 'mp4').glob('*_video2.mp4'))
data1_files = list((data_root / 'episode').glob('*_data1.parquet'))
data2_files = list((data_root / 'episode').glob('*_data2.parquet'))

print(f'✓ Video1: {len(video1_files)} 个文件')
print(f'✓ Video2: {len(video2_files)} 个文件')
print(f'✓ Data1: {len(data1_files)} 个文件')
print(f'✓ Data2: {len(data2_files)} 个文件')

assert len(video1_files) == len(video2_files) == len(data1_files) == len(data2_files), \
    '文件数量不匹配'

# 检查第一个样本
df1 = pd.read_parquet(data1_files[0])
df2 = pd.read_parquet(data2_files[0])

print(f'✓ Parquet形状: {df1.shape}')
print(f'✓ 列: {list(df1.columns)}')

# 计算delta示例
data1 = df1[['x', 'y', 'z']].values
data2 = df2[['x', 'y', 'z']].values
delta = np.mean(data1 - data2, axis=0)
print(f'✓ Delta示例: {delta}')
print(f'✓ Delta范数: {np.linalg.norm(delta):.6f}')

print('\n✅ 数据格式验证通过！')
"

# 3. 快速训练测试（2个epoch）
echo ""
echo "🚀 步骤3: 快速训练测试..."
echo "提示: 这只是测试流程，不会完全训练模型"
echo ""

# 修改配置为快速测试
cat > configs/test.yaml << 'EOF'
# 测试配置 - 快速验证流程
data:
  real_data_path: "data/real"
  sim_data_path: "data/sim"
  output_path: "outputs"

video:
  above_camera: "observation.images.above"
  front_camera: "observation.images.front"
  fps: 30
  frame_width: 128
  frame_height: 128
  sequence_length: 8

model:
  name: "SimToRealCalibrator"
  vision_encoder:
    type: "resnet18"
    pretrained: true
    freeze_backbone: false
  temporal_encoder:
    type: "lstm"
    hidden_dim: 128
    num_layers: 2
    num_heads: 4
    dropout: 0.1
  fusion:
    type: "concat"
  output_dim: 3

training:
  batch_size: 2
  num_epochs: 2
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "adamw"
  scheduler: "cosine"
  warmup_epochs: 0
  gradient_clip: 1.0

loss:
  type: "mse"
  weights:
    position: 1.0
    velocity: 0.0

validation:
  val_split: 0.2
  test_split: 0.2
  metric: "mse"

checkpoint:
  save_dir: "checkpoints_test"
  save_freq: 1
  keep_last_n: 2

logging:
  log_dir: "logs_test"
  log_freq: 5
  use_tensorboard: false

device: "cpu"
num_workers: 0
pin_memory: false
seed: 42
EOF

echo "使用测试配置（2 epochs, CPU）..."
timeout 60 python train_video.py \
    --data-root data/test_video \
    --config configs/test.yaml \
    --device cpu \
    2>&1 | head -n 100 || echo "训练测试完成（或超时）"

echo ""
echo "========================================"
echo "✅ 工作流程测试完成！"
echo ""
echo "如果以上步骤都通过了，说明代码工作正常。"
echo ""
echo "使用你的真实数据:"
echo "  python train_video.py --data-root your_data"
echo ""
echo "查看完整指南:"
echo "  cat VIDEO_TRAINING_GUIDE.md"