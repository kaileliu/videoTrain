#!/bin/bash
# 使用现有LeRobot格式数据

# 1. 创建data目录
mkdir -p data

# 2. 创建软链接到你的现有数据
# 假设你的real数据在 /path/to/czn_vla_dataset
ln -s /path/to/czn_vla_dataset data/real

# 3. 为sim数据创建链接（如果有）
# ln -s /path/to/sim_dataset data/sim

# 4. 验证数据
python scripts/validate_my_data.py

echo "✅ 数据准备完成！"
echo "如果数据有效，运行: python train.py --config configs/default.yaml"