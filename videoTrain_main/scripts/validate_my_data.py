"""快速验证你的数据集是否符合要求"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from scripts.prepare_data import validate_data_structure, inspect_sample_data

if __name__ == '__main__':
    # 指定你的数据路径
    REAL_DATA_PATH = "data/real"
    SIM_DATA_PATH = "data/sim"
    
    print("🔍 开始验证数据集...")
    print("="*60)
    
    # 验证real数据
    print("\n📁 检查 REAL 数据...")
    real_valid = validate_data_structure(Path(REAL_DATA_PATH))
    
    # 验证sim数据
    print("\n📁 检查 SIM 数据...")
    sim_valid = validate_data_structure(Path(SIM_DATA_PATH))
    
    # 查看数据内容
    if real_valid:
        print("\n🔬 查看 REAL 数据样例...")
        inspect_sample_data(Path(REAL_DATA_PATH), num_samples=3)
    
    if sim_valid:
        print("\n🔬 查看 SIM 数据样例...")
        inspect_sample_data(Path(SIM_DATA_PATH), num_samples=3)
    
    # 总结
    print("\n" + "="*60)
    if real_valid and sim_valid:
        print("✅ 数据验证通过！可以开始训练")
        print("\n运行命令:")
        print("  python train.py --config configs/default.yaml")
    else:
        print("❌ 数据验证失败，请检查以上错误信息")