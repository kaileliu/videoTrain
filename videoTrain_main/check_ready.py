"""检查训练环境是否就绪"""

import sys
from pathlib import Path

print("="*60)
print("  RTX 5090 训练环境检查")
print("="*60)

all_good = True

# 1. 检查Python
print("\n1. Python环境:")
print(f"   ✓ Python版本: {sys.version.split()[0]}")

# 2. 检查PyTorch
try:
    import torch
    print(f"\n2. PyTorch:")
    print(f"   ✓ 版本: {torch.__version__}")
    print(f"   ✓ CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✓ CUDA版本: {torch.version.cuda}")
        print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"   ⚠️  将使用CPU训练（较慢）")
except ImportError:
    print(f"\n2. PyTorch:")
    print(f"   ❌ PyTorch未安装")
    print(f"      安装: pip3 install torch torchvision")
    all_good = False

# 3. 检查其他依赖
print(f"\n3. 其他依赖:")
deps = ['cv2', 'pandas', 'numpy', 'tqdm']
missing = []
for dep in deps:
    try:
        __import__(dep)
        print(f"   ✓ {dep}")
    except ImportError:
        print(f"   ❌ {dep} (缺失)")
        missing.append(dep)
        all_good = False

if missing:
    print(f"\n   安装缺失的包:")
    print(f"   pip3 install {' '.join(missing)}")

# 4. 检查数据
print(f"\n4. 数据检查:")
data_root = Path('data')

if not data_root.exists():
    print(f"   ❌ data/目录不存在")
    all_good = False
else:
    mp4_dir = data_root / 'mp4'
    parquet_dir = data_root / 'parquet'
    
    if not mp4_dir.exists():
        print(f"   ❌ data/mp4/目录不存在")
        all_good = False
    else:
        mp4_files = list(mp4_dir.glob('*.mp4'))
        print(f"   ✓ 视频文件: {len(mp4_files)} 个")
        if len(mp4_files) == 0:
            print(f"      ⚠️  警告: 没有MP4文件")
            all_good = False
    
    if not parquet_dir.exists():
        print(f"   ❌ data/parquet/目录不存在")
        all_good = False
    else:
        parquet_files = list(parquet_dir.glob('*.parquet'))
        print(f"   ✓ Parquet文件: {len(parquet_files)} 个")
        if len(parquet_files) == 0:
            print(f"      ⚠️  警告: 没有Parquet文件")
            all_good = False
        
        if mp4_dir.exists() and parquet_dir.exists():
            if len(mp4_files) != len(parquet_files):
                print(f"   ⚠️  警告: 视频和Parquet文件数量不匹配")
                print(f"      ({len(mp4_files)} vs {len(parquet_files)})")

# 5. 检查训练脚本
print(f"\n5. 训练脚本:")
scripts = ['train_direct.py', 'train_5090_optimized.py']
for script in scripts:
    if Path(script).exists():
        print(f"   ✓ {script}")
    else:
        print(f"   ❌ {script} (缺失)")
        all_good = False

# 总结
print("\n" + "="*60)
if all_good:
    print("✅ 环境检查通过！可以开始训练")
    print("\n启动训练:")
    print("  ./start_training.sh")
    print("  或")
    print("  python3 train_5090_optimized.py")
else:
    print("❌ 发现问题，请先解决上述错误")
print("="*60)