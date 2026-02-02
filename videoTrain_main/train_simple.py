"""简化的训练脚本 - 只需要两个视频流parquet文件和一个标签文件"""

import argparse
import torch
import random
import numpy as np
from pathlib import Path

from src.data.simple_loader import create_simple_dataloaders
from src.models import SimToRealCalibrator
from src.training import Trainer
from src.utils import load_config


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="简化训练脚本 - 只需提供数据文件")
    parser.add_argument('--video1', type=str, required=True,
                       help='第一个视频流的parquet文件路径')
    parser.add_argument('--video2', type=str, required=True,
                       help='第二个视频流的parquet文件路径')
    parser.add_argument('--labels', type=str, required=True,
                       help='标签文件路径（包含delta向量）')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    parser.add_argument('--device', type=str, default=None,
                       help='设备 (cuda/cpu/mps)')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    print(f"✓ 加载配置: {args.config}")
    
    # 覆盖设备
    if args.device:
        config['device'] = args.device
    
    # 设置设备
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        config['device'] = 'cpu'
    elif config['device'] == 'mps' and not torch.backends.mps.is_available():
        print("⚠️  MPS不可用，使用CPU")
        config['device'] = 'cpu'
    
    device = config['device']
    print(f"✓ 使用设备: {device}")
    
    # 设置随机种子
    set_seed(config.get('seed', 42))
    
    # 创建输出目录
    Path(config['data']['output_path']).mkdir(parents=True, exist_ok=True)
    Path(config['checkpoint']['save_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['logging']['log_dir']).mkdir(parents=True, exist_ok=True)
    
    # 创建数据加载器
    print("\n📊 创建数据加载器...")
    print(f"  视频流1: {args.video1}")
    print(f"  视频流2: {args.video2}")
    print(f"  标签: {args.labels}")
    
    train_loader, val_loader, test_loader = create_simple_dataloaders(
        video1_path=args.video1,
        video2_path=args.video2,
        labels_path=args.labels,
        config=config,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )
    
    # 创建模型
    print("\n🧠 创建模型...")
    model = SimToRealCalibrator(config)
    
    # 统计参数
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ 模型参数量: {num_params:,}")
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # 从检查点恢复
    if args.resume:
        print(f"\n📥 从检查点恢复: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    print("\n" + "="*60)
    print("🚀 开始训练...")
    print("="*60)
    trainer.train()
    
    print("\n✅ 训练完成！")
    print(f"最佳模型保存在: {config['checkpoint']['save_dir']}/best_model.pt")


if __name__ == '__main__':
    main()