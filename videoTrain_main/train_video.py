"""超简化训练脚本 - 直接从MP4和parquet训练"""

import argparse
import torch
import random
import numpy as np
from pathlib import Path

from src.data.video_loader import create_video_dataloaders
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
    parser = argparse.ArgumentParser(description="从MP4视频和parquet训练")
    parser.add_argument('--data-root', type=str, required=True,
                       help='数据根目录（包含mp4/和episode/子目录）')
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
    
    # 验证数据目录
    data_root = Path(args.data_root)
    if not (data_root / 'mp4').exists():
        raise ValueError(f"未找到mp4目录: {data_root / 'mp4'}")
    if not (data_root / 'episode').exists():
        raise ValueError(f"未找到episode目录: {data_root / 'episode'}")
    
    print(f"\n📊 数据目录: {data_root}")
    print(f"  - MP4目录: {data_root / 'mp4'}")
    print(f"  - Episode目录: {data_root / 'episode'}")
    
    # 创建数据加载器
    print("\n📊 创建数据加载器...")
    train_loader, val_loader, test_loader = create_video_dataloaders(
        data_root=str(data_root),
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