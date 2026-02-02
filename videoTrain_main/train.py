"""Training script for Sim-to-Real calibration"""

import argparse
import torch
import random
import numpy as np
from pathlib import Path

from src.data import create_dataloaders
from src.models import SimToRealCalibrator
from src.training import Trainer
from src.utils import load_config


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train Sim-to-Real Calibrator")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu/mps), overrides config')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Override device if specified
    if args.device:
        config['device'] = args.device
    
    # Set device
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config['device'] = 'cpu'
    elif config['device'] == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        config['device'] = 'cpu'
    
    device = config['device']
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(config.get('seed', 42))
    
    # Create output directories
    Path(config['data']['output_path']).mkdir(parents=True, exist_ok=True)
    Path(config['checkpoint']['save_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['logging']['log_dir']).mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )
    
    # Create model
    print("Creating model...")
    model = SimToRealCalibrator(config)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} trainable parameters")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()