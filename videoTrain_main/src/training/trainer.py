"""Trainer class for model training and evaluation"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

from ..models.calibrator import SimToRealCalibrator
from .losses import CalibrationLoss


class Trainer:
    """Handles training, validation, and checkpointing"""
    
    def __init__(
        self,
        model: SimToRealCalibrator,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup loss
        loss_config = config['loss']
        self.criterion = CalibrationLoss(
            loss_type=loss_config['type'],
            position_weight=loss_config['weights']['position'],
            velocity_weight=loss_config['weights'].get('velocity', 0.0)
        )
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup logging
        log_dir = Path(config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir) if config['logging']['use_tensorboard'] else None
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def _create_optimizer(self):
        """Create optimizer based on config"""
        opt_config = self.config['training']
        
        if opt_config['optimizer'] == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['optimizer'] == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['optimizer'] == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                momentum=0.9,
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['optimizer']}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        sched_config = self.config['training']
        
        if sched_config['scheduler'] == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['num_epochs'],
                eta_min=sched_config['learning_rate'] * 0.01
            )
        elif sched_config['scheduler'] == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['num_epochs'] // 3,
                gamma=0.1
            )
        elif sched_config['scheduler'] == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_position_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch)
            
            # Compute loss (delta现在是单个向量)
            loss_dict = self.criterion(
                outputs['delta_pred'],  # (B, 3)
                batch['delta']  # (B, 3)
            )
            
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_position_loss += loss_dict['position_loss'].item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'pos_loss': loss_dict['position_loss'].item()
            })
            
            # Log to tensorboard
            if self.writer and self.global_step % self.config['logging']['log_freq'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/position_loss', 
                                     loss_dict['position_loss'].item(), 
                                     self.global_step)
                self.writer.add_scalar('train/lr', 
                                     self.optimizer.param_groups[0]['lr'], 
                                     self.global_step)
        
        return {
            'loss': total_loss / num_batches,
            'position_loss': total_position_loss / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        
        total_loss = 0.0
        total_position_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            
            # Compute loss
            loss_dict = self.criterion(
                outputs['delta_pred'],  # (B, 3)
                batch['delta']  # (B, 3)
            )
            
            # Compute metrics
            mae = torch.abs(outputs['delta_pred'] - batch['delta']).mean()
            rmse = torch.sqrt(torch.mean((outputs['delta_pred'] - batch['delta']) ** 2))
            
            total_loss += loss_dict['total_loss'].item()
            total_position_loss += loss_dict['position_loss'].item()
            total_mae += mae.item()
            total_rmse += rmse.item()
            num_batches += 1
        
        metrics = {
            'loss': total_loss / num_batches,
            'position_loss': total_position_loss / num_batches,
            'mae': total_mae / num_batches,
            'rmse': total_rmse / num_batches
        }
        
        # Log to tensorboard
        if self.writer:
            for key, value in metrics.items():
                self.writer.add_scalar(f'val/{key}', value, self.current_epoch)
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {self.best_val_loss:.6f}")
        
        # Keep only last N checkpoints
        keep_last_n = self.config['checkpoint'].get('keep_last_n', 5)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > keep_last_n:
            for ckpt in checkpoints[:-keep_last_n]:
                ckpt.unlink()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: Optional[int] = None):
        """Full training loop"""
        if num_epochs is None:
            num_epochs = self.config['training']['num_epochs']
        
        print(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            print(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.6f}")
            
            # Validate
            val_metrics = self.validate()
            print(f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.6f}, "
                  f"MAE: {val_metrics['mae']:.6f}, RMSE: {val_metrics['rmse']:.6f}")
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            if (epoch + 1) % self.config['checkpoint']['save_freq'] == 0:
                self.save_checkpoint(is_best)
        
        print("Training completed!")
        if self.writer:
            self.writer.close()
