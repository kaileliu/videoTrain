"""RTX 5090优化版训练脚本 - 使用混合精度和更大模型"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
from typing import Tuple
import warnings
import torchvision.models as models
warnings.filterwarnings('ignore')

# RTX 5090配置
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP = torch.cuda.is_available()  # 混合精度训练
print(f"设备: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"混合精度: {'开启' if USE_AMP else '关闭'}")

# ==================== 数据集 ====================
class OptimizedVideoDataset(Dataset):
    def __init__(self, data_root="data", sequence_length=16, frame_size=(224, 224), mode='train'):
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.mode = mode
        
        # 查找所有文件
        self.mp4_dir = self.data_root / 'mp4'
        self.parquet_dir = self.data_root / 'parquet'
        
        # 获取样本ID（基于_video1后缀的文件）
        video1_files = sorted(self.mp4_dir.glob('*_video1.mp4'))
        self.sample_ids = [f.stem.replace('_video1', '') for f in video1_files]
        
        if not self.sample_ids:
            # 如果没有_video1后缀的文件，使用所有文件（假设每对文件有对应关系）
            all_mp4 = sorted(self.mp4_dir.glob('*.mp4'))
            self.sample_ids = [f.stem for f in all_mp4 if not f.stem.endswith('_video2')]
        
        assert len(self.sample_ids) > 0, "未找到视频文件"
        
        print(f"{mode}: {len(self.sample_ids)} 样本")
        print(f"示例样本ID: {self.sample_ids[:3]}")
    
    def _load_video(self, path):
        """加载视频帧"""
        cap = cv2.VideoCapture(str(path))
        frames = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        indices = np.linspace(0, max(0, total-1), self.sequence_length, dtype=int) if total > self.sequence_length else list(range(total))
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.frame_size)
            frames.append(frame)
        
        cap.release()
        
        # 填充不足的帧
        while len(frames) < self.sequence_length:
            frames.append(frames[-1].copy() if frames else np.zeros((*self.frame_size, 3), dtype=np.uint8))
        
        return np.stack(frames[:self.sequence_length])
    
    def _compute_ground_truth(self, df1, df2):
        """
        计算真实的ground_truth:
        action1 = df1最后一个 - df1第一个
        action2 = df2最后一个 - df2第一个  
        ground_truth = action1 - action2
        """
        # 提取数据
        data1 = None
        data2 = None
        
        # 尝试多种列名
        for cols in [['x','y','z'], ['action'], ['observation.state'], ['state']]:
            if all(c in df1.columns for c in cols):
                if len(cols) == 3:
                    data1 = df1[cols].values
                else:
                    data1 = np.array([np.array(row)[:3] for row in df1[cols[0]]])
                break
        else:
            data1 = df1.iloc[:, :3].values
        
        for cols in [['x','y','z'], ['action'], ['observation.state'], ['state']]:
            if all(c in df2.columns for c in cols):
                if len(cols) == 3:
                    data2 = df2[cols].values
                else:
                    data2 = np.array([np.array(row)[:3] for row in df2[cols[0]]])
                break
        else:
            data2 = df2.iloc[:, :3].values
        
        # 确保数据长度足够
        if len(data1) < 2 or len(data2) < 2:
            raise ValueError(f"数据长度不足: data1有{len(data1)}个时间步，data2有{len(data2)}个时间步")
        
        # 计算action1和action2
        action1 = data1[-1] - data1[0]  # 第一个parquet的首尾差
        action2 = data2[-1] - data2[0]  # 第二个parquet的首尾差
        
        # ground_truth = action1 - action2
        ground_truth = action1 - action2
        
        return ground_truth.astype(np.float32)
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # 构造文件路径（支持多种命名约定）
        possible_video1 = [
            self.mp4_dir / f"{sample_id}_video1.mp4",
            self.mp4_dir / f"{sample_id}_traj1.mp4",
            self.mp4_dir / f"{sample_id}_cam1.mp4",
            self.mp4_dir / f"{sample_id}.mp4"  # 单个文件情况
        ]
        
        possible_video2 = [
            self.mp4_dir / f"{sample_id}_video2.mp4",
            self.mp4_dir / f"{sample_id}_traj2.mp4", 
            self.mp4_dir / f"{sample_id}_cam2.mp4",
        ]
        
        possible_parquet1 = [
            self.parquet_dir / f"{sample_id}_data1.parquet",
            self.parquet_dir / f"{sample_id}_traj1.parquet",
            self.parquet_dir / f"{sample_id}_path1.parquet",
            self.parquet_dir / f"{sample_id}.parquet"  # 单个文件情况
        ]
        
        possible_parquet2 = [
            self.parquet_dir / f"{sample_id}_data2.parquet",
            self.parquet_dir / f"{sample_id}_traj2.parquet",
            self.parquet_dir / f"{sample_id}_path2.parquet",
        ]
        
        # 找到存在的文件
        video1_path = next((p for p in possible_video1 if p.exists()), None)
        video2_path = next((p for p in possible_video2 if p.exists()), None)
        parquet1_path = next((p for p in possible_parquet1 if p.exists()), None)
        parquet2_path = next((p for p in possible_parquet2 if p.exists()), None)
        
        if not video1_path or not parquet1_path:
            raise FileNotFoundError(f"找不到样本{sample_id}的视频1或parquet1文件")
        
        # 加载两个视频
        video1 = self._load_video(video1_path)
        video2 = self._load_video(video2_path) if video2_path else video1.copy()  # 如果没有video2，使用video1
        
        # 计算ground_truth
        df1 = pd.read_parquet(parquet1_path)
        df2 = pd.read_parquet(parquet2_path) if parquet2_path else df1.copy()  # 如果没有parquet2，使用parquet1
        
        ground_truth = self._compute_ground_truth(df1, df2)
        
        # 转换为tensor
        video1 = torch.from_numpy(video1).float().permute(0, 3, 1, 2) / 255.0
        video2 = torch.from_numpy(video2).float().permute(0, 3, 1, 2) / 255.0
        ground_truth = torch.from_numpy(ground_truth).float()
        
        return {
            'video1': video1,      # 第一个视频
            'video2': video2,      # 第二个视频  
            'delta': ground_truth, # ground_truth = action1 - action2
            'sample_id': sample_id
        }

# ==================== ResNet + Transformer模型 ====================
class ResNetTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ResNet18 backbone (预训练)
        resnet = models.resnet18(pretrained=True)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-2])  # 去掉最后的FC和池化
        
        # 特征投影
        self.feature_proj = nn.Linear(512, 256)
        
        # 双流Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer1 = nn.TransformerEncoder(encoder_layer.clone(), num_layers=2)
        self.transformer2 = nn.TransformerEncoder(encoder_layer.clone(), num_layers=2)
        
        # 融合模块
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),  # 合并两个256维特征
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 预测头
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)
        )
    
    def forward(self, video1, video2):
        """
        前向传播，处理两个视频流
        
        Args:
            video1: (B, T, 3, H, W) 第一个视频
            video2: (B, T, 3, H, W) 第二个视频
        
        Returns:
            predicted_delta: (B, 3) 预测的delta
        """
        B, T, C, H, W = video1.shape
        
        # 处理第一个视频
        video1_flat = video1.view(B * T, C, H, W)
        features1 = self.vision_encoder(video1_flat)  # (B*T, 512, 7, 7)
        features1 = torch.nn.functional.adaptive_avg_pool2d(features1, (1, 1))  # (B*T, 512, 1, 1)
        features1 = features1.view(B, T, 512)  # (B, T, 512)
        features1 = self.feature_proj(features1)  # (B, T, 256)
        features1 = self.transformer1(features1)  # (B, T, 256)
        features1 = features1[:, -1, :]  # (B, 256) 取最后一帧
        
        # 处理第二个视频
        video2_flat = video2.view(B * T, C, H, W)
        features2 = self.vision_encoder(video2_flat)  # (B*T, 512, 7, 7)
        features2 = torch.nn.functional.adaptive_avg_pool2d(features2, (1, 1))  # (B*T, 512, 1, 1)
        features2 = features2.view(B, T, 512)  # (B, T, 512)
        features2 = self.feature_proj(features2)  # (B, T, 256)
        features2 = self.transformer2(features2)  # (B, T, 256)
        features2 = features2[:, -1, :]  # (B, 256) 取最后一帧
        
        # 融合两个视频的特征
        combined_features = torch.cat([features1, features2], dim=1)  # (B, 512)
        fused_features = self.fusion(combined_features)  # (B, 256)
        
        # 预测delta
        predicted_delta = self.head(fused_features)  # (B, 3)
        
        return predicted_delta

# ==================== 训练函数 ====================
def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc='训练')
    for batch in pbar:
        video1 = batch['video1'].to(device)
        video2 = batch['video2'].to(device)
        delta = batch['delta'].to(device)
        
        optimizer.zero_grad()
        
        # 混合精度
        with autocast(enabled=USE_AMP):
            pred = model(video1, video2)
            loss = criterion(pred, delta)
        
        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='验证'):
            video1 = batch['video1'].to(device)
            video2 = batch['video2'].to(device)
            delta = batch['delta'].to(device)
            
            with autocast(enabled=USE_AMP):
                pred = model(video1, video2)
                loss = criterion(pred, delta)
            
            total_loss += loss.item()
            all_preds.append(pred.cpu())
            all_targets.append(delta.cpu())
    
    # 计算指标
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mae = torch.abs(all_preds - all_targets).mean().item()
    
    return total_loss / len(loader), mae

# ==================== 主函数 ====================
def main():
    # RTX 5090优化参数
    BATCH_SIZE = 32  # 5090可以用更大的batch
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 8  # 更多worker
    
    # 数据
    print("加载数据...")
    dataset = OptimizedVideoDataset('data')
    
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"训练: {train_size}, 验证: {val_size}, 测试: {test_size}")
    
    # 模型
    model = ResNetTransformerModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler(enabled=USE_AMP)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: {params:,}")
    
    # 训练
    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)
    
    print("\n开始训练...\n")
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_loss, val_mae = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | MAE: {val_mae:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
            }, 'checkpoints/best_model_5090.pt')
            print(f"✓ 保存最佳模型")
        
        print()
    
    print(f"\n训练完成！最佳Val Loss: {best_val_loss:.6f}")

if __name__ == '__main__':
    main()