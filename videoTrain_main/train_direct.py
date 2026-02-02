"""直接训练脚本 - 适配当前data/目录结构，针对RTX 5090优化"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# 设置设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ==================== 数据集类 ====================
class DirectVideoDataset(Dataset):
    """直接适配data/目录的数据集"""
    
    def __init__(
        self,
        data_root: str = "data",
        sequence_length: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        mode: str = 'train'
    ):
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.mode = mode
        
        # 查找所有文件
        self.mp4_files = sorted((self.data_root / 'mp4').glob('*.mp4'))
        self.parquet_files = sorted((self.data_root / 'parquet').glob('*.parquet'))
        
        # 验证文件数量
        assert len(self.mp4_files) == len(self.parquet_files), \
            f"视频和parquet文件数量不匹配: {len(self.mp4_files)} vs {len(self.parquet_files)}"
        
        print(f"{mode} 数据集: {len(self.mp4_files)} 个样本")
    
    def _load_video(self, video_path: Path) -> np.ndarray:
        """加载视频帧"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 均匀采样
        if total_frames > self.sequence_length:
            indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
        else:
            indices = list(range(total_frames))
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, self.frame_size)
            frames.append(frame_resized)
        
        cap.release()
        
        # 填充
        while len(frames) < self.sequence_length:
            frames.append(frames[-1] if frames else np.zeros((*self.frame_size, 3), dtype=np.uint8))
        
        return np.stack(frames[:self.sequence_length], axis=0)
    
    def _compute_delta(self, df: pd.DataFrame) -> np.ndarray:
        """
        从parquet计算delta
        假设parquet包含两个轨迹的数据
        """
        # 尝试提取列
        possible_cols = [
            ['x', 'y', 'z'],
            ['action'],
            ['observation.state'],
            ['state'],
        ]
        
        data = None
        for cols in possible_cols:
            if all(c in df.columns for c in cols):
                if len(cols) == 3:
                    data = df[cols].values
                else:
                    # 提取前3维
                    data = np.array([np.array(row)[:3] for row in df[cols[0]]])
                break
        
        if data is None:
            # 如果找不到标准列，假设前3列是坐标
            data = df.iloc[:, :3].values
        
        # 计算首尾差
        if len(data) >= 2:
            delta = data[-1] - data[0]
        else:
            delta = np.zeros(3)
        
        return delta.astype(np.float32)
    
    def __len__(self):
        return len(self.mp4_files)
    
    def __getitem__(self, idx):
        # 加载视频
        video = self._load_video(self.mp4_files[idx])
        
        # 加载parquet并计算delta
        df = pd.read_parquet(self.parquet_files[idx])
        delta = self._compute_delta(df)
        
        # 转tensor
        video = torch.from_numpy(video).float().permute(0, 3, 1, 2) / 255.0
        delta = torch.from_numpy(delta).float()
        
        return {'video': video, 'delta': delta}

# ==================== 模型定义 ====================
class SimpleVideoModel(nn.Module):
    """简化的视频回归模型"""
    
    def __init__(self):
        super().__init__()
        
        # 视觉编码器 - 使用轻量级CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 时序编码器 - LSTM
        self.lstm = nn.LSTM(
            input_size=128 * 16,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # 预测头
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3)
        )
    
    def forward(self, x):
        # x: (B, T, 3, H, W)
        B, T, C, H, W = x.shape
        
        # CNN特征提取
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)  # (B*T, 128, 4, 4)
        features = features.view(B, T, -1)  # (B, T, 128*16)
        
        # LSTM
        lstm_out, _ = self.lstm(features)  # (B, T, 256)
        
        # 取最后一帧
        final_features = lstm_out[:, -1, :]  # (B, 256)
        
        # 预测
        delta = self.head(final_features)  # (B, 3)
        
        return delta

# ==================== 训练函数 ====================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc='训练中')
    for batch in pbar:
        video = batch['video'].to(device)
        delta = batch['delta'].to(device)
        
        optimizer.zero_grad()
        pred = model(video)
        loss = criterion(pred, delta)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='验证中'):
            video = batch['video'].to(device)
            delta = batch['delta'].to(device)
            
            pred = model(video)
            loss = criterion(pred, delta)
            total_loss += loss.item()
    
    return total_loss / len(loader)

# ==================== 主训练流程 ====================
def main():
    # 超参数
    BATCH_SIZE = 16  # RTX 5090可以用更大的batch size
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    
    # 创建数据集
    print("加载数据...")
    full_dataset = DirectVideoDataset(data_root='data')
    
    # 划分数据集
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"训练集: {train_size}, 验证集: {val_size}, 测试集: {test_size}")
    
    # 创建模型
    model = SimpleVideoModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # 统计参数
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {num_params:,}")
    
    # 训练循环
    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)
    
    print("\n开始训练...")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # 验证
        val_loss = validate(model, val_loader, criterion, DEVICE)
        
        # 更新学习率
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'checkpoints/best_model.pt')
            print(f"✓ 保存最佳模型 (val_loss: {val_loss:.6f})")
    
    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.6f}")

if __name__ == '__main__':
    main()