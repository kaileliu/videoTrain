"""Main Sim-to-Real calibration model"""

import torch
import torch.nn as nn
from typing import Dict
from .vision_encoder import DualCameraEncoder
from .temporal_encoder import TemporalEncoder


class SimToRealCalibrator(nn.Module):
    """
    Full model for predicting pose delta between sim and real trajectories
    
    Architecture:
    1. Dual camera encoders (separate for real and sim)
    2. Temporal encoders (process sequences)
    3. Fusion and prediction head
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Vision encoders
        vision_config = config['model']['vision_encoder']
        vision_output_dim = 512
        
        self.real_camera_encoder = DualCameraEncoder(
            backbone=vision_config['type'],
            pretrained=vision_config['pretrained'],
            freeze_backbone=vision_config['freeze_backbone'],
            output_dim=vision_output_dim,
            fusion_type=config['model']['fusion']['type']
        )
        
        self.sim_camera_encoder = DualCameraEncoder(
            backbone=vision_config['type'],
            pretrained=vision_config['pretrained'],
            freeze_backbone=vision_config['freeze_backbone'],
            output_dim=vision_output_dim,
            fusion_type=config['model']['fusion']['type']
        )
        
        # Temporal encoders
        temporal_config = config['model']['temporal_encoder']
        
        # Input dim: fused camera features + pose (3D)
        temporal_input_dim = self.real_camera_encoder.fused_dim + 3
        
        self.real_temporal_encoder = TemporalEncoder(
            encoder_type=temporal_config['type'],
            input_dim=temporal_input_dim,
            hidden_dim=temporal_config['hidden_dim'],
            num_layers=temporal_config['num_layers'],
            num_heads=temporal_config['num_heads'],
            dropout=temporal_config['dropout']
        )
        
        self.sim_temporal_encoder = TemporalEncoder(
            encoder_type=temporal_config['type'],
            input_dim=temporal_input_dim,
            hidden_dim=temporal_config['hidden_dim'],
            num_layers=temporal_config['num_layers'],
            num_heads=temporal_config['num_heads'],
            dropout=temporal_config['dropout']
        )
        
        # Fusion and prediction head
        fusion_dim = self.real_temporal_encoder.hidden_dim + self.sim_temporal_encoder.hidden_dim
        
        self.prediction_head = nn.Sequential(
            nn.Linear(fusion_dim, temporal_config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(temporal_config['dropout']),
            nn.Linear(temporal_config['hidden_dim'], temporal_config['hidden_dim'] // 2),
            nn.ReLU(),
            nn.Dropout(temporal_config['dropout']),
            nn.Linear(temporal_config['hidden_dim'] // 2, config['model']['output_dim'])
        )
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Dictionary containing:
                - video1: (B, T, 3, H, W) - 第一个视频流 (或包含above+front)
                - video2: (B, T, 3, H, W) - 第二个视频流 (或包含above+front)
                OR (如果使用双视角):
                - real_above: (B, T, 3, H, W)
                - real_front: (B, T, 3, H, W)
                - sim_above: (B, T, 3, H, W)
                - sim_front: (B, T, 3, H, W)
        
        Returns:
            Dictionary containing:
                - delta_pred: (B, 3) or (B, T, 3) predicted pose delta
        """
        # 检测输入格式
        if 'video1' in batch and 'video2' in batch:
            # 简化格式：单个视频流（可能是单视角或已融合的双视角）
            video1 = batch['video1']  # (B, T, 3, H, W)
            video2 = batch['video2']
            
            # 假设video1和video2各自可能包含双视角信息，或者是单视角
            # 这里我们直接用单编码器处理
            B, T, C, H, W = video1.shape
            
            # 使用real_camera_encoder处理video1
            # 需要将单个视频分成above和front，或者修改encoder
            # 简单起见，我们假设video1就是"合并后的视角"
            # 重塑为 (B, T, C, H, W) 保持不变，但我们需要拆分或使用单编码器
            
            # 这里简化处理：假设是单视角，复用到above和front
            video1_features = self.real_camera_encoder(video1, video1)  # (B, T, fused_dim)
            video2_features = self.sim_camera_encoder(video2, video2)  # (B, T, fused_dim)
            
        else:
            # 原始格式：双视角
            real_video_features = self.real_camera_encoder(
                batch['real_above'], 
                batch['real_front']
            )  # (B, T, fused_dim)
            
            sim_video_features = self.sim_camera_encoder(
                batch['sim_above'],
                batch['sim_front']
            )  # (B, T, fused_dim)
            
            video1_features = real_video_features
            video2_features = sim_video_features
        
        # 不再需要拼接位姿，直接用视频特征
        # Temporal encoding
        video1_temporal = self.real_temporal_encoder(video1_features)  # (B, T, hidden_dim)
        video2_temporal = self.sim_temporal_encoder(video2_features)  # (B, T, hidden_dim)
        
        # Fuse features
        fused = torch.cat([video1_temporal, video2_temporal], dim=-1)  # (B, T, 2*hidden_dim)
        
        # Predict delta
        delta_pred = self.prediction_head(fused)  # (B, T, 3)
        
        # 如果需要单个向量输出，可以在时间维度上平均或取最后一帧
        # delta_pred_single = delta_pred.mean(dim=1)  # (B, 3)
        # 或者
        delta_pred_single = delta_pred[:, -1, :]  # (B, 3) 取最后一帧
        
        return {
            'delta_pred': delta_pred_single,  # 单个向量 (B, 3)
            'delta_pred_sequence': delta_pred,  # 完整序列 (B, T, 3)
            'video1_features': video1_temporal,
            'video2_features': video2_temporal
        }
    
    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inference mode: return only predicted delta
        
        Args:
            batch: Input batch
            
        Returns:
            delta_pred: (B, T, 3)
        """
        with torch.no_grad():
            outputs = self.forward(batch)
            return outputs['delta_pred']