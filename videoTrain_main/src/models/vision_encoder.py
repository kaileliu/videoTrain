"""Vision encoder for extracting features from video frames"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class VisionEncoder(nn.Module):
    """
    Encodes video frames into feature representations
    Supports multiple backbone architectures
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: int = 512
    ):
        """
        Args:
            backbone: Backbone architecture ('resnet18', 'resnet50', 'efficientnet_b0')
            pretrained: Use pretrained weights
            freeze_backbone: Freeze backbone parameters
            output_dim: Output feature dimension
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.output_dim = output_dim
        
        # Initialize backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_out_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer
            
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_out_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            backbone_out_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection head to desired output dimension
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, T, C, H, W) or (B, C, H, W)
            
        Returns:
            features: (B, T, output_dim) or (B, output_dim)
        """
        has_temporal = x.dim() == 5
        
        if has_temporal:
            B, T, C, H, W = x.shape
            # Reshape to process all frames in batch
            x = x.view(B * T, C, H, W)
        
        # Extract features
        features = self.backbone(x)  # (B*T, backbone_dim) or (B, backbone_dim)
        features = self.projection(features)  # (B*T, output_dim) or (B, output_dim)
        
        if has_temporal:
            # Reshape back to (B, T, output_dim)
            features = features.view(B, T, self.output_dim)
        
        return features


class DualCameraEncoder(nn.Module):
    """
    Encodes features from two camera views (above + front)
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        output_dim: int = 512,
        fusion_type: str = "concat"
    ):
        """
        Args:
            backbone: Backbone architecture
            pretrained: Use pretrained weights
            freeze_backbone: Freeze backbone parameters
            output_dim: Output feature dimension per camera
            fusion_type: How to fuse features ('concat', 'add', 'attention')
        """
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # Separate encoders for each camera (could also share weights)
        self.above_encoder = VisionEncoder(
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            output_dim=output_dim
        )
        
        self.front_encoder = VisionEncoder(
            backbone=backbone,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            output_dim=output_dim
        )
        
        # Fusion layer
        if fusion_type == "concat":
            self.fused_dim = output_dim * 2
        elif fusion_type == "add":
            self.fused_dim = output_dim
        elif fusion_type == "attention":
            self.fused_dim = output_dim
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=8,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
    
    def forward(
        self, 
        above_frames: torch.Tensor, 
        front_frames: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            above_frames: (B, T, C, H, W) above camera frames
            front_frames: (B, T, C, H, W) front camera frames
            
        Returns:
            fused_features: (B, T, fused_dim)
        """
        # Encode each camera view
        above_features = self.above_encoder(above_frames)  # (B, T, output_dim)
        front_features = self.front_encoder(front_frames)  # (B, T, output_dim)
        
        # Fuse features
        if self.fusion_type == "concat":
            fused = torch.cat([above_features, front_features], dim=-1)  # (B, T, 2*output_dim)
            
        elif self.fusion_type == "add":
            fused = above_features + front_features  # (B, T, output_dim)
            
        elif self.fusion_type == "attention":
            # Cross-attention between camera views
            fused, _ = self.attention(
                above_features, 
                front_features, 
                front_features
            )  # (B, T, output_dim)
        
        return fused