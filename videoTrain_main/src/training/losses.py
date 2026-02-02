"""Loss functions for calibration"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibrationLoss(nn.Module):
    """
    Loss function for sim-to-real calibration
    Supports multiple loss types and optional velocity penalty
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        position_weight: float = 1.0,
        velocity_weight: float = 0.0
    ):
        """
        Args:
            loss_type: Type of loss ('mse', 'l1', 'smooth_l1', 'huber')
            position_weight: Weight for position loss
            velocity_weight: Weight for velocity consistency loss
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        
        # Initialize base loss function
        if loss_type == "mse":
            self.base_loss = nn.MSELoss(reduction='none')
        elif loss_type == "l1":
            self.base_loss = nn.L1Loss(reduction='none')
        elif loss_type == "smooth_l1":
            self.base_loss = nn.SmoothL1Loss(reduction='none')
        elif loss_type == "huber":
            self.base_loss = nn.HuberLoss(reduction='none', delta=1.0)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def compute_velocity_loss(
        self, 
        delta_pred: torch.Tensor, 
        delta_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity consistency loss
        Penalizes sudden changes in predicted delta
        
        Args:
            delta_pred: (B, T, 3) predicted delta
            delta_target: (B, T, 3) target delta
            
        Returns:
            velocity_loss: scalar
        """
        # Compute velocities (differences between consecutive timesteps)
        pred_velocity = delta_pred[:, 1:] - delta_pred[:, :-1]  # (B, T-1, 3)
        target_velocity = delta_target[:, 1:] - delta_target[:, :-1]  # (B, T-1, 3)
        
        # L2 loss on velocities
        velocity_loss = F.mse_loss(pred_velocity, target_velocity)
        
        return velocity_loss
    
    def forward(
        self, 
        delta_pred: torch.Tensor, 
        delta_target: torch.Tensor,
        mask: torch.Tensor = None
    ) -> dict:
        """
        Compute total loss
        
        Args:
            delta_pred: (B, 3) or (B, T, 3) predicted delta
            delta_target: (B, 3) or (B, T, 3) target delta
            mask: Optional (B,) or (B, T) mask for valid timesteps
            
        Returns:
            Dictionary with loss components
        """
        # Position loss
        position_loss = self.base_loss(delta_pred, delta_target)  # (B, 3) or (B, T, 3)
        
        # Apply mask if provided
        if mask is not None:
            if delta_pred.dim() == 3:  # (B, T, 3)
                mask = mask.unsqueeze(-1)  # (B, T, 1)
            else:  # (B, 3)
                mask = mask.unsqueeze(-1)  # (B, 1)
            position_loss = position_loss * mask
            num_valid = mask.sum()
        else:
            num_valid = delta_pred.numel() / 3  # Total number of valid positions
        
        position_loss = position_loss.sum() / num_valid
        
        # Total loss
        total_loss = self.position_weight * position_loss
        
        loss_dict = {
            'total_loss': total_loss,
            'position_loss': position_loss,
        }
        
        # Add velocity loss if weight > 0
        if self.velocity_weight > 0:
            velocity_loss = self.compute_velocity_loss(delta_pred, delta_target)
            total_loss = total_loss + self.velocity_weight * velocity_loss
            loss_dict['velocity_loss'] = velocity_loss
            loss_dict['total_loss'] = total_loss
        
        return loss_dict


class AdaptiveCalibrationLoss(CalibrationLoss):
    """
    Adaptive loss that adjusts weights based on prediction difficulty
    """
    
    def __init__(
        self,
        loss_type: str = "mse",
        position_weight: float = 1.0,
        velocity_weight: float = 0.0,
        adaptive_weight: bool = True
    ):
        super().__init__(loss_type, position_weight, velocity_weight)
        self.adaptive_weight = adaptive_weight
    
    def forward(
        self, 
        delta_pred: torch.Tensor, 
        delta_target: torch.Tensor,
        mask: torch.Tensor = None
    ) -> dict:
        """
        Compute loss with adaptive weighting based on error magnitude
        """
        # Compute base losses
        loss_dict = super().forward(delta_pred, delta_target, mask)
        
        if self.adaptive_weight:
            # Compute per-sample error magnitude
            errors = torch.norm(delta_pred - delta_target, dim=-1)  # (B, T)
            
            # Weight harder samples more (inverse weighting)
            # Normalize to [0.5, 2.0] range to avoid extreme weights
            weights = 1.0 + torch.tanh(errors)  # (B, T)
            
            if mask is not None:
                weights = weights * mask
            
            # Recompute weighted loss
            weighted_loss = self.base_loss(delta_pred, delta_target)  # (B, T, 3)
            weighted_loss = (weighted_loss * weights.unsqueeze(-1)).sum() / weights.sum()
            
            loss_dict['total_loss'] = self.position_weight * weighted_loss
            loss_dict['weighted_position_loss'] = weighted_loss
        
        return loss_dict