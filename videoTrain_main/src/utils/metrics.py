"""Evaluation metrics"""

import torch
import numpy as np
from typing import Dict


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute evaluation metrics
    
    Args:
        predictions: (N, 3) predicted deltas
        targets: (N, 3) target deltas
        
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    # Compute errors
    errors = predictions - targets  # (N, 3)
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    # Per-axis metrics
    metrics = {}
    for i, axis in enumerate(['x', 'y', 'z']):
        metrics[f'mae_{axis}'] = np.mean(abs_errors[:, i])
        metrics[f'rmse_{axis}'] = np.sqrt(np.mean(squared_errors[:, i]))
        metrics[f'std_{axis}'] = np.std(errors[:, i])
    
    # Overall metrics
    metrics['mae_overall'] = np.mean(abs_errors)
    metrics['rmse_overall'] = np.sqrt(np.mean(squared_errors))
    metrics['mse_overall'] = np.mean(squared_errors)
    
    # Euclidean distance metrics
    euclidean_errors = np.linalg.norm(errors, axis=1)
    metrics['mean_euclidean_error'] = np.mean(euclidean_errors)
    metrics['median_euclidean_error'] = np.median(euclidean_errors)
    metrics['max_euclidean_error'] = np.max(euclidean_errors)
    metrics['std_euclidean_error'] = np.std(euclidean_errors)
    
    # Percentiles
    for p in [50, 75, 90, 95, 99]:
        metrics[f'p{p}_euclidean_error'] = np.percentile(euclidean_errors, p)
    
    return metrics


def compute_trajectory_metrics(
    pred_trajectory: torch.Tensor, 
    target_trajectory: torch.Tensor
) -> Dict[str, float]:
    """
    Compute metrics for full trajectories
    
    Args:
        pred_trajectory: (T, 3) predicted trajectory
        target_trajectory: (T, 3) target trajectory
        
    Returns:
        Dictionary of trajectory metrics
    """
    if torch.is_tensor(pred_trajectory):
        pred_trajectory = pred_trajectory.cpu().numpy()
    if torch.is_tensor(target_trajectory):
        target_trajectory = target_trajectory.cpu().numpy()
    
    metrics = {}
    
    # Endpoint error
    endpoint_error = np.linalg.norm(pred_trajectory[-1] - target_trajectory[-1])
    metrics['endpoint_error'] = endpoint_error
    
    # Average trajectory error
    trajectory_errors = np.linalg.norm(pred_trajectory - target_trajectory, axis=1)
    metrics['avg_trajectory_error'] = np.mean(trajectory_errors)
    metrics['max_trajectory_error'] = np.max(trajectory_errors)
    
    # Path length difference
    pred_path_length = np.sum(np.linalg.norm(np.diff(pred_trajectory, axis=0), axis=1))
    target_path_length = np.sum(np.linalg.norm(np.diff(target_trajectory, axis=0), axis=1))
    metrics['path_length_diff'] = abs(pred_path_length - target_path_length)
    metrics['path_length_ratio'] = pred_path_length / (target_path_length + 1e-8)
    
    return metrics