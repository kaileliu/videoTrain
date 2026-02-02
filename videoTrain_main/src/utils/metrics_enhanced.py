"""增强的评估指标 - 针对首尾差值预测任务"""

import torch
import numpy as np
from typing import Dict, Tuple


def compute_delta_metrics(
    predictions: torch.Tensor, 
    targets: torch.Tensor,
    return_detailed: bool = True
) -> Dict[str, float]:
    """
    计算delta预测的评估指标
    
    针对 delta = (data1[-1] - data1[0]) - (data2[-1] - data2[0]) 的任务
    
    Args:
        predictions: (N, 3) 预测的delta
        targets: (N, 3) 真实的delta
        return_detailed: 是否返回详细指标
        
    Returns:
        指标字典
    """
    # 转换为numpy
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(targets):
        targets = targets.cpu().numpy()
    
    metrics = {}
    
    # ============ 1. 基础误差指标 ============
    errors = predictions - targets  # (N, 3)
    abs_errors = np.abs(errors)
    squared_errors = errors ** 2
    
    # 1.1 分轴指标
    for i, axis in enumerate(['x', 'y', 'z']):
        metrics[f'mae_{axis}'] = np.mean(abs_errors[:, i])
        metrics[f'rmse_{axis}'] = np.sqrt(np.mean(squared_errors[:, i]))
        metrics[f'std_{axis}'] = np.std(errors[:, i])
        
        if return_detailed:
            metrics[f'max_error_{axis}'] = np.max(abs_errors[:, i])
            metrics[f'min_error_{axis}'] = np.min(abs_errors[:, i])
    
    # 1.2 整体指标
    metrics['mae_overall'] = np.mean(abs_errors)
    metrics['rmse_overall'] = np.sqrt(np.mean(squared_errors))
    metrics['mse_overall'] = np.mean(squared_errors)
    
    # ============ 2. 欧氏距离指标 ============
    euclidean_errors = np.linalg.norm(errors, axis=1)  # (N,)
    
    metrics['mean_euclidean_error'] = np.mean(euclidean_errors)
    metrics['median_euclidean_error'] = np.median(euclidean_errors)
    metrics['max_euclidean_error'] = np.max(euclidean_errors)
    metrics['std_euclidean_error'] = np.std(euclidean_errors)
    
    # ============ 3. 百分位数指标 ============
    for p in [50, 75, 90, 95, 99]:
        metrics[f'p{p}_euclidean_error'] = np.percentile(euclidean_errors, p)
    
    # ============ 4. 方向误差指标（新增）============
    # 衡量预测方向与真实方向的一致性
    
    # 4.1 余弦相似度
    pred_norm = np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8
    target_norm = np.linalg.norm(targets, axis=1, keepdims=True) + 1e-8
    
    pred_normalized = predictions / pred_norm
    target_normalized = targets / target_norm
    
    cosine_similarities = np.sum(pred_normalized * target_normalized, axis=1)
    metrics['mean_cosine_similarity'] = np.mean(cosine_similarities)
    metrics['median_cosine_similarity'] = np.median(cosine_similarities)
    
    # 4.2 角度误差（度）
    # angle = arccos(cosine_similarity)
    cosine_similarities_clipped = np.clip(cosine_similarities, -1, 1)
    angle_errors_rad = np.arccos(cosine_similarities_clipped)
    angle_errors_deg = np.degrees(angle_errors_rad)
    
    metrics['mean_angle_error_deg'] = np.mean(angle_errors_deg)
    metrics['median_angle_error_deg'] = np.median(angle_errors_deg)
    
    if return_detailed:
        metrics['max_angle_error_deg'] = np.max(angle_errors_deg)
        metrics['std_angle_error_deg'] = np.std(angle_errors_deg)
    
    # ============ 5. 幅值误差指标（新增）============
    # 衡量预测幅值与真实幅值的差异
    
    pred_magnitudes = np.linalg.norm(predictions, axis=1)
    target_magnitudes = np.linalg.norm(targets, axis=1)
    magnitude_errors = pred_magnitudes - target_magnitudes
    
    metrics['mean_magnitude_error'] = np.mean(np.abs(magnitude_errors))
    metrics['rmse_magnitude'] = np.sqrt(np.mean(magnitude_errors ** 2))
    metrics['mean_magnitude_ratio'] = np.mean(pred_magnitudes / (target_magnitudes + 1e-8))
    
    if return_detailed:
        metrics['median_magnitude_error'] = np.median(np.abs(magnitude_errors))
        metrics['std_magnitude_error'] = np.std(magnitude_errors)
    
    # ============ 6. 相对误差指标（新增）============
    # 相对于真实值的误差百分比
    
    relative_errors = abs_errors / (np.abs(targets) + 1e-8)  # (N, 3)
    metrics['mean_relative_error'] = np.mean(relative_errors)
    metrics['median_relative_error'] = np.median(relative_errors)
    
    # ============ 7. 成功率指标（新增）============
    # 定义不同阈值下的成功率
    
    thresholds = [0.01, 0.02, 0.05, 0.1]  # 单位：米（根据你的数据调整）
    for thresh in thresholds:
        success_rate = np.mean(euclidean_errors < thresh) * 100
        metrics[f'success_rate_{int(thresh*1000)}mm'] = success_rate
    
    # ============ 8. R² 决定系数（新增）============
    # 衡量模型的拟合优度
    
    ss_res = np.sum(squared_errors)  # 残差平方和
    ss_tot = np.sum((targets - np.mean(targets, axis=0)) ** 2)  # 总平方和
    r2_score = 1 - (ss_res / (ss_tot + 1e-8))
    metrics['r2_score'] = r2_score
    
    return metrics


def compute_confusion_stats(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.05
) -> Dict[str, float]:
    """
    计算混淆矩阵相关统计（将任务视为分类：好预测 vs 差预测）
    
    Args:
        predictions: (N, 3) 预测值
        targets: (N, 3) 真实值
        threshold: 判定为"好预测"的欧氏距离阈值
        
    Returns:
        统计指标
    """
    errors = np.linalg.norm(predictions - targets, axis=1)
    
    # 将样本分为"简单"和"困难"
    target_magnitudes = np.linalg.norm(targets, axis=1)
    is_difficult = target_magnitudes > np.median(target_magnitudes)
    
    # 计算简单和困难样本的指标
    easy_errors = errors[~is_difficult]
    hard_errors = errors[is_difficult]
    
    stats = {
        'easy_samples_count': int(np.sum(~is_difficult)),
        'hard_samples_count': int(np.sum(is_difficult)),
        'easy_mean_error': np.mean(easy_errors) if len(easy_errors) > 0 else 0,
        'hard_mean_error': np.mean(hard_errors) if len(hard_errors) > 0 else 0,
        'easy_success_rate': np.mean(easy_errors < threshold) * 100 if len(easy_errors) > 0 else 0,
        'hard_success_rate': np.mean(hard_errors < threshold) * 100 if len(hard_errors) > 0 else 0,
    }
    
    return stats


def print_metrics_summary(metrics: Dict[str, float], top_n: int = 10):
    """
    打印指标摘要
    
    Args:
        metrics: 指标字典
        top_n: 显示最重要的前N个指标
    """
    print("\n" + "="*60)
    print("核心评估指标".center(60))
    print("="*60)
    
    # 定义最重要的指标及其优先级
    priority_metrics = [
        'mae_overall',
        'rmse_overall',
        'mean_euclidean_error',
        'mean_angle_error_deg',
        'mean_magnitude_error',
        'r2_score',
        'success_rate_50mm',
        'mean_cosine_similarity',
    ]
    
    # 打印优先指标
    for key in priority_metrics:
        if key in metrics:
            print(f"  {key:30s}: {metrics[key]:10.6f}")
    
    print("="*60)
    
    # 分轴指标
    print("\n分轴指标:")
    for axis in ['x', 'y', 'z']:
        mae_key = f'mae_{axis}'
        rmse_key = f'rmse_{axis}'
        if mae_key in metrics and rmse_key in metrics:
            print(f"  {axis.upper()}: MAE={metrics[mae_key]:.6f}, RMSE={metrics[rmse_key]:.6f}")
    
    # 百分位数
    print("\n误差分布（百分位数）:")
    for p in [50, 75, 90, 95, 99]:
        key = f'p{p}_euclidean_error'
        if key in metrics:
            print(f"  P{p}: {metrics[key]:.6f}")
    
    print("="*60 + "\n")