"""Visualization utilities"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Optional


def visualize_predictions(
    real_poses: np.ndarray,
    sim_poses: np.ndarray,
    predicted_deltas: np.ndarray,
    target_deltas: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Trajectory Comparison"
):
    """
    Visualize predicted vs target trajectories
    
    Args:
        real_poses: (T, 3) real trajectory
        sim_poses: (T, 3) sim trajectory
        predicted_deltas: (T, 3) predicted deltas
        target_deltas: (T, 3) target deltas
        save_path: Optional path to save figure
        title: Plot title
    """
    if torch.is_tensor(real_poses):
        real_poses = real_poses.cpu().numpy()
    if torch.is_tensor(sim_poses):
        sim_poses = sim_poses.cpu().numpy()
    if torch.is_tensor(predicted_deltas):
        predicted_deltas = predicted_deltas.cpu().numpy()
    if torch.is_tensor(target_deltas):
        target_deltas = target_deltas.cpu().numpy()
    
    # Compute corrected trajectories
    corrected_real = real_poses + predicted_deltas
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(real_poses[:, 0], real_poses[:, 1], real_poses[:, 2], 
             'r-', label='Real', linewidth=2)
    ax1.plot(sim_poses[:, 0], sim_poses[:, 1], sim_poses[:, 2], 
             'b-', label='Sim', linewidth=2)
    ax1.plot(corrected_real[:, 0], corrected_real[:, 1], corrected_real[:, 2], 
             'g--', label='Corrected Real', linewidth=2)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title('3D Trajectories')
    
    # Delta comparison
    ax2 = fig.add_subplot(2, 2, 2)
    timesteps = np.arange(len(predicted_deltas))
    for i, axis in enumerate(['X', 'Y', 'Z']):
        ax2.plot(timesteps, predicted_deltas[:, i], 
                label=f'Pred {axis}', linestyle='--')
        ax2.plot(timesteps, target_deltas[:, i], 
                label=f'Target {axis}', linestyle='-')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Delta')
    ax2.legend()
    ax2.set_title('Predicted vs Target Deltas')
    ax2.grid(True)
    
    # Error over time
    ax3 = fig.add_subplot(2, 2, 3)
    errors = np.linalg.norm(predicted_deltas - target_deltas, axis=1)
    ax3.plot(timesteps, errors, 'r-', linewidth=2)
    ax3.axhline(y=np.mean(errors), color='b', linestyle='--', 
                label=f'Mean Error: {np.mean(errors):.4f}')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Euclidean Error')
    ax3.legend()
    ax3.set_title('Prediction Error Over Time')
    ax3.grid(True)
    
    # Per-axis error
    ax4 = fig.add_subplot(2, 2, 4)
    axis_errors = np.abs(predicted_deltas - target_deltas)
    x = np.arange(3)
    means = [np.mean(axis_errors[:, i]) for i in range(3)]
    stds = [np.std(axis_errors[:, i]) for i in range(3)]
    ax4.bar(x, means, yerr=stds, alpha=0.7, capsize=5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(['X', 'Y', 'Z'])
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_title('Per-Axis Error Statistics')
    ax4.grid(True, axis='y')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    save_path: Optional[str] = None
):
    """
    Plot training and validation loss curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()