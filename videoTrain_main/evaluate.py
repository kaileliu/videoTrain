"""Evaluation script for Sim-to-Real calibration"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.data import create_dataloaders
from src.models import SimToRealCalibrator
from src.utils import load_config, compute_metrics, visualize_predictions


def evaluate_model(model, test_loader, device, config):
    """
    Evaluate model on test set
    
    Args:
        model: Trained model
        test_loader: Test dataloader
        device: Device to use
        config: Configuration dict
        
    Returns:
        Dictionary of metrics and predictions
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_real_poses = []
    all_sim_poses = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Move to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Collect predictions
            all_predictions.append(outputs['delta_pred'].cpu())
            all_targets.append(batch['delta'].cpu())
            all_real_poses.append(batch['real_poses'].cpu())
            all_sim_poses.append(batch['sim_poses'].cpu())
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)  # (N, T, 3)
    targets = torch.cat(all_targets, dim=0)
    real_poses = torch.cat(all_real_poses, dim=0)
    sim_poses = torch.cat(all_sim_poses, dim=0)
    
    # Flatten for metrics
    predictions_flat = predictions.view(-1, 3)
    targets_flat = targets.view(-1, 3)
    
    # Compute metrics
    metrics = compute_metrics(predictions_flat, targets_flat)
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    for key, value in sorted(metrics.items()):
        print(f"{key:30s}: {value:.6f}")
    print("="*50)
    
    return {
        'metrics': metrics,
        'predictions': predictions,
        'targets': targets,
        'real_poses': real_poses,
        'sim_poses': sim_poses
    }


def visualize_samples(results, output_dir, num_samples=5):
    """
    Visualize sample predictions
    
    Args:
        results: Evaluation results dictionary
        output_dir: Output directory for visualizations
        num_samples: Number of samples to visualize
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = results['predictions']
    targets = results['targets']
    real_poses = results['real_poses']
    sim_poses = results['sim_poses']
    
    num_samples = min(num_samples, len(predictions))
    
    print(f"\nGenerating visualizations for {num_samples} samples...")
    for i in range(num_samples):
        visualize_predictions(
            real_poses=real_poses[i].numpy(),
            sim_poses=sim_poses[i].numpy(),
            predicted_deltas=predictions[i].numpy(),
            target_deltas=targets[i].numpy(),
            save_path=output_dir / f"sample_{i}.png",
            title=f"Sample {i} - Trajectory Comparison"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sim-to-Real Calibrator")
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--num-vis-samples', type=int, default=5,
                       help='Number of samples to visualize')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")
    
    # Set device
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    _, _, test_loader = create_dataloaders(
        config,
        num_workers=config.get('num_workers', 4),
        pin_memory=config.get('pin_memory', True)
    )
    
    # Create model
    print("Creating model...")
    model = SimToRealCalibrator(config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, config)
    
    # Save metrics
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = output_dir / 'metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("EVALUATION METRICS\n")
        f.write("="*50 + "\n")
        for key, value in sorted(results['metrics'].items()):
            f.write(f"{key:30s}: {value:.6f}\n")
        f.write("="*50 + "\n")
    
    print(f"\nMetrics saved to {metrics_file}")
    
    # Generate visualizations
    if args.visualize:
        vis_dir = output_dir / 'visualizations'
        visualize_samples(results, vis_dir, num_samples=args.num_vis_samples)
        print(f"Visualizations saved to {vis_dir}")
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()