"""Script to prepare and validate data format"""

import argparse
from pathlib import Path
import pandas as pd
import json


def validate_data_structure(data_path: Path) -> bool:
    """
    Validate that data follows the expected LeRobot format
    
    Args:
        data_path: Path to data directory
        
    Returns:
        True if valid, False otherwise
    """
    print(f"Validating data structure at: {data_path}")
    
    required_dirs = ['data/chunk-000', 'meta/episodes/chunk-000', 'videos']
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = data_path / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    # Check for data files
    data_dir = data_path / 'data' / 'chunk-000'
    data_files = list(data_dir.glob('file-*.parquet'))
    if not data_files:
        print(f"❌ No data files found in {data_dir}")
        return False
    
    print(f"✓ Found {len(data_files)} data files")
    
    # Check for episode files
    episodes_dir = data_path / 'meta' / 'episodes' / 'chunk-000'
    episode_files = list(episodes_dir.glob('file-*.parquet'))
    if not episode_files:
        print(f"❌ No episode files found in {episodes_dir}")
        return False
    
    print(f"✓ Found {len(episode_files)} episode files")
    
    # Check for video directories
    video_dir = data_path / 'videos'
    camera_dirs = ['observation.images.above', 'observation.images.front']
    
    for camera in camera_dirs:
        camera_dir = video_dir / camera / 'chunk-000'
        if not camera_dir.exists():
            print(f"❌ Missing camera directory: {camera}")
            return False
        
        video_files = list(camera_dir.glob('file-*.mp4'))
        print(f"✓ Found {len(video_files)} video files for {camera}")
    
    # Check metadata
    meta_dir = data_path / 'meta'
    if (meta_dir / 'info.json').exists():
        print("✓ Found info.json")
    else:
        print("⚠️  No info.json found (optional)")
    
    print("\n✅ Data structure validation passed!")
    return True


def inspect_sample_data(data_path: Path, num_samples: int = 3):
    """
    Inspect sample data to understand the structure
    
    Args:
        data_path: Path to data directory
        num_samples: Number of samples to inspect
    """
    print(f"\nInspecting sample data from: {data_path}")
    
    # Load first data file
    data_dir = data_path / 'data' / 'chunk-000'
    data_files = sorted(data_dir.glob('file-*.parquet'))
    
    if not data_files:
        print("No data files found")
        return
    
    df = pd.read_parquet(data_files[0])
    
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst {num_samples} rows:")
    print(df.head(num_samples))
    
    # Check for pose-related columns
    pose_columns = [col for col in df.columns if any(
        key in col.lower() for key in ['state', 'position', 'pose', 'x', 'y', 'z']
    )]
    
    if pose_columns:
        print(f"\nPotential pose columns: {pose_columns}")
    else:
        print("\n⚠️  Warning: No obvious pose columns found. You may need to adapt the dataset class.")
    
    # Load episode metadata
    episodes_dir = data_path / 'meta' / 'episodes' / 'chunk-000'
    episode_files = sorted(episodes_dir.glob('file-*.parquet'))
    
    if episode_files:
        episodes_df = pd.read_parquet(episode_files[0])
        print(f"\nEpisode metadata shape: {episodes_df.shape}")
        print(f"Episode columns: {list(episodes_df.columns)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare and validate data")
    parser.add_argument('--real-data', type=str, required=True,
                       help='Path to real world data')
    parser.add_argument('--sim-data', type=str, required=True,
                       help='Path to simulation data')
    parser.add_argument('--inspect', action='store_true',
                       help='Inspect sample data structure')
    args = parser.parse_args()
    
    real_path = Path(args.real_data)
    sim_path = Path(args.sim_data)
    
    # Validate real data
    print("="*60)
    print("VALIDATING REAL WORLD DATA")
    print("="*60)
    real_valid = validate_data_structure(real_path)
    
    # Validate sim data
    print("\n" + "="*60)
    print("VALIDATING SIMULATION DATA")
    print("="*60)
    sim_valid = validate_data_structure(sim_path)
    
    # Inspect if requested
    if args.inspect:
        print("\n" + "="*60)
        print("INSPECTING REAL WORLD DATA")
        print("="*60)
        inspect_sample_data(real_path)
        
        print("\n" + "="*60)
        print("INSPECTING SIMULATION DATA")
        print("="*60)
        inspect_sample_data(sim_path)
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    if real_valid and sim_valid:
        print("✅ All data validation passed!")
        print("\nYou can now run training with:")
        print(f"  python train.py --config configs/default.yaml")
    else:
        print("❌ Data validation failed. Please fix the issues above.")


if __name__ == '__main__':
    main()