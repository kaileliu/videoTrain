"""批量转换你的数据到标准格式"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm


def convert_data(
    video_dir: str,
    trajectory_dir: str,
    output_dir: str,
    video1_pattern: str = "*_cam1.mp4",
    video2_pattern: str = "*_cam2.mp4",
    traj1_pattern: str = "*_traj1.csv",
    traj2_pattern: str = "*_traj2.csv",
):
    """
    批量转换数据
    
    Args:
        video_dir: 视频文件目录
        trajectory_dir: 轨迹数据目录
        output_dir: 输出目录
        video1_pattern: 视频1的文件名模式
        video2_pattern: 视频2的文件名模式
        traj1_pattern: 轨迹1的文件名模式
        traj2_pattern: 轨迹2的文件名模式
    """
    video_dir = Path(video_dir)
    trajectory_dir = Path(trajectory_dir)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    mp4_dir = output_dir / "mp4"
    episode_dir = output_dir / "episode"
    mp4_dir.mkdir(parents=True, exist_ok=True)
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有视频文件
    video1_files = sorted(video_dir.glob(video1_pattern))
    
    if not video1_files:
        raise ValueError(f"未找到匹配 {video1_pattern} 的视频文件")
    
    print(f"找到 {len(video1_files)} 个样本")
    
    for i, video1_file in enumerate(tqdm(video1_files, desc="转换数据")):
        sample_id = f"sample_{i:03d}"
        
        # 获取对应的文件
        # 假设文件名有相同的前缀
        base_name = video1_file.stem.replace('_cam1', '')
        
        video2_file = video_dir / f"{base_name}_cam2.mp4"
        traj1_file = trajectory_dir / f"{base_name}_traj1.csv"
        traj2_file = trajectory_dir / f"{base_name}_traj2.csv"
        
        # 检查文件是否存在
        if not video2_file.exists():
            print(f"⚠️  跳过 {sample_id}: 找不到video2")
            continue
        if not traj1_file.exists():
            print(f"⚠️  跳过 {sample_id}: 找不到traj1")
            continue
        if not traj2_file.exists():
            print(f"⚠️  跳过 {sample_id}: 找不到traj2")
            continue
        
        # 复制视频
        shutil.copy2(video1_file, mp4_dir / f"{sample_id}_video1.mp4")
        shutil.copy2(video2_file, mp4_dir / f"{sample_id}_video2.mp4")
        
        # 转换轨迹数据
        try:
            df1 = pd.read_csv(traj1_file)
            df2 = pd.read_csv(traj2_file)
            
            # 检查是否有x,y,z列
            if all(col in df1.columns for col in ['x', 'y', 'z']):
                df1[['x', 'y', 'z']].to_parquet(episode_dir / f"{sample_id}_data1.parquet")
                df2[['x', 'y', 'z']].to_parquet(episode_dir / f"{sample_id}_data2.parquet")
            elif 'action' in df1.columns:
                # 如果是action列，直接保存
                df1.to_parquet(episode_dir / f"{sample_id}_data1.parquet")
                df2.to_parquet(episode_dir / f"{sample_id}_data2.parquet")
            else:
                print(f"⚠️  跳过 {sample_id}: CSV缺少x,y,z或action列")
                print(f"    可用列: {list(df1.columns)}")
                continue
                
        except Exception as e:
            print(f"⚠️  跳过 {sample_id}: 转换失败 - {e}")
            continue
    
    print(f"\n✅ 转换完成！")
    print(f"输出目录: {output_dir}")
    print(f"\n下一步:")
    print(f"  python train_video.py --data-root {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="转换数据到标准格式")
    parser.add_argument('--video-dir', type=str, required=True,
                       help='视频文件目录')
    parser.add_argument('--trajectory-dir', type=str, required=True,
                       help='轨迹数据目录')
    parser.add_argument('--output-dir', type=str, default='data/converted',
                       help='输出目录')
    parser.add_argument('--video1-pattern', type=str, default='*_cam1.mp4',
                       help='视频1文件名模式')
    parser.add_argument('--video2-pattern', type=str, default='*_cam2.mp4',
                       help='视频2文件名模式')
    
    args = parser.parse_args()
    
    convert_data(
        video_dir=args.video_dir,
        trajectory_dir=args.trajectory_dir,
        output_dir=args.output_dir,
        video1_pattern=args.video1_pattern,
        video2_pattern=args.video2_pattern,
    )