"""将自定义格式数据转换为LeRobot格式"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil

def convert_episode_to_lerobot(
    episode_id: int,
    video_above_path: str,
    video_front_path: str,
    trajectory_data: np.ndarray,  # (T, 3) - (x, y, z) positions
    output_dir: Path,
    fps: int = 30
):
    """
    将单个episode转换为LeRobot格式
    
    Args:
        episode_id: Episode编号
        video_above_path: 顶部相机视频路径
        video_front_path: 腕部相机视频路径
        trajectory_data: 轨迹数据 (T, 3) - x, y, z坐标
        output_dir: 输出目录
        fps: 视频帧率
    """
    output_dir = Path(output_dir)
    
    # 创建目录结构
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "observation.images.above" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "observation.images.front" / "chunk-000").mkdir(parents=True, exist_ok=True)
    
    num_frames = len(trajectory_data)
    
    # 1. 保存轨迹数据
    data_df = pd.DataFrame({
        'episode_index': [episode_id] * num_frames,
        'frame_index': list(range(num_frames)),
        'timestamp': [i / fps for i in range(num_frames)],
        # ⭐ 核心：保存位姿数据
        'observation.state': [trajectory_data[i].tolist() for i in range(num_frames)],
        # 或者分开保存
        'x': trajectory_data[:, 0].tolist(),
        'y': trajectory_data[:, 1].tolist(),
        'z': trajectory_data[:, 2].tolist(),
    })
    
    data_path = output_dir / "data" / "chunk-000" / f"file-{episode_id:03d}.parquet"
    data_df.to_parquet(data_path)
    
    # 2. 保存episode元数据
    episode_df = pd.DataFrame({
        'episode_index': [episode_id],
        'length': [num_frames],
        'task': ['manipulation'],  # 可自定义
    })
    
    episode_path = output_dir / "meta" / "episodes" / "chunk-000" / f"file-{episode_id:03d}.parquet"
    episode_df.to_parquet(episode_path)
    
    # 3. 复制视频文件
    video_above_out = output_dir / "videos" / "observation.images.above" / "chunk-000" / f"file-{episode_id:03d}.mp4"
    video_front_out = output_dir / "videos" / "observation.images.front" / "chunk-000" / f"file-{episode_id:03d}.mp4"
    
    shutil.copy2(video_above_path, video_above_out)
    shutil.copy2(video_front_path, video_front_out)
    
    print(f"✓ Converted episode {episode_id}: {num_frames} frames")


def example_batch_convert():
    """
    批量转换示例
    假设你有以下数据：
    - /path/to/real_videos/episode_0_above.mp4
    - /path/to/real_videos/episode_0_front.mp4
    - /path/to/real_trajectories/episode_0.npy  # shape: (T, 3)
    """
    
    # 配置路径
    REAL_VIDEO_DIR = Path("your_data/real_videos")
    REAL_TRAJ_DIR = Path("your_data/real_trajectories")
    SIM_VIDEO_DIR = Path("your_data/sim_videos")
    SIM_TRAJ_DIR = Path("your_data/sim_trajectories")
    
    OUTPUT_REAL = Path("data/real")
    OUTPUT_SIM = Path("data/sim")
    
    # 获取所有episode
    num_episodes = 100  # 你的episode数量
    
    print("Converting REAL data...")
    for ep_id in tqdm(range(num_episodes)):
        # 加载视频路径
        video_above = REAL_VIDEO_DIR / f"episode_{ep_id}_above.mp4"
        video_front = REAL_VIDEO_DIR / f"episode_{ep_id}_front.mp4"
        
        # 加载轨迹数据
        trajectory = np.load(REAL_TRAJ_DIR / f"episode_{ep_id}.npy")  # (T, 3)
        
        # 转换
        convert_episode_to_lerobot(
            episode_id=ep_id,
            video_above_path=str(video_above),
            video_front_path=str(video_front),
            trajectory_data=trajectory,
            output_dir=OUTPUT_REAL,
            fps=30
        )
    
    print("\nConverting SIM data...")
    for ep_id in tqdm(range(num_episodes)):
        video_above = SIM_VIDEO_DIR / f"episode_{ep_id}_above.mp4"
        video_front = SIM_VIDEO_DIR / f"episode_{ep_id}_front.mp4"
        trajectory = np.load(SIM_TRAJ_DIR / f"episode_{ep_id}.npy")
        
        convert_episode_to_lerobot(
            episode_id=ep_id,
            video_above_path=str(video_above),
            video_front_path=str(video_front),
            trajectory_data=trajectory,
            output_dir=OUTPUT_SIM,
            fps=30
        )
    
    print("\n✅ Conversion completed!")
    print("Run validation: python scripts/validate_my_data.py")


if __name__ == '__main__':
    # 📝 根据你的数据修改这里
    example_batch_convert()
