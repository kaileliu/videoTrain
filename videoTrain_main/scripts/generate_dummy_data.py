"""生成模拟数据用于测试"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm


def generate_dummy_episode(
    episode_id: int,
    output_dir: Path,
    num_frames: int = 100,
    fps: int = 30,
    frame_size: tuple = (480, 640, 3)
):
    """生成一个模拟episode的数据"""
    
    # 创建目录
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "observation.images.above" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos" / "observation.images.front" / "chunk-000").mkdir(parents=True, exist_ok=True)
    
    # 1. 生成轨迹数据（正弦波轨迹）
    t = np.linspace(0, 2*np.pi, num_frames)
    trajectory = np.stack([
        0.5 + 0.2 * np.sin(t),      # x: [0.3, 0.7]
        0.3 + 0.1 * np.cos(t),      # y: [0.2, 0.4]
        0.2 + 0.05 * np.sin(2*t),   # z: [0.15, 0.25]
    ], axis=1)  # (T, 3)
    
    # 保存轨迹数据
    data_df = pd.DataFrame({
        'episode_index': [episode_id] * num_frames,
        'frame_index': list(range(num_frames)),
        'timestamp': [i / fps for i in range(num_frames)],
        'observation.state': [trajectory[i].tolist() for i in range(num_frames)],
        'x': trajectory[:, 0].tolist(),
        'y': trajectory[:, 1].tolist(),
        'z': trajectory[:, 2].tolist(),
    })
    data_df.to_parquet(output_dir / "data" / "chunk-000" / f"file-{episode_id:03d}.parquet")
    
    # 2. 保存episode元数据
    episode_df = pd.DataFrame({
        'episode_index': [episode_id],
        'length': [num_frames],
        'task': ['dummy_task'],
    })
    episode_df.to_parquet(output_dir / "meta" / "episodes" / "chunk-000" / f"file-{episode_id:03d}.parquet")
    
    # 3. 生成模拟视频（两个视角）
    for camera in ['above', 'front']:
        video_path = output_dir / "videos" / f"observation.images.{camera}" / "chunk-000" / f"file-{episode_id:03d}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (frame_size[1], frame_size[0]))
        
        for i in range(num_frames):
            # 生成随机彩色图像（模拟相机视图）
            if camera == 'above':
                # 顶部视角：蓝色调
                frame = np.random.randint(100, 200, frame_size, dtype=np.uint8)
                frame[:, :, 2] = 200  # 蓝色通道
            else:
                # 腕部视角：绿色调
                frame = np.random.randint(100, 200, frame_size, dtype=np.uint8)
                frame[:, :, 1] = 200  # 绿色通道
            
            # 添加一些移动的标记（模拟机械臂）
            center_x = int(frame_size[1] * (0.5 + 0.2 * np.sin(t[i])))
            center_y = int(frame_size[0] * (0.5 + 0.2 * np.cos(t[i])))
            cv2.circle(frame, (center_x, center_y), 20, (255, 0, 0), -1)
            
            # 添加文字信息
            cv2.putText(frame, f"Frame {i}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            writer.write(frame)
        
        writer.release()


def generate_dataset(
    output_dir: Path,
    num_episodes: int = 10,
    num_frames_per_episode: int = 100,
    add_noise: bool = False
):
    """
    生成完整的模拟数据集
    
    Args:
        output_dir: 输出目录（如 data/real 或 data/sim）
        num_episodes: episode数量
        num_frames_per_episode: 每个episode的帧数
        add_noise: 是否添加噪声（用于区分real和sim）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_episodes} episodes to {output_dir}...")
    
    for ep_id in tqdm(range(num_episodes)):
        # 添加随机变化
        num_frames = num_frames_per_episode + np.random.randint(-10, 10)
        generate_dummy_episode(ep_id, output_dir, num_frames=num_frames)
    
    # 创建info.json
    import json
    info = {
        'dataset_name': output_dir.name,
        'num_episodes': num_episodes,
        'fps': 30,
        'encoding': {
            'observation.images.above': {'codec': 'mp4v'},
            'observation.images.front': {'codec': 'mp4v'},
        }
    }
    with open(output_dir / "meta" / "info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"✅ Generated {num_episodes} episodes")


if __name__ == '__main__':
    print("🎬 生成模拟数据集...")
    
    # 生成real数据（无噪声）
    print("\n生成 REAL 数据...")
    generate_dataset(
        output_dir=Path("data/real"),
        num_episodes=20,
        num_frames_per_episode=100,
        add_noise=False
    )
    
    # 生成sim数据（轻微不同的轨迹）
    print("\n生成 SIM 数据...")
    generate_dataset(
        output_dir=Path("data/sim"),
        num_episodes=20,
        num_frames_per_episode=100,
        add_noise=True
    )
    
    print("\n✅ 数据生成完成！")
    print("\n下一步：")
    print("  1. 验证数据: python scripts/validate_my_data.py")
    print("  2. 开始训练: python train.py --config configs/default.yaml")