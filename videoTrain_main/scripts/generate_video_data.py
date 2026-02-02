"""生成示例MP4和parquet数据用于测试"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm


def generate_sample_video(output_path: str, num_frames: int = 60, fps: int = 30):
    """
    生成一个示例MP4视频
    
    Args:
        output_path: 输出视频路径
        num_frames: 帧数
        fps: 帧率
    """
    frame_size = (480, 640)  # (H, W)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_size[1], frame_size[0]))
    
    for i in range(num_frames):
        # 创建彩色背景
        frame = np.random.randint(50, 150, (*frame_size, 3), dtype=np.uint8)
        
        # 添加移动的圆圈（模拟机械臂末端）
        t = i / num_frames
        center_x = int(frame_size[1] * (0.3 + 0.4 * np.sin(2 * np.pi * t)))
        center_y = int(frame_size[0] * (0.3 + 0.4 * np.cos(2 * np.pi * t)))
        
        cv2.circle(frame, (center_x, center_y), 30, (255, 0, 0), -1)
        
        # 添加帧编号
        cv2.putText(frame, f"Frame {i}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        writer.write(frame)
    
    writer.release()


def generate_sample_parquet(output_path: str, num_steps: int = 60):
    """
    生成示例parquet文件（包含位姿数据）
    
    Args:
        output_path: 输出parquet路径
        num_steps: 时间步数
    """
    # 生成正弦波轨迹
    t = np.linspace(0, 2*np.pi, num_steps)
    
    data = {
        'timestamp': [i * 0.033 for i in range(num_steps)],  # 30 FPS
        'x': 0.5 + 0.2 * np.sin(t),
        'y': 0.3 + 0.1 * np.cos(t),
        'z': 0.2 + 0.05 * np.sin(2*t),
    }
    
    df = pd.DataFrame(data)
    df.to_parquet(output_path)


def generate_dataset(output_dir: str = "data/video_sample", num_samples: int = 20):
    """
    生成完整的示例数据集
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
    """
    output_dir = Path(output_dir)
    mp4_dir = output_dir / 'mp4'
    episode_dir = output_dir / 'episode'
    
    mp4_dir.mkdir(parents=True, exist_ok=True)
    episode_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 生成 {num_samples} 个样本到: {output_dir}")
    print(f"  - MP4目录: {mp4_dir}")
    print(f"  - Episode目录: {episode_dir}")
    
    for i in tqdm(range(num_samples), desc="生成数据"):
        sample_id = f"sample_{i:03d}"
        
        # 生成视频1
        video1_path = mp4_dir / f"{sample_id}_video1.mp4"
        generate_sample_video(str(video1_path), num_frames=60, fps=30)
        
        # 生成视频2（稍有不同）
        video2_path = mp4_dir / f"{sample_id}_video2.mp4"
        generate_sample_video(str(video2_path), num_frames=60, fps=30)
        
        # 生成parquet数据1（轨迹1）
        data1_path = episode_dir / f"{sample_id}_data1.parquet"
        generate_sample_parquet(str(data1_path), num_steps=60)
        
        # 生成parquet数据2（轨迹2，添加噪声）
        # 读取data1
        df1 = pd.read_parquet(data1_path)
        
        # 添加噪声创建data2
        df2 = df1.copy()
        df2['x'] += np.random.uniform(-0.05, 0.05, len(df2))
        df2['y'] += np.random.uniform(-0.05, 0.05, len(df2))
        df2['z'] += np.random.uniform(-0.05, 0.05, len(df2))
        
        data2_path = episode_dir / f"{sample_id}_data2.parquet"
        df2.to_parquet(data2_path)
    
    print(f"\n✅ 数据生成完成！")
    print(f"\n📁 目录结构:")
    print(f"{output_dir}/")
    print(f"├── mp4/")
    print(f"│   ├── sample_000_video1.mp4")
    print(f"│   ├── sample_000_video2.mp4")
    print(f"│   ├── sample_001_video1.mp4")
    print(f"│   └── ...")
    print(f"└── episode/")
    print(f"    ├── sample_000_data1.parquet")
    print(f"    ├── sample_000_data2.parquet")
    print(f"    ├── sample_001_data1.parquet")
    print(f"    └── ...")
    
    # 显示示例delta
    print(f"\n?? 示例数据和Delta计算:")
    df1 = pd.read_parquet(episode_dir / "sample_000_data1.parquet")
    df2 = pd.read_parquet(episode_dir / "sample_000_data2.parquet")
    
    data1 = df1[['x', 'y', 'z']].values
    data2 = df2[['x', 'y', 'z']].values
    
    print(f"  data1形状: {data1.shape}")
    print(f"  data2形状: {data2.shape}")
    print(f"  data1首: {data1[0]}")
    print(f"  data1尾: {data1[-1]}")
    print(f"  data2首: {data2[0]}")
    print(f"  data2尾: {data2[-1]}")
    
    # 新的计算方式
    delta_1 = data1[-1] - data1[0]
    delta_2 = data2[-1] - data2[0]
    delta = delta_1 - delta_2
    
    print(f"\n  Delta计算:")
    print(f"    delta_1 (data1尾-首): {delta_1}")
    print(f"    delta_2 (data2尾-首): {delta_2}")
    print(f"    最终Delta (delta_1 - delta_2): {delta}")
    print(f"    Delta范数: {np.linalg.norm(delta):.6f}")
    
    print(f"\n下一步:")
    print(f"  python train_video.py --data-root {output_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="生成示例视频数据")
    parser.add_argument('--output-dir', type=str, default='data/video_sample',
                       help='输出目录')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='样本数量')
    args = parser.parse_args()
    
    generate_dataset(args.output_dir, args.num_samples)