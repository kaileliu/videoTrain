"""创建示例数据文件 - 演示正确的数据格式"""

import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import io


def create_sample_parquet_data(output_dir: str = "data/sample", num_samples: int = 100):
    """
    创建示例parquet数据文件
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📝 创建示例数据到: {output_dir}")
    
    # 参数设置
    sequence_length = 16  # 每个视频序列的帧数
    frame_size = (224, 224)
    
    # 1. 创建video_stream_1.parquet
    print("\n1️⃣ 创建 video_stream_1.parquet...")
    video1_data = []
    
    for i in range(num_samples):
        # 为每个样本创建一个视频序列（16帧）
        frames = []
        for t in range(sequence_length):
            # 创建一个彩色图像（蓝色渐变）
            img = np.zeros((*frame_size, 3), dtype=np.uint8)
            img[:, :, 2] = int(100 + 100 * (t / sequence_length))  # 蓝色通道渐变
            
            # 添加一些运动（移动的圆圈）
            center_x = int(frame_size[1] * (0.3 + 0.4 * np.sin(2 * np.pi * t / sequence_length)))
            center_y = int(frame_size[0] * 0.5)
            
            y, x = np.ogrid[:frame_size[0], :frame_size[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= 20**2
            img[mask] = [255, 0, 0]  # 红色圆圈
            
            # 将图像转换为bytes（可选，也可以直接存numpy数组）
            pil_img = Image.fromarray(img)
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format='PNG')
            frames.append(img_bytes.getvalue())
        
        video1_data.append({
            'sample_id': i,
            'frames': frames  # 存储为bytes列表
        })
    
    df_video1 = pd.DataFrame(video1_data)
    df_video1.to_parquet(output_dir / 'video_stream_1.parquet')
    print(f"✓ 保存 {len(df_video1)} 个样本到 video_stream_1.parquet")
    
    # 2. 创建video_stream_2.parquet
    print("\n2️⃣ 创建 video_stream_2.parquet...")
    video2_data = []
    
    for i in range(num_samples):
        frames = []
        for t in range(sequence_length):
            # 创建绿色渐变的图像
            img = np.zeros((*frame_size, 3), dtype=np.uint8)
            img[:, :, 1] = int(100 + 100 * (t / sequence_length))  # 绿色通道
            
            # 添加不同的运动模式
            center_x = int(frame_size[1] * 0.5)
            center_y = int(frame_size[0] * (0.3 + 0.4 * np.cos(2 * np.pi * t / sequence_length)))
            
            y, x = np.ogrid[:frame_size[0], :frame_size[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= 15**2
            img[mask] = [0, 255, 255]  # 青色圆圈
            
            pil_img = Image.fromarray(img)
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format='PNG')
            frames.append(img_bytes.getvalue())
        
        video2_data.append({
            'sample_id': i,
            'frames': frames
        })
    
    df_video2 = pd.DataFrame(video2_data)
    df_video2.to_parquet(output_dir / 'video_stream_2.parquet')
    print(f"✓ 保存 {len(df_video2)} 个样本到 video_stream_2.parquet")
    
    # 3. 创建labels.parquet
    print("\n3️⃣ 创建 labels.parquet...")
    labels_data = []
    
    for i in range(num_samples):
        # 生成随机的delta向量 (dx, dy, dz)
        # 范围在 [-0.1, 0.1] 之间
        delta = np.random.uniform(-0.1, 0.1, size=3).astype(np.float32)
        
        labels_data.append({
            'sample_id': i,
            'delta': delta.tolist()  # 转为列表存储
        })
    
    df_labels = pd.DataFrame(labels_data)
    df_labels.to_parquet(output_dir / 'labels.parquet')
    print(f"✓ 保存 {len(df_labels)} 个标签到 labels.parquet")
    
    # 4. 显示数据格式
    print("\n" + "="*60)
    print("📋 数据格式示例:")
    print("="*60)
    
    print("\n📁 video_stream_1.parquet 结构:")
    print(df_video1.head(2))
    print(f"   - sample_id: {df_video1['sample_id'].dtype}")
    print(f"   - frames: list of {len(df_video1.iloc[0]['frames'])} images (bytes)")
    
    print("\n📁 video_stream_2.parquet 结构:")
    print(df_video2.head(2))
    
    print("\n📁 labels.parquet 结构:")
    print(df_labels.head(5))
    print(f"   - sample_id: {df_labels['sample_id'].dtype}")
    print(f"   - delta: shape {np.array(df_labels.iloc[0]['delta']).shape}, dtype float32")
    
    print("\n" + "="*60)
    print("✅ 示例数据创建完成！")
    print("="*60)
    print("\n下一步:")
    print(f"  python train_simple.py \\")
    print(f"    --video1 {output_dir}/video_stream_1.parquet \\")
    print(f"    --video2 {output_dir}/video_stream_2.parquet \\")
    print(f"    --labels {output_dir}/labels.parquet \\")
    print(f"    --config configs/default.yaml")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="创建示例parquet数据")
    parser.add_argument('--output-dir', type=str, default='data/sample',
                       help='输出目录')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='样本数量')
    args = parser.parse_args()
    
    create_sample_parquet_data(args.output_dir, args.num_samples)