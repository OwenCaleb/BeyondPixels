import cv2
import os
from pathlib import Path
import argparse


def extract_frames(video_path: str, out_dir: str, target_fps: float = 10.0, resize_to=(336, 336)):
    """
    从视频中按照 1 秒 10 帧的频率抽帧，并保存为图片。
    
    :param video_path: 输入视频路径
    :param out_dir: 导出图片的目录
    :param target_fps: 目标抽帧频率（默认 10fps，即 1s 10 帧）
    :param resize_to: 导出的图片尺寸（默认 (336, 336)，和 TraceVLA 里一致）
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if native_fps <= 0:
        print("Warning: failed to read FPS from video, will treat every frame as a sample.")
        native_fps = target_fps  # 避免除零

    print(f"Video: {video_path}")
    print(f"  native fps: {native_fps:.3f}")
    print(f"  total frames: {total_frames}")
    print(f"  target fps: {target_fps}")

    # 计算采样间隔：大致每隔多少帧取一帧
    frame_interval = max(int(round(native_fps / target_fps)), 1)
    print(f"  frame_interval (sample every N frames): {frame_interval}")

    frame_idx = 0        # 视频里的帧编号
    saved_idx = 0        # 已保存的图片编号

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 只在 frame_idx 能被 frame_interval 整除时抽帧
        if frame_idx % frame_interval == 0:
            saved_idx += 1

            # BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 可选：resize 到 TraceProcessor 代码中假定的 336x336（和 _filter_points 的 (336,336) 对齐）
            if resize_to is not None:
                frame_rgb = cv2.resize(frame_rgb, resize_to)

            # 保存为 PNG
            out_path = out_dir / f"{saved_idx:05d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        frame_idx += 1

    cap.release()
    print(f"Done. Saved {saved_idx} frames to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video at 1s 10 frames.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video.")
    parser.add_argument("--out_dir", type=str, default="frames_10fps", help="Directory to save frames.")
    parser.add_argument("--target_fps", type=float, default=10.0, help="Target extraction fps (default 10).")
    args = parser.parse_args()

    extract_frames(args.video, args.out_dir, target_fps=args.target_fps)


if __name__ == "__main__":
    main()

'''
python extract_frames.py \
  --video demo.mp4 \
  --out_dir png \
  --target_fps 10
'''