import os
from pathlib import Path
import argparse

import cv2
import numpy as np


def images_to_video(
    image_dir: str,
    out_path: str,
    fps: float = 10.0,
    resize_to: tuple | None = None,
):
    """
    将指定文件夹下的图片按文件名顺序合成为视频。

    :param image_dir: 图片所在目录
    :param out_path: 输出视频路径（例如 output.mp4）
    :param fps: 视频帧率
    :param resize_to: (width, height)，如果为 None，则使用第一张图片的尺寸
    """
    image_dir = Path(image_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])

    if not files:
        raise ValueError(f"No image files found in directory: {image_dir}")

    print(f"Found {len(files)} images in {image_dir}")

    # 读取第一张图，确定尺寸
    first = cv2.imread(str(files[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first image: {files[0]}")

    if resize_to is None:
        height, width = first.shape[:2]
    else:
        width, height = resize_to
        first = cv2.resize(first, (width, height))

    # 使用 mp4v 编码器，你也可以改成 XVID 等
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    print(f"Writing video to: {out_path}")
    print(f"  fps: {fps}")
    print(f"  size: {width}x{height}")

    # 先写入第一张
    writer.write(first)

    # 依次写入后续图片
    for i, p in enumerate(files[1:], start=2):
        frame = cv2.imread(str(p))
        if frame is None:
            print(f"[WARN] Failed to read {p}, skip.")
            continue

        # 尺寸不一致时做 resize
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        writer.write(frame)
        if i % 50 == 0 or i == len(files):
            print(f"  wrote {i}/{len(files)} frames")

    writer.release()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Convert images in a folder to a video.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory with input images.")
    parser.add_argument("--out", type=str, default="output.mp4", help="Output video path.")
    parser.add_argument("--fps", type=float, default=10.0, help="Frames per second.")
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional target width. If set, will resize all frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional target height. If set, will resize all frames.",
    )
    args = parser.parse_args()

    resize_to = None
    if args.width is not None and args.height is not None:
        resize_to = (args.width, args.height)

    images_to_video(
        image_dir=args.image_dir,
        out_path=args.out,
        fps=args.fps,
        resize_to=resize_to,
    )


if __name__ == "__main__":
    main()


'''
python images_to_video.py \
  --image_dir trace_vis \
  --out trace_tracevla.mp4 \
  --fps 10

'''