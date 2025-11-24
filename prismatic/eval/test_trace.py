import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch

# 如果你的 TraceProcessor 在 prismatic/eval 下面，用这个：
# from prismatic.eval.trace_processor import TraceProcessor

# 如果你把类单独存成 trace_processor.py 放在同一目录，就用这个：
from trace_processor import TraceProcessor


def load_images_from_dir(image_dir: Path):
    """从文件夹中按文件名排序加载所有图片，返回 [PIL.Image, ...] 列表。"""
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])
    if not files:
        raise ValueError(f"No images found in {image_dir}")
    images = []
    for p in files:
        img = Image.open(p).convert("RGB")
        # TraceVLA 代码里后面用 (336, 336) 过滤点，这里直接统一 resize，避免尺寸不一致
        img = img.resize((336, 336))
        images.append((p.name, img))
    return images


def main():
    parser = argparse.ArgumentParser(description="Test TraceProcessor with a sequence of images.")
    parser.add_argument(
        "--cotracker_ckpt",
        type=str,
        required=True,
        help="Path to CoTracker checkpoint, e.g., scaled_offline.pth",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing image frames to process.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="trace_output",
        help="Directory to save overlaid images.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='Device for CoTracker, e.g. "cuda:0" or "cpu".',
    )
    parser.add_argument(
        "--begin_track_step",
        type=int,
        default=10,
        help="At which step to start tracking.",
    )
    parser.add_argument(
        "--redraw_frequency",
        type=int,
        default=25,
        help="How often (in steps) to recompute tracks from scratch.",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=5,
        help="Number of active points to keep for visualization.",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=10,
        help="Number of recent frames to keep in the video buffer.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=15,
        help="Number of recent timesteps to visualize in the trace.",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 将 device 转成 TraceProcessor 期望的形式
    # 原类里是 .to(device)，如果你写的是 "cuda:0" / "cpu" 也可以
    device = args.device

    print(f"Loading images from: {image_dir}")
    images = load_images_from_dir(image_dir)
    print(f"Found {len(images)} images.")

    # 初始化 TraceProcessor
    print("Initializing TraceProcessor...")
    tp = TraceProcessor(
        cotracker_model_path=args.cotracker_ckpt,
        begin_track_step=args.begin_track_step,
        redraw_frequency=args.redraw_frequency,
        num_points=args.num_points,
        buffer_size=args.buffer_size,
        window_size=args.window_size,
        device=device,
    )

    print("Processing frames...")
    first_trace_step = None

    for idx, (name, img) in enumerate(images, start=1):
        image_overlaid, has_trace = tp.process_image(img)

        if has_trace and first_trace_step is None:
            first_trace_step = tp.step

        # 保存可视化图像
        out_path = out_dir / f"{idx:05d}_{'trace' if has_trace else 'plain'}_{name}"
        image_overlaid.save(out_path)

        print(
            f"[step {tp.step:03d}] file={name} "
            f"has_trace={has_trace} "
            f"traced={tp.traced} "
            f"trace_buffer_len={0 if tp.trace_buffer is None else tp.trace_buffer.shape[0]}"
        )

    print("Done.")
    if first_trace_step is not None:
        print(f"First valid trace appeared at step {first_trace_step}.")
    else:
        print("No valid trace was found (CoTracker did not detect enough active points).")


if __name__ == "__main__":
    main()

'''
python test_trace.py \
  --cotracker_ckpt /home/liwenbo/projects/Robotic_Manipulation/VLA/Tools/co-tracker/checkpoints/scaled_offline.pth \
  --image_dir png \
  --out_dir trace_vis \
  --device cuda:0
'''