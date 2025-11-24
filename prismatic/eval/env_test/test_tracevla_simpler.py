#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TraceVLA + SimplerEnv 简单联调脚本

功能：
- 创建一个 SimplerEnv 环境（默认 google_robot_pick_coke_can）
- 用 TraceVLAInference 作为策略
- 每一步：
    - 从环境取图像 + 指令
    - 走 TraceProcessor 生成轨迹叠加图
    - 调用 TraceVLA.predict_action 拿到 7D action
    - 转成 SimplerEnv 期望的 7 维向量 [dx, dy, dz, rot_axangle(3), gripper]
    - env.step(action)
"""

import argparse
import os
from typing import Dict, Any, List

import numpy as np

import simpler_env
from simpler_env.utils.env.observation_utils import (
    get_image_from_maniskill2_obs_dict,
)

# 这行根据你实际文件名修改：
# 比如你是 prismatic/eval/tracevla_policy.py 里面定义的 TraceVLAInference
from ../tracevla_inference import TraceVLAInference


def action_dict_to_vec(action: Dict[str, np.ndarray]) -> np.ndarray:
    """
    TraceVLAInference.get_action 返回的是一个 dict：
        {
          "world_vector": np.array(3,),
          "rot_axangle": np.array(3,),
          "gripper": np.array(1,) 或 标量,
          "terminate_episode": np.array([0.0]),
        }
    SimplerEnv 的 env.action_space.sample() 是 7 维：
        [:3]  = delta xyz
        [3:6] = delta rotation (axis-angle)
        [6]   = gripper
    这里做一个小转换。
    """
    vec = np.zeros(7, dtype=np.float32)
    vec[:3] = np.asarray(action["world_vector"], dtype=np.float32)
    vec[3:6] = np.asarray(action["rot_axangle"], dtype=np.float32)
    # gripper 可能是标量，也可能是 shape (1,)
    g = np.asarray(action["gripper"], dtype=np.float32).reshape(-1)[0]
    vec[6] = g
    return vec


def run(args: argparse.Namespace):
    # 1. 创建 SimplerEnv 环境
    print(f"[INFO] Creating SimplerEnv env: {args.env_name}")
    env = simpler_env.make(args.env_name)

    # 2. 创建 TraceVLA 策略
    print(f"[INFO] Initializing TraceVLAInference with model: {args.model_path}")
    policy = TraceVLAInference(
        model_path=args.model_path,
        cotracker_model_path=args.cotracker_ckpt,
        dataset_stats_path=args.dataset_stats_path,
        action_scale=args.action_scale,
        sample=args.sample,
        temperature=args.temperature,
        device=args.device,
    )

    # 3. 启动 policy，多环境接口，这里只开 1 个 env
    policy.start(num_envs=1)

    os.makedirs(args.out_dir, exist_ok=True)

    for ep in range(args.num_episodes):
        print(f"\n[INFO] ===== Episode {ep} =====")
        obs, reset_info = env.reset()
        instruction = env.get_language_instruction()
        print("[INFO] Reset info:", reset_info)
        print("[INFO] Initial instruction:", instruction)

        # 构造 obs_dicts / info_dicts，符合 TraceVLAInference 的期望格式
        obs_dicts: List[Dict[str, Any]] = [obs]

        image = get_image_from_maniskill2_obs_dict(env, obs)
        info_dicts: List[Dict[str, Any]] = [
            {
                "image": image,                      # H×W×3, uint8
                "task_description": instruction,     # 文本指令
                "policy_setup": args.policy_setup,   # "widowx_bridge" or "google_robot"
            }
        ]

        ids = [0]
        output_dirs = [args.out_dir]

        # 通知 policy：一个新 episode 开始了
        policy.start_episode(
            obs_dicts=obs_dicts,
            info_dicts=info_dicts,
            output_dirs=output_dirs,
            ids=ids,
        )

        done, truncated = False, False
        step = 0

        while not (done or truncated) and step < args.max_steps:
            # 更新 obs / info，用最新观测构造 batch
            image = get_image_from_maniskill2_obs_dict(env, obs)
            obs_dicts = [obs]
            info_dicts = [
                {
                    "image": image,
                    "task_description": instruction,
                    "policy_setup": args.policy_setup,
                }
            ]

            # 4. 用 TraceVLAInference 预测动作
            actions_list = policy.get_action(
                obs_dicts=obs_dicts,
                info_dicts=info_dicts,
                ids=ids,
            )
            assert len(actions_list) == 1
            action_dict = actions_list[0]
            action_vec = action_dict_to_vec(action_dict)

            # 打个 log 看看
            if step % args.log_every == 0:
                print(
                    f"[Step {step:03d}] action_vec = {action_vec}, "
                    f"world = {action_dict['world_vector']}, "
                    f"gripper = {action_dict['gripper']}"
                )

            # 5. 把动作喂给 SimplerEnv
            obs, reward, done, truncated, info = env.step(action_vec)

            # 对于长程任务，指令可能在子任务之间发生变化
            new_instruction = env.get_language_instruction()
            if new_instruction != instruction:
                print(f"[Step {step:03d}] New instruction: {new_instruction}")
                instruction = new_instruction
                # 同时重置 policy 内部关于该 env 的 sticky gripper / trace 状态
                policy.reset_states_at(0, instruction)

            step += 1

        episode_stats = info.get("episode_stats", {})
        print(f"[INFO] Episode {ep} finished. episode_stats = {episode_stats}")

    env.close()
    print("[INFO] Done.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimal TraceVLA + SimplerEnv test script"
    )
    # 环境 & 机器人配置
    parser.add_argument(
        "--env_name",
        type=str,
        default="google_robot_pick_coke_can",
        help="SimplerEnv env name, e.g. google_robot_pick_coke_can / widowx_bridge_pick_coke_can",
    )
    parser.add_argument(
        "--policy_setup",
        type=str,
        default="google_robot",
        choices=["google_robot", "widowx_bridge"],
        help="Should match the robot type used by the env.",
    )

    # TraceVLA 模型 & 统计 & cotracker
    parser.add_argument(
        "--model_path",
        type=str,
        default="furonghuang-lab/tracevla_7b",
        help="HuggingFace model id or local path for TraceVLA.",
    )
    parser.add_argument(
        "--dataset_stats_path",
        type=str,
        required=True,
        help="Path to dataset_stats.json used for un-normalizing actions.",
    )
    parser.add_argument(
        "--cotracker_ckpt",
        type=str,
        required=True,
        help="Path to CoTracker checkpoint, e.g. scaled_offline.pth",
    )

    # 推理相关
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for TraceVLA (e.g. cuda:0 / cpu).",
    )
    parser.add_argument(
        "--action_scale",
        type=float,
        default=1.0,
        help="Global scale for position & rotation actions.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use stochastic sampling for actions (do_sample=True).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature if --sample is enabled.",
    )

    # 运行控制
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="Max steps per episode.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=5,
        help="Print action every N steps.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="tracevla_simpler_logs",
        help="Where to store any logs / debug outputs.",
    )

    return parser


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    run(args)

'''
python test_tracevla_simpler.py \
  --dataset_stats_path /path/to/dataset_stats.json \
  --cotracker_ckpt /home/liwenbo/projects/Robotic_Manipulation/VLA/Tools/co-tracker/checkpoints/scaled_offline.pth \
  --env_name widowx_bridge_pick_coke_can \
  --policy_setup widowx_bridge \
  --model_path furonghuang-lab/tracevla_7b \
  --device cuda:0

'''