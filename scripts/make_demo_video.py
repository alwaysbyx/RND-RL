#!/usr/bin/env python3
"""Render a 2x2 demo video stitching the four StackCube PPO variants.

For each variant:
  - load `runs/<run_name>/final_ckpt.pt`
  - roll out the policy for DEMO_STEPS env steps with a single environment
  - record a single mp4 via ManiSkill RecordEpisode

Then ffmpeg combines the four mp4s into a 2x2 grid with a label overlay.

Run after `run_stackcube_variants.py` finishes:
    python scripts/make_demo_video.py
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import imageio_ffmpeg
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mani_skill.envs  # noqa: F401  (registers envs)
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from rnd import Agent
from train import load_config_hydra


ENV_ID = "StackCube-v1"
SEED = 0
NUM_DEMO_ENVS = 1
DEMO_STEPS = 300        # 300 / 30fps = 10s per panel
VIDEO_FPS = 30
BASE_CONFIG = str(ROOT / "configs/maniskill-state.yaml")

# Edit `label` for the on-screen text of each panel.
VARIANTS = [
    {"name": "discrete_residual", "label": "PPO  Discrete + Residual",
     "discrete_action": True,  "use_residual_blocks": True},
    {"name": "discrete_mlp", "label": "PPO  Discrete + MLP",
     "discrete_action": True,  "use_residual_blocks": False},
    {"name": "continuous_residual", "label": "PPO  Continuous + Residual",
     "discrete_action": False, "use_residual_blocks": True},
    {"name": "continuous_mlp", "label": "PPO  Continuous + MLP",
     "discrete_action": False, "use_residual_blocks": False},
]


def render_variant(variant: dict, ckpt_path: Path, out_dir: Path, device: torch.device) -> Path:
    overrides = [
        f"env_id={ENV_ID}",
        f"seed={SEED}",
        f"discrete_action={str(variant['discrete_action']).lower()}",
        f"use_residual_blocks={str(variant['use_residual_blocks']).lower()}",
    ]
    args = load_config_hydra(BASE_CONFIG, overrides=overrides)

    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="physx_cuda",
        control_mode=args.control_mode,
    )
    env = gym.make(ENV_ID, num_envs=NUM_DEMO_ENVS, reconfiguration_freq=1, **env_kwargs)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = RecordEpisode(
        env,
        output_dir=str(out_dir),
        save_trajectory=False,
        save_video=True,
        trajectory_name=variant["name"],
        max_steps_per_video=DEMO_STEPS,
        video_fps=VIDEO_FPS,
    )
    env = ManiSkillVectorEnv(env, NUM_DEMO_ENVS, ignore_terminations=True, record_metrics=False)

    n_obs = int(torch.prod(torch.tensor(env.single_observation_space.shape)))
    n_act = int(torch.prod(torch.tensor(env.single_action_space.shape)))
    agent = Agent(
        n_obs=n_obs,
        n_act=n_act,
        action_space=env.single_action_space,
        args=args,
        device=device,
        sample_obs=None,
    ).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()

    obs, _ = env.reset(seed=SEED)
    with torch.no_grad():
        for _ in range(DEMO_STEPS):
            action, _, _, _ = agent.get_action_and_value(obs)
            obs, _, _, _, _ = env.step(action)
    env.close()  # flushes the mp4

    mp4s = sorted(out_dir.glob("*.mp4"))
    if not mp4s:
        raise RuntimeError(f"No mp4 produced in {out_dir}")
    return mp4s[0]


def stitch(per_variant_videos: list, output_path: Path, ffmpeg_bin: str) -> None:
    inputs = []
    for v in per_variant_videos:
        inputs.extend(["-i", str(v)])

    # Per-panel: scale to a uniform 720x720, then overlay a centered top label
    label_filters = []
    for i, v in enumerate(VARIANTS):
        text = v["label"].replace(":", r"\:").replace("'", r"\'")
        label_filters.append(
            f"[{i}:v]scale=720:720:force_original_aspect_ratio=decrease,"
            f"pad=720:720:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"drawtext=text='{text}':"
            f"fontcolor=white:fontsize=34:"
            f"box=1:boxcolor=black@0.6:boxborderw=12:"
            f"x=(w-text_w)/2:y=24[v{i}]"
        )

    grid = (
        "[v0][v1]hstack=inputs=2[top];"
        "[v2][v3]hstack=inputs=2[bot];"
        "[top][bot]vstack=inputs=2[out]"
    )
    filter_complex = ";".join(label_filters + [grid])

    cmd = [
        ffmpeg_bin, "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-r", str(VIDEO_FPS),
        str(output_path),
    ]
    print("[stitch] running ffmpeg")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-name", default="final_ckpt.pt",
        help="checkpoint filename inside each run directory",
    )
    parser.add_argument(
        "--output", default=str(ROOT / "runs/demo/stackcube_demo.mp4"),
    )
    parser.add_argument("--keep-temp", action="store_true",
                        help="keep per-variant intermediate videos")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    out_root = ROOT / "runs/demo"
    out_root.mkdir(parents=True, exist_ok=True)

    per_variant_videos = []
    for v in VARIANTS:
        run_dir = ROOT / f"runs/{ENV_ID}_{v['name']}_seed{SEED}"
        ckpt = run_dir / args.ckpt_name
        if not ckpt.exists():
            raise FileNotFoundError(
                f"missing checkpoint: {ckpt} — wait for run_stackcube_variants.py to finish"
            )
        v_dir = out_root / v["name"]
        if v_dir.exists():
            shutil.rmtree(v_dir)
        v_dir.mkdir(parents=True)
        print(f"[render] {v['name']}  <-  {ckpt}")
        mp4 = render_variant(v, ckpt, v_dir, device)
        per_variant_videos.append(mp4)
        print(f"        -> {mp4}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    stitch(per_variant_videos, output_path, ffmpeg_bin)

    print(f"\n[done] demo video: {output_path}")
    if not args.keep_temp:
        for v in VARIANTS:
            shutil.rmtree(out_root / v["name"], ignore_errors=True)


if __name__ == "__main__":
    main()
