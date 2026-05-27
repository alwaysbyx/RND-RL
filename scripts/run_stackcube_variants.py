#!/usr/bin/env python3
"""Run PPO on StackCube-v1 (seed=0, single GPU) across four architecture variants.

Variants:
    1. discrete  + residual blocks
    2. discrete  + MLP
    3. continuous + residual blocks
    4. continuous + MLP

Each run captures evaluation videos.
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Fixed experiment settings
ENV_ID = "StackCube-v1"
SEED = 0
# GPU assignment per variant. Repeat IDs to share a GPU across variants
# (each StackCube state-mode run uses only ~3.8 GB, so four fit easily on one 97 GB card).
GPU_IDS = [0, 0, 0, 0]
BASE_CONFIG = "configs/maniskill-state.yaml"

# Per-env extras carried over from run_maniskill_state_experiments.py
ENV_SETTINGS = {
    "num_envs": 4096,
    "total_timesteps": 50_000_000,
}

# The four architecture variants
VARIANTS = [
    {
        "name": "discrete_residual",
        "discrete_action": True,
        "use_residual_blocks": True,
    },
    {
        "name": "discrete_mlp",
        "discrete_action": True,
        "use_residual_blocks": False,
    },
    {
        "name": "continuous_residual",
        "discrete_action": False,
        "use_residual_blocks": True,
    },
    {
        "name": "continuous_mlp",
        "discrete_action": False,
        "use_residual_blocks": False,
    },
]

# Common settings for all variants
COMMON_SETTINGS = {
    "wandb_project_name": "stackcube_variants",
    "wandb_entity": None,  # TODO: set to your wandb entity
    "track": True,
    "capture_video": True,
    "save_model": True,
}


def build_overrides(variant: dict) -> list:
    overrides = [f"env_id={ENV_ID}", f"seed={SEED}"]

    for key, value in ENV_SETTINGS.items():
        overrides.append(f"{key}={value}")

    for key in ("discrete_action", "use_residual_blocks"):
        overrides.append(f"{key}={str(variant[key]).lower()}")

    overrides.append(
        f"exp_name={ENV_ID}_{variant['name']}_seed{SEED}"
    )

    for key, value in COMMON_SETTINGS.items():
        if isinstance(value, bool):
            overrides.append(f"{key}={str(value).lower()}")
        else:
            overrides.append(f"{key}={value}")

    return overrides


def run_variant(variant: dict, gpu_id: int) -> bool:
    task_name = f"{ENV_ID}_{variant['name']}_seed{SEED}"
    print("=" * 60)
    print(f"Launching: {task_name} (GPU {gpu_id})")
    print("=" * 60)

    overrides = build_overrides(variant)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    cmd = [
        sys.executable,
        "train.py",
        "--config", BASE_CONFIG,
        "--config_overrides",
    ]
    cmd.extend(overrides)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = os.path.join(
        script_dir, f"stackcube_{variant['name']}.log"
    )
    print(f"Command: {' '.join(cmd)}")
    print(f"GPU: {gpu_id} | Log: {log_path}")
    print()

    with open(log_path, "w") as logf:
        try:
            subprocess.run(
                cmd,
                check=True,
                cwd=project_dir,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
            print(f"[OK]  {task_name} (GPU {gpu_id})")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {task_name} (GPU {gpu_id}): {e}")
            return False


def main():
    assert len(GPU_IDS) >= len(VARIANTS), (
        f"Need at least {len(VARIANTS)} GPUs for full parallelism, got {len(GPU_IDS)}"
    )

    print(f"StackCube-v1 four-variant sweep | seed={SEED}")
    print(f"Variants: {[v['name'] for v in VARIANTS]}")
    print(f"GPUs:     {GPU_IDS[: len(VARIANTS)]}")
    print(f"Mode:     parallel (one variant per GPU)")
    print()

    start = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=len(VARIANTS)) as executor:
        future_to_name = {}
        for variant, gpu_id in zip(VARIANTS, GPU_IDS):
            fut = executor.submit(run_variant, variant, gpu_id)
            future_to_name[fut] = variant["name"]

        for fut in as_completed(future_to_name):
            name = future_to_name[fut]
            try:
                ok = fut.result()
            except Exception as e:
                print(f"[EXC] {name}: {e}")
                ok = False
            results.append((name, ok))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, ok in results:
        print(f"  {name}: {'SUCCESS' if ok else 'FAILED'}")

    total = time.time() - start
    n_ok = sum(1 for _, ok in results if ok)
    print(f"\n{n_ok}/{len(results)} variants completed in {total/60:.1f} min")


if __name__ == "__main__":
    main()
