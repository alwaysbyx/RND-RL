#!/usr/bin/env python3
"""Dynamic GPU scheduler for Gym PPO experiments.

Monitors free VRAM on GPUs and dynamically schedules Gym PPO experiments
using train.py. Gym tasks are lightweight (~2-3 GB VRAM each).

Features:
  - Monitors GPU free VRAM every POLL_INTERVAL seconds
  - Submits a new task when a GPU has enough free VRAM
  - Checks wandb for already-finished runs and skips them
  - Dry-run mode to preview what would run

Experiment matrix:
  5 Gym envs × 4 actor variants × 5 seeds = 100 runs

Usage:
  python scripts/run_gym_dynamic.py              # run
  python scripts/run_gym_dynamic.py --dry-run    # preview
"""
import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict

# ---------------------------------------------------------------------------
# Environment configurations
# ---------------------------------------------------------------------------

GYM_ENV_CONFIGS = [
    {"env_id": "HalfCheetah-v4"},
    {"env_id": "Hopper-v4"},
    {"env_id": "Ant-v4"},
    {"env_id": "Walker2d-v4"},
    {"env_id": "Humanoid-v4"},
]

ACTOR_VARIANTS = [
    {"name": "discrete_residual", "discrete_action": True, "use_residual_blocks": True},
  #  {"name": "discrete_mlp", "discrete_action": True, "use_residual_blocks": False},
  #  {"name": "continuous_residual", "discrete_action": False, "use_residual_blocks": True},
  #  {"name": "continuous_mlp", "discrete_action": False, "use_residual_blocks": False},
]

SEEDS = [0, 1, 2, 3, 4]

# ---------------------------------------------------------------------------
# Gym-specific common settings
# ---------------------------------------------------------------------------

GYM_COMMON_SETTINGS = {
    "total_timesteps": 20_000_000,
    "num_envs": 32,
    "num_steps": 512,
    "num_bins": 41,
    "critic_width": 64,
    "critic_depth": 2,
}

# ---------------------------------------------------------------------------
# GPU settings (gym tasks are small, ~2-3 GB each)
# ---------------------------------------------------------------------------

GPU_IDS = [4, 5, 6, 7]
VRAM_THRESHOLD_MB = 20_000    # Need 5 GB free to launch a gym task
POLL_INTERVAL = 60           # Seconds between scheduling checks

# ---------------------------------------------------------------------------
# Wandb settings
# ---------------------------------------------------------------------------

WANDB_PROJECT = "ppo_gym_experiments"  # change to your wandb project
WANDB_ENTITY = None  # TODO: set to your wandb entity (required for the finished-run skip query)
BASE_CONFIG = "configs/gym.yaml"

GLOBAL_COMMON_SETTINGS = {
    "wandb_project_name": WANDB_PROJECT,
    "wandb_entity": WANDB_ENTITY,
    "track": True,
}

# Keys that exist in the YAML configs (no + prefix needed)
_YAML_KEYS = {
    "env_type", "env_id", "exp_name", "seed", "torch_deterministic", "cuda",
    "track", "wandb_project_name", "wandb_entity", "wandb_group",
    "capture_video", "save_trajectory", "save_model", "evaluate", "checkpoint",
    "num_envs", "num_eval_envs", "num_steps", "num_eval_steps",
    "partial_reset", "eval_partial_reset", "reconfiguration_freq",
    "eval_reconfiguration_freq", "control_mode", "save_train_video_freq",
    "total_timesteps", "learning_rate", "anneal_lr", "gamma", "gae_lambda",
    "num_minibatches", "update_epochs", "norm_adv", "clip_coef", "clip_vloss",
    "ent_coef", "vf_coef", "max_grad_norm", "target_kl", "reward_scale",
    "finite_horizon_gae", "eval_freq", "discrete_action", "num_bins",
    "actor_width", "actor_depth", "critic_width", "critic_depth",
    "use_residual_blocks",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def task_key(env_id, variant_name, seed):
    """Unique string identifier for a task."""
    return f"{env_id}|{variant_name}|seed{seed}"


def format_override(key, value):
    if isinstance(value, bool):
        return f"{key}={str(value).lower()}"
    return f"{key}={value}"


def hydra_override(key, value):
    formatted = format_override(key, value)
    if key not in _YAML_KEYS:
        return f"+{formatted}"
    return formatted


def build_overrides(env_config, actor_variant, seed):
    overrides = [
        hydra_override("env_id", env_config["env_id"]),
        hydra_override("seed", seed),
        hydra_override("discrete_action", actor_variant["discrete_action"]),
        hydra_override("use_residual_blocks", actor_variant["use_residual_blocks"]),
    ]
    # Gym-specific common settings
    for key, value in GYM_COMMON_SETTINGS.items():
        overrides.append(hydra_override(key, value))
    # Global shared settings
    for key, value in GLOBAL_COMMON_SETTINGS.items():
        overrides.append(hydra_override(key, value))
    return overrides


def build_all_tasks():
    """Build list of all tasks as dicts."""
    tasks = []
    for env_config in GYM_ENV_CONFIGS:
        for actor_variant in ACTOR_VARIANTS:
            for seed in SEEDS:
                tasks.append({
                    "env_id": env_config["env_id"],
                    "env_config": env_config,
                    "variant_name": actor_variant["name"],
                    "actor_variant": actor_variant,
                    "seed": seed,
                    "key": task_key(env_config["env_id"],
                                    actor_variant["name"], seed),
                })
    return tasks


def query_wandb_finished_runs():
    """Query wandb for finished runs and return set of task keys."""
    finished = set()
    try:
        import wandb
        api = wandb.Api(timeout=60)
        runs = api.runs(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}",
            filters={"state": "finished"},
        )
        for run in runs:
            run_full = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run.id}")
            cfg = run_full.config
            env_id = cfg.get("env_id", "")
            seed = cfg.get("seed", "")
            discrete = cfg.get("discrete_action", None)
            residual = cfg.get("use_residual_blocks", None)

            if discrete is None or residual is None:
                continue

            if discrete and residual:
                variant = "discrete_residual"
            elif discrete and not residual:
                variant = "discrete_mlp"
            elif not discrete and residual:
                variant = "continuous_residual"
            else:
                variant = "continuous_mlp"

            key = task_key(env_id, variant, seed)
            finished.add(key)

        print(f"[wandb] Found {len(finished)} finished runs in "
              f"{WANDB_ENTITY}/{WANDB_PROJECT}")
    except Exception as e:
        print(f"[wandb] Warning: Could not query wandb: {e}")
        print("[wandb] Proceeding without skipping — all tasks will be queued.")

    return finished


def get_gpu_free_vram():
    """Return dict of {gpu_id: free_vram_mb}."""
    gpu_ids_str = ",".join(str(g) for g in GPU_IDS)
    try:
        result = subprocess.run(
            ["nvidia-smi",
             f"--id={gpu_ids_str}",
             "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        free = {}
        for line in result.stdout.strip().split("\n"):
            parts = line.split(",")
            gpu_id = int(parts[0].strip())
            free_mb = int(parts[1].strip())
            free[gpu_id] = free_mb
        return free
    except Exception as e:
        print(f"[gpu] Warning: nvidia-smi failed: {e}")
        return {g: 0 for g in GPU_IDS}


def launch_task(task, gpu_id, project_dir):
    """Launch a single experiment as a subprocess. Returns Popen object."""
    overrides = build_overrides(
        task["env_config"], task["actor_variant"], task["seed"],
    )
    cmd = [
        sys.executable,
        "train.py",
        "--config", BASE_CONFIG,
        "--config_overrides",
    ] + overrides

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    proc = subprocess.Popen(
        cmd,
        cwd=project_dir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


# ---------------------------------------------------------------------------
# Main scheduler
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dynamic Gym PPO scheduler")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check wandb and print remaining tasks without running")
    parser.add_argument("--no-wandb-check", action="store_true",
                        help="Skip wandb check, run all tasks")
    cli_args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Build full task list
    all_tasks = build_all_tasks()
    print(f"Total experiments in matrix: {len(all_tasks)}")

    # Check wandb for finished runs
    if not cli_args.no_wandb_check:
        finished_keys = query_wandb_finished_runs()
        pending_tasks = [t for t in all_tasks if t["key"] not in finished_keys]
        skipped = len(all_tasks) - len(pending_tasks)
        print(f"Skipping {skipped} already-finished runs on wandb")
    else:
        pending_tasks = list(all_tasks)
        print("Skipping wandb check — all tasks queued")

    print(f"Remaining experiments to run: {len(pending_tasks)}")
    print(f"GPUs: {GPU_IDS}")
    print(f"VRAM threshold to launch: {VRAM_THRESHOLD_MB} MB free")
    print()

    if cli_args.dry_run:
        print("=" * 70)
        print("DRY RUN — remaining tasks:")
        print("=" * 70)
        by_env = defaultdict(list)
        for t in pending_tasks:
            by_env[t["env_id"]].append(t)
        for env_id, tasks in sorted(by_env.items()):
            print(f"\n  {env_id} ({len(tasks)} runs):")
            for t in tasks:
                print(f"    {t['variant_name']}:seed{t['seed']}")
        print(f"\nTotal remaining: {len(pending_tasks)} experiments")

        # Show current GPU state
        print("\n" + "=" * 70)
        print("Current GPU state:")
        print("=" * 70)
        free_vram = get_gpu_free_vram()
        for gpu_id in GPU_IDS:
            free = free_vram.get(gpu_id, 0)
            can_fit = max(0, free // VRAM_THRESHOLD_MB)
            print(f"  GPU {gpu_id}: free {free:,} MB "
                  f"→ can fit ~{can_fit} gym tasks now")
        return

    # --- Dynamic scheduling loop ---
    task_queue = list(pending_tasks)
    # {gpu_id: [(proc, task), ...]}
    running = defaultdict(list)
    completed = 0
    failed = 0
    total = len(task_queue)
    start_time = time.time()

    print("=" * 70)
    print(f"Starting dynamic scheduler — {total} tasks to run")
    print("=" * 70)

    while task_queue or any(running[g] for g in GPU_IDS):
        # 1. Reap finished processes
        for gpu_id in GPU_IDS:
            still_running = []
            for proc, task in running[gpu_id]:
                ret = proc.poll()
                if ret is None:
                    still_running.append((proc, task))
                elif ret == 0:
                    completed += 1
                    elapsed = time.time() - start_time
                    remaining = total - completed - failed
                    eta = (elapsed / completed * remaining) if completed else 0
                    print(f"[DONE]  {task['key']} (GPU {gpu_id})  "
                          f"[{completed + failed}/{total}, "
                          f"elapsed {elapsed/60:.0f}m, ETA {eta/60:.0f}m]")
                else:
                    failed += 1
                    print(f"[FAIL]  {task['key']} (GPU {gpu_id}, exit={ret})")
            running[gpu_id] = still_running

        # 2. Check free VRAM and submit tasks
        if task_queue:
            free_vram = get_gpu_free_vram()
            for gpu_id in GPU_IDS:
                if not task_queue:
                    break
                free = free_vram.get(gpu_id, 0)
                if free >= VRAM_THRESHOLD_MB:
                    task = task_queue.pop(0)
                    proc = launch_task(task, gpu_id, project_dir)
                    running[gpu_id].append((proc, task))
                    n_running = sum(len(v) for v in running.values())
                    print(f"[START] {task['key']} → GPU {gpu_id} "
                          f"(free: {free:,} MB, "
                          f"queued: {len(task_queue)}, running: {n_running})")

        # 3. Status line
        n_running = sum(len(v) for v in running.values())
        if n_running > 0 or task_queue:
            per_gpu = {g: len(running[g]) for g in GPU_IDS}
            elapsed = time.time() - start_time
            print(f"  [{time.strftime('%H:%M:%S')}] "
                  f"running={n_running} "
                  f"(GPU {', '.join(f'{g}:{per_gpu[g]}' for g in GPU_IDS)}) "
                  f"queued={len(task_queue)} "
                  f"done={completed} fail={failed} "
                  f"elapsed={elapsed/60:.0f}m",
                  flush=True)

        # 4. Wait before next poll
        if task_queue or any(running[g] for g in GPU_IDS):
            time.sleep(POLL_INTERVAL)

    # --- Final summary ---
    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print("Final Summary")
    print(f"{'=' * 70}")
    print(f"Succeeded: {completed}/{total}")
    print(f"Failed:    {failed}/{total}")
    print(f"Wall time: {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")


if __name__ == "__main__":
    main()
