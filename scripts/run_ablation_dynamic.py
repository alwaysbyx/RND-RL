#!/usr/bin/env python3
"""Dynamic GPU scheduler for RND-actor d_h (actor_width) ablation.

Sweeps actor_width over {64, 128, 256, 512} on two environments
(StackCube-v1 state + Humanoid-v4 gym) using train.py with the RND
actor (discrete_action=True, use_residual_blocks=True).

Experiment matrix:
  2 envs * 4 widths * 5 seeds = 40 runs on GPUs 4/5/6/7

Usage:
  python scripts/run_ablation_dynamic.py              # run
  python scripts/run_ablation_dynamic.py --dry-run    # preview
"""
import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict

# ---------------------------------------------------------------------------
# Task families: (gym + maniskill_state) each use their own base config
# ---------------------------------------------------------------------------

TASK_FAMILIES = [
    {
        "name": "gym",
        "base_config": "configs/gym.yaml",
        "env_configs": [{"env_id": "Humanoid-v4"}],
        "common_settings": {
            "total_timesteps": 20_000_000,
            "num_envs": 16,
            "num_steps": 1024,
            "num_bins": 41,
            "critic_width": 64,
            "critic_depth": 2,
        },
    },
    {
        "name": "maniskill_state",
        "base_config": "configs/maniskill-state.yaml",
        "env_configs": [
            {"env_id": "StackCube-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": ""},
        ],
        "common_settings": {},
    },
]

# d_h sweep (actor hidden width)
ACTOR_WIDTHS = [64, 128, 256, 512]

SEEDS = [0, 1, 2, 3, 4]

# RND actor = discrete bins + residual blocks
RND_METHOD = {"discrete_action": True, "use_residual_blocks": True}

# ---------------------------------------------------------------------------
# GPU settings
# ---------------------------------------------------------------------------

GPU_IDS = [4, 5, 6, 7]
# Gym Humanoid-v4 with num_envs=16 takes ~2-3 GB.
# StackCube state with num_envs=4096 takes ~10-15 GB.
# Threshold of 15 GB lets state tasks fit safely; gym tasks easily fit.
VRAM_THRESHOLD_MB = 15_000
POLL_INTERVAL = 60
LAUNCH_COOLDOWN = 60

# ---------------------------------------------------------------------------
# Global wandb settings
# ---------------------------------------------------------------------------

GLOBAL_COMMON_SETTINGS = {
    "wandb_project_name": "rnd_dh_ablation",
    "wandb_entity": None,  # TODO: set to your wandb entity
    "track": True,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def task_key(family_name, env_id, width, seed):
    return f"{family_name}|{env_id}|dh{width}|seed{seed}"


def parse_extra_args(extra_str: str):
    if not extra_str or not extra_str.strip():
        return {}

    overrides = {}
    parts = extra_str.strip().split()
    i = 0
    while i < len(parts):
        arg = parts[i]
        if arg.startswith("--"):
            if "=" in arg:
                key, value = arg[2:].split("=", 1)
            else:
                key = arg[2:]
                if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                    value = parts[i + 1]
                    i += 1
                else:
                    overrides[key.replace("-", "_")] = True
                    i += 1
                    continue

            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
            overrides[key.replace("-", "_")] = value
        i += 1
    return overrides


def format_override(key, value):
    if isinstance(value, bool):
        return f"{key}={str(value).lower()}"
    return f"{key}={value}"


def build_overrides(task_family, env_config, width, seed):
    overrides = [f"env_id={env_config['env_id']}", f"seed={seed}"]

    # RND actor (discrete + residual)
    overrides.append(format_override("discrete_action", RND_METHOD["discrete_action"]))
    overrides.append(format_override("use_residual_blocks", RND_METHOD["use_residual_blocks"]))

    # d_h sweep
    overrides.append(format_override("actor_width", width))

    # Env-specific settings (num_envs, total_timesteps)
    for key, value in env_config.items():
        if key in {"env_id", "extra"}:
            continue
        overrides.append(format_override(key, value))

    # Env-specific extra-args string
    if "extra" in env_config:
        extra_overrides = parse_extra_args(env_config["extra"])
        for key, value in extra_overrides.items():
            overrides.append(format_override(key, value))

    # Task-family defaults
    for key, value in task_family["common_settings"].items():
        overrides.append(format_override(key, value))

    # Global shared settings
    for key, value in GLOBAL_COMMON_SETTINGS.items():
        overrides.append(format_override(key, value))

    return overrides


def build_all_tasks():
    tasks = []
    for family in TASK_FAMILIES:
        for env_config in family["env_configs"]:
            for width in ACTOR_WIDTHS:
                for seed in SEEDS:
                    tasks.append({
                        "family": family,
                        "family_name": family["name"],
                        "env_id": env_config["env_id"],
                        "env_config": env_config,
                        "width": width,
                        "seed": seed,
                        "key": task_key(family["name"], env_config["env_id"], width, seed),
                    })
    return tasks


def get_gpu_free_vram():
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
    overrides = build_overrides(task["family"], task["env_config"], task["width"], task["seed"])
    cmd = [
        sys.executable,
        "train.py",
        "--config", task["family"]["base_config"],
        "--config_overrides",
    ] + overrides

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_dir = os.path.join(project_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"ablation_{task['key'].replace('|', '_')}.log")
    fh = open(log_file, "w")

    proc = subprocess.Popen(
        cmd,
        cwd=project_dir,
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
    )
    return proc, fh, log_file


# ---------------------------------------------------------------------------
# Main scheduler
# ---------------------------------------------------------------------------

def main():
    global GPU_IDS, VRAM_THRESHOLD_MB, POLL_INTERVAL, LAUNCH_COOLDOWN

    parser = argparse.ArgumentParser(description="d_h ablation scheduler")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print remaining tasks and current GPU state without running")
    parser.add_argument("--vram-threshold-mb", type=int, default=VRAM_THRESHOLD_MB,
                        help=f"Min free VRAM (MB) needed to launch (default: {VRAM_THRESHOLD_MB})")
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL,
                        help=f"Seconds between scheduling checks (default: {POLL_INTERVAL})")
    parser.add_argument("--launch-cooldown", type=int, default=LAUNCH_COOLDOWN,
                        help=f"Cooldown after launching before re-querying VRAM (default: {LAUNCH_COOLDOWN})")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs (default: %s)" % ",".join(str(g) for g in GPU_IDS))
    cli_args = parser.parse_args()

    if cli_args.gpus:
        GPU_IDS = [int(g) for g in cli_args.gpus.split(",")]
    VRAM_THRESHOLD_MB = cli_args.vram_threshold_mb
    POLL_INTERVAL = cli_args.poll_interval
    LAUNCH_COOLDOWN = cli_args.launch_cooldown

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    all_tasks = build_all_tasks()
    print("=" * 70)
    print("RND-actor d_h Ablation Scheduler")
    print("=" * 70)
    print(f"Total experiments: {len(all_tasks)} "
          f"({sum(len(f['env_configs']) for f in TASK_FAMILIES)} envs * "
          f"{len(ACTOR_WIDTHS)} widths * {len(SEEDS)} seeds)")
    print(f"Envs: {[(f['name'], [e['env_id'] for e in f['env_configs']]) for f in TASK_FAMILIES]}")
    print(f"Actor widths (d_h): {ACTOR_WIDTHS}")
    print(f"Seeds: {SEEDS}")
    print(f"Actor: RND (discrete=true, residual=true)")
    print(f"GPUs: {GPU_IDS}")
    print(f"VRAM threshold to launch: {VRAM_THRESHOLD_MB:,} MB free")
    print(f"Poll interval: {POLL_INTERVAL}s, launch cooldown: {LAUNCH_COOLDOWN}s")
    print(f"W&B project: {GLOBAL_COMMON_SETTINGS['wandb_project_name']}")
    print()

    if cli_args.dry_run:
        print("=" * 70)
        print("DRY RUN - tasks:")
        print("=" * 70)
        by_env = defaultdict(list)
        for t in all_tasks:
            by_env[(t["family_name"], t["env_id"])].append(t)
        for (family_name, env_id), tasks in sorted(by_env.items()):
            print(f"\n  [{family_name}] {env_id} ({len(tasks)} runs):")
            by_width = defaultdict(list)
            for t in tasks:
                by_width[t["width"]].append(t)
            for width in sorted(by_width.keys()):
                seeds_str = ", ".join(f"seed{t['seed']}" for t in by_width[width])
                print(f"    d_h={width}: {seeds_str}")

        print("\n" + "=" * 70)
        print("Current GPU state:")
        print("=" * 70)
        free_vram = get_gpu_free_vram()
        for gpu_id in GPU_IDS:
            free = free_vram.get(gpu_id, 0)
            can_fit = max(0, free // VRAM_THRESHOLD_MB)
            print(f"  GPU {gpu_id}: free {free:,} MB -> can fit ~{can_fit} tasks now")
        return

    # --- Dynamic scheduling loop ---
    task_queue = list(all_tasks)
    running = defaultdict(list)
    completed = 0
    failed = 0
    total = len(task_queue)
    start_time = time.time()

    print("=" * 70)
    print(f"Starting dynamic scheduler - {total} tasks to run")
    print("=" * 70)

    while task_queue or any(running[g] for g in GPU_IDS):
        # 1. Reap finished
        for gpu_id in GPU_IDS:
            still_running = []
            for proc, task, fh in running[gpu_id]:
                ret = proc.poll()
                if ret is None:
                    still_running.append((proc, task, fh))
                else:
                    fh.close()
                    if ret == 0:
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

        # 2. Round-robin launches across GPUs
        launched_this_cycle = 0
        while task_queue:
            free_vram = get_gpu_free_vram()
            gpu_order = sorted(GPU_IDS, key=lambda g: len(running[g]))
            launched_this_pass = 0
            for gpu_id in gpu_order:
                if not task_queue:
                    break
                free = free_vram.get(gpu_id, 0)
                if free < VRAM_THRESHOLD_MB:
                    continue
                task = task_queue.pop(0)
                proc, fh, log_file = launch_task(task, gpu_id, project_dir)
                running[gpu_id].append((proc, task, fh))
                n_running = sum(len(v) for v in running.values())
                print(f"[START] {task['key']} -> GPU {gpu_id} "
                      f"(free: {free:,} MB, queued: {len(task_queue)}, "
                      f"running: {n_running}, log: {log_file})", flush=True)
                launched_this_pass += 1
                launched_this_cycle += 1
            if launched_this_pass == 0:
                break
            if task_queue:
                time.sleep(LAUNCH_COOLDOWN)

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
            wait_s = POLL_INTERVAL if launched_this_cycle == 0 else 5
            time.sleep(wait_s)

    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print("Final Summary")
    print(f"{'=' * 70}")
    print(f"Succeeded: {completed}/{total}")
    print(f"Failed:    {failed}/{total}")
    print(f"Wall time: {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")


if __name__ == "__main__":
    main()
