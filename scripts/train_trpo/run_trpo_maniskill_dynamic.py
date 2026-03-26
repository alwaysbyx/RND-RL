#!/usr/bin/env python3
"""Dynamic GPU scheduler for ManiSkill TRPO experiments.

Monitors free VRAM on GPUs 4-7 and dynamically schedules ManiSkill experiments
to maintain ~80% GPU utilization. Coexists with running gym experiments.

Features:
  - Monitors GPU free VRAM every POLL_INTERVAL seconds
  - Submits a new task when a GPU has enough free VRAM (VRAM_THRESHOLD_MB)
  - Checks wandb for already-finished runs and skips them
  - Dry-run mode to preview what would run

Experiment matrix:
  10 ManiSkill envs × 4 actor variants × 1 max_kl (0.01) × 5 seeds = 200 runs

Usage:
  python scripts/train_trpo/run_trpo_maniskill_dynamic.py              # run
  python scripts/train_trpo/run_trpo_maniskill_dynamic.py --dry-run    # preview
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

MANISKILL_STATE_ENV_CONFIGS = [
    {"env_id": "PushCube-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=4"},
    {"env_id": "PickCube-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=4"},
    {"env_id": "PickCubeSO100-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=8"},
    {"env_id": "PushT-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=16 --gamma=0.99 --num-eval-steps=100"},
    {"env_id": "StackCube-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": ""},
    {"env_id": "RollBall-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=16 --num-eval-steps=80 --gamma=0.95"},
    {"env_id": "PullCube-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=4"},
    {"env_id": "PokeCube-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=4"},
    {"env_id": "LiftPegUpright-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=4"},
    {"env_id": "PickSingleYCB-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=16"},
]

ACTOR_VARIANTS = [
    {"name": "discrete_residual", "discrete_action": True, "use_residual_blocks": True},
    {"name": "discrete_mlp", "discrete_action": True, "use_residual_blocks": False},
    {"name": "continuous_residual", "discrete_action": False, "use_residual_blocks": True},
    {"name": "continuous_mlp", "discrete_action": False, "use_residual_blocks": False},
]

MAX_KL_VALUES = [0.01]
SEEDS = [0, 1, 2, 3, 4]

# ---------------------------------------------------------------------------
# GPU settings
# ---------------------------------------------------------------------------

GPU_IDS = [4, 5, 6, 7]
VRAM_THRESHOLD_MB = 12_000   # Need 12 GB free to launch a task (~10 GB + margin)
VRAM_SAFETY_MB = 19_000      # Keep 19 GB free per GPU (target ~80% of 96 GB)
POLL_INTERVAL = 30           # Seconds between scheduling checks

# ---------------------------------------------------------------------------
# Wandb settings
# ---------------------------------------------------------------------------

WANDB_PROJECT = "trpo_maniskill_experiments"
WANDB_ENTITY = "jif005-ucsd"
BASE_CONFIG = "configs/maniskill-state.yaml"

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

def task_key(env_id, variant_name, max_kl, seed):
    """Unique string identifier for a task."""
    return f"{env_id}|{variant_name}|kl{max_kl}|seed{seed}"


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


def hydra_override(key, value):
    formatted = format_override(key, value)
    if key not in _YAML_KEYS:
        return f"+{formatted}"
    return formatted


def build_overrides(env_config, actor_variant, max_kl, seed):
    overrides = [
        hydra_override("env_id", env_config["env_id"]),
        hydra_override("seed", seed),
        hydra_override("discrete_action", actor_variant["discrete_action"]),
        hydra_override("use_residual_blocks", actor_variant["use_residual_blocks"]),
        hydra_override("max_kl", max_kl),
        hydra_override("wandb_group", "TRPO"),
    ]
    for key, value in env_config.items():
        if key in {"env_id", "extra"}:
            continue
        overrides.append(hydra_override(key, value))
    if "extra" in env_config:
        for key, value in parse_extra_args(env_config["extra"]).items():
            overrides.append(hydra_override(key, value))
    for key, value in GLOBAL_COMMON_SETTINGS.items():
        overrides.append(hydra_override(key, value))
    return overrides


def build_all_tasks():
    """Build list of all tasks as dicts."""
    tasks = []
    for env_config in MANISKILL_STATE_ENV_CONFIGS:
        for actor_variant in ACTOR_VARIANTS:
            for max_kl in MAX_KL_VALUES:
                for seed in SEEDS:
                    tasks.append({
                        "env_id": env_config["env_id"],
                        "env_config": env_config,
                        "variant_name": actor_variant["name"],
                        "actor_variant": actor_variant,
                        "max_kl": max_kl,
                        "seed": seed,
                        "key": task_key(env_config["env_id"],
                                        actor_variant["name"], max_kl, seed),
                    })
    return tasks


def query_wandb_finished_runs():
    """Query wandb for finished runs and return set of task keys."""
    finished = set()
    try:
        import wandb
        api = wandb.Api()
        runs = api.runs(
            f"{WANDB_ENTITY}/{WANDB_PROJECT}",
            filters={"state": "finished"},
        )
        for run in runs:
            cfg = run.config
            env_id = cfg.get("env_id", "")
            seed = cfg.get("seed", "")
            discrete = cfg.get("discrete_action", None)
            residual = cfg.get("use_residual_blocks", None)
            max_kl = cfg.get("max_kl", "")

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

            key = task_key(env_id, variant, f"kl{max_kl}" if not str(max_kl).startswith("kl") else max_kl, seed)
            # Normalize key format
            key = task_key(env_id, variant, max_kl, seed)
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
        task["env_config"], task["actor_variant"],
        task["max_kl"], task["seed"],
    )
    cmd = [
        sys.executable,
        "train_trpo.py",
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
    parser = argparse.ArgumentParser(description="Dynamic ManiSkill TRPO scheduler")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check wandb and print remaining tasks without running")
    parser.add_argument("--no-wandb-check", action="store_true",
                        help="Skip wandb check, run all tasks")
    cli_args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))

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
    print(f"VRAM safety margin: {VRAM_SAFETY_MB} MB kept free (target ~80% util)")
    print()

    if cli_args.dry_run:
        print("=" * 70)
        print("DRY RUN — remaining tasks:")
        print("=" * 70)
        by_env = defaultdict(list)
        for t in pending_tasks:
            by_env[t["env_id"]].append(t)
        for env_id, tasks in by_env.items():
            print(f"\n  {env_id} ({len(tasks)} runs):")
            for t in tasks:
                print(f"    {t['variant_name']}:kl{t['max_kl']}:seed{t['seed']}")
        print(f"\nTotal remaining: {len(pending_tasks)} experiments")

        # Show current GPU state
        print("\n" + "=" * 70)
        print("Current GPU state:")
        print("=" * 70)
        free_vram = get_gpu_free_vram()
        for gpu_id in GPU_IDS:
            free = free_vram.get(gpu_id, 0)
            total = 97887
            used = total - free
            pct = used / total * 100
            can_fit = max(0, (free - VRAM_SAFETY_MB) // 10_000)
            print(f"  GPU {gpu_id}: {used:,} / {total:,} MB "
                  f"({pct:.0f}% used), free {free:,} MB "
                  f"→ can fit ~{can_fit} ManiSkill tasks now")
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
                # Only launch if we have enough headroom
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
