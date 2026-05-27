#!/usr/bin/env python3
"""Dynamic GPU scheduler for ManiSkill RGB PPO experiments.

Monitors free VRAM on GPUs and dynamically schedules ManiSkill RGB
experiments using train.py (PPO). Before launching, queries the
ppo_rgb wandb project to skip any (env_id, method, seed) tuple that
already has a finished run. Only re-runs the unfinished ones.

Experiment matrix:
  1 algo * 5 RGB envs * 4 methods * 5 seeds = 100 runs (minus what wandb already has finished)
  - PPO (train.py)  -> wandb project: ppo_rgb

Usage:
  python scripts/run_rgb_dynamic.py                  # run, skipping finished wandb runs
  python scripts/run_rgb_dynamic.py --dry-run        # preview pending tasks
  python scripts/run_rgb_dynamic.py --no-wandb-check # skip wandb query, run all 100
"""
import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict

# ---------------------------------------------------------------------------
# Environment configurations (matches run_maniskill_rgb_experiments.py)
# ---------------------------------------------------------------------------

ENV_CONFIGS = [
    {"env_id": "PushCube-v1", "num_envs": 1024, "total_timesteps": 50_000_000, "extra": ""},
    {"env_id": "PickCube-v1", "num_envs": 1024, "total_timesteps": 50_000_000, "extra": "--num-steps=16"},
    {"env_id": "PokeCube-v1", "num_envs": 1024, "total_timesteps": 50_000_000, "extra": "--num-steps=16"},
    {"env_id": "PushT-v1", "num_envs": 1024, "total_timesteps": 50_000_000, "extra": "--num-steps=16 --gamma=0.99 --num-eval-steps=100"},
    {"env_id": "StackCube-v1", "num_envs": 1024, "total_timesteps": 50_000_000, "extra": ""},
]

SEEDS = [0, 1, 2, 3, 4]

# Methods to evaluate
METHOD_CONFIGS = [
    {"name": "discrete_true_residual_true", "discrete_action": True, "use_residual_blocks": True},
    {"name": "discrete_true_residual_false", "discrete_action": True, "use_residual_blocks": False},
    {"name": "discrete_false_residual_true", "discrete_action": False, "use_residual_blocks": True},
    {"name": "discrete_false_residual_false", "discrete_action": False, "use_residual_blocks": False},
]

# Algorithms: each maps to a training script + wandb project
ALGO_CONFIGS = [
    {"name": "ppo", "script": "train.py", "wandb_project_name": "ppo_rgb"},
]

# ---------------------------------------------------------------------------
# GPU settings
# ---------------------------------------------------------------------------

GPU_IDS = [4, 5, 6, 7]
# Minimum free VRAM (MB) required before launching a new task on a GPU.
# RGB tasks with 1024 envs + image encoder typically use ~8-15 GB each.
VRAM_THRESHOLD_MB = 20_000
POLL_INTERVAL = 60          # Seconds between scheduling checks when all GPUs are full
LAUNCH_COOLDOWN = 60        # Seconds to wait after a launch pass so VRAM can settle

# ---------------------------------------------------------------------------
# Common settings (wandb_project_name comes from each algo_config)
# ---------------------------------------------------------------------------

COMMON_SETTINGS = {
    "wandb_entity": None,  # TODO: set to your wandb entity
    "track": True,
}

BASE_CONFIG = "configs/maniskill-rgb.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def task_key(algo_name, env_id, method_name, seed):
    """Unique string identifier for a task."""
    return f"{algo_name}|{env_id}|{method_name}|seed{seed}"


def parse_extra_args(extra_str: str):
    """Parse extra command-line arguments into key-value overrides."""
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


def build_overrides(env_config, method_config, seed, algo_config):
    overrides = [f"env_id={env_config['env_id']}", f"seed={seed}"]

    # Method settings (discrete_action + use_residual_blocks)
    overrides.append(format_override("discrete_action", method_config["discrete_action"]))
    overrides.append(format_override("use_residual_blocks", method_config["use_residual_blocks"]))

    if "num_envs" in env_config:
        overrides.append(f"num_envs={env_config['num_envs']}")
    if "total_timesteps" in env_config:
        overrides.append(f"total_timesteps={env_config['total_timesteps']}")

    if "extra" in env_config:
        extra_overrides = parse_extra_args(env_config["extra"])
        for key, value in extra_overrides.items():
            overrides.append(format_override(key, value))

    # Per-algorithm wandb project
    overrides.append(format_override("wandb_project_name", algo_config["wandb_project_name"]))

    for key, value in COMMON_SETTINGS.items():
        overrides.append(format_override(key, value))

    return overrides


def method_name_from_flags(discrete, residual):
    """Map (discrete_action, use_residual_blocks) -> METHOD_CONFIGS name."""
    if discrete and residual:
        return "discrete_true_residual_true"
    if discrete and not residual:
        return "discrete_true_residual_false"
    if not discrete and residual:
        return "discrete_false_residual_true"
    return "discrete_false_residual_false"


def query_wandb_finished_runs(algo_config):
    """Query wandb and return set of finished task keys for the given algo's project.

    A run counts as 'finished' only if its wandb state == 'finished'.
    Returns a set of task_key strings matching this script's build_all_tasks output.
    """
    finished = set()
    project = algo_config["wandb_project_name"]
    entity = COMMON_SETTINGS["wandb_entity"]
    try:
        import wandb
        api = wandb.Api(timeout=60)
        runs = api.runs(f"{entity}/{project}", filters={"state": "finished"})
        total = 0
        for run in runs:
            total += 1
            # The list-view's run.config is shallow/empty; fetch the full run.
            cfg = api.run(f"{entity}/{project}/{run.id}").config
            env_id = cfg.get("env_id", "")
            seed = cfg.get("seed", None)
            discrete = cfg.get("discrete_action", None)
            residual = cfg.get("use_residual_blocks", None)
            if seed is None or discrete is None or residual is None or not env_id:
                continue
            method_name = method_name_from_flags(discrete, residual)
            key = task_key(algo_config["name"], env_id, method_name, seed)
            finished.add(key)
        print(f"[wandb] Project {entity}/{project}: {total} finished runs, "
              f"{len(finished)} matched current matrix")
    except Exception as e:
        print(f"[wandb] Warning: could not query wandb: {e}")
        print("[wandb] Proceeding without skipping - all tasks will be queued.")
    return finished


def build_all_tasks():
    """Build list of all tasks as dicts."""
    tasks = []
    for algo_config in ALGO_CONFIGS:
        for env_config in ENV_CONFIGS:
            for method_config in METHOD_CONFIGS:
                for seed in SEEDS:
                    tasks.append({
                        "algo_name": algo_config["name"],
                        "algo_config": algo_config,
                        "env_id": env_config["env_id"],
                        "env_config": env_config,
                        "method_name": method_config["name"],
                        "method_config": method_config,
                        "seed": seed,
                        "key": task_key(algo_config["name"], env_config["env_id"],
                                        method_config["name"], seed),
                    })
    return tasks


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
    """Launch a single experiment as a subprocess. Returns (Popen, fh, log_file)."""
    overrides = build_overrides(
        task["env_config"], task["method_config"], task["seed"], task["algo_config"]
    )
    cmd = [
        sys.executable,
        task["algo_config"]["script"],   # train.py for PPO, train_trpo.py for TRPO
        "--config", BASE_CONFIG,
        "--config_overrides",
    ] + overrides

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_dir = os.path.join(project_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"rgb_{task['key'].replace('|', '_')}.log")
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

    parser = argparse.ArgumentParser(
        description="Dynamic ManiSkill RGB PPO scheduler (skips finished wandb runs)"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print remaining tasks and current GPU state without running")
    parser.add_argument("--no-wandb-check", action="store_true",
                        help="Skip wandb finished-run query; queue all tasks")
    parser.add_argument("--vram-threshold-mb", type=int, default=VRAM_THRESHOLD_MB,
                        help=f"Min free VRAM (MB) needed to launch a task (default: {VRAM_THRESHOLD_MB})")
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL,
                        help=f"Seconds between scheduling checks (default: {POLL_INTERVAL})")
    parser.add_argument("--launch-cooldown", type=int, default=LAUNCH_COOLDOWN,
                        help=f"Seconds to wait after launching before checking VRAM again (default: {LAUNCH_COOLDOWN})")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs to use (default: %s)" % ",".join(str(g) for g in GPU_IDS))
    cli_args = parser.parse_args()

    if cli_args.gpus:
        GPU_IDS = [int(g) for g in cli_args.gpus.split(",")]
    VRAM_THRESHOLD_MB = cli_args.vram_threshold_mb
    POLL_INTERVAL = cli_args.poll_interval
    LAUNCH_COOLDOWN = cli_args.launch_cooldown

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    full_matrix = build_all_tasks()

    # Filter out tasks already finished in wandb (unless --no-wandb-check)
    if cli_args.no_wandb_check:
        all_tasks = list(full_matrix)
        skipped = 0
    else:
        finished_keys = set()
        for algo_config in ALGO_CONFIGS:
            finished_keys |= query_wandb_finished_runs(algo_config)
        all_tasks = [t for t in full_matrix if t["key"] not in finished_keys]
        skipped = len(full_matrix) - len(all_tasks)

    print("=" * 70)
    print("Dynamic ManiSkill RGB PPO Scheduler")
    print("=" * 70)
    print(f"Full matrix: {len(full_matrix)} "
          f"({len(ALGO_CONFIGS)} algos * {len(ENV_CONFIGS)} envs * "
          f"{len(METHOD_CONFIGS)} methods * {len(SEEDS)} seeds)")
    print(f"Skipped (finished in wandb): {skipped}")
    print(f"Pending experiments: {len(all_tasks)}")
    print(f"Algos: {[(a['name'], a['script'], a['wandb_project_name']) for a in ALGO_CONFIGS]}")
    print(f"Methods: {[m['name'] for m in METHOD_CONFIGS]}")
    print(f"GPUs: {GPU_IDS}")
    print(f"VRAM threshold to launch: {VRAM_THRESHOLD_MB:,} MB free")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print(f"Launch cooldown: {LAUNCH_COOLDOWN}s")
    print()

    if cli_args.dry_run:
        print("=" * 70)
        print("DRY RUN - remaining tasks:")
        print("=" * 70)
        by_algo_env = defaultdict(list)
        for t in all_tasks:
            by_algo_env[(t["algo_name"], t["env_id"])].append(t)
        for (algo_name, env_id), tasks in sorted(by_algo_env.items()):
            print(f"\n  [{algo_name}] {env_id} ({len(tasks)} runs):")
            by_method = defaultdict(list)
            for t in tasks:
                by_method[t["method_name"]].append(t)
            for method_name, mtasks in by_method.items():
                seeds_str = ", ".join(f"seed{t['seed']}" for t in mtasks)
                print(f"    {method_name}: {seeds_str}")

        print("\n" + "=" * 70)
        print("Current GPU state:")
        print("=" * 70)
        free_vram = get_gpu_free_vram()
        for gpu_id in GPU_IDS:
            free = free_vram.get(gpu_id, 0)
            can_fit = max(0, free // VRAM_THRESHOLD_MB)
            print(f"  GPU {gpu_id}: free {free:,} MB "
                  f"-> can fit ~{can_fit} tasks now")
        return

    # --- Dynamic scheduling loop ---
    task_queue = list(all_tasks)
    # {gpu_id: [(proc, task, file_handle), ...]}
    running = defaultdict(list)
    completed = 0
    failed = 0
    total = len(task_queue)
    start_time = time.time()

    print("=" * 70)
    print(f"Starting dynamic scheduler - {total} tasks to run")
    print("=" * 70)

    while task_queue or any(running[g] for g in GPU_IDS):
        # 1. Reap finished processes
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

        # 2. Round-robin across GPUs: in each inner pass, launch at most ONE
        #    task on every GPU that has enough free VRAM. Then sleep the
        #    cooldown so freshly-launched processes can allocate, and loop.
        launched_this_cycle = 0
        while task_queue:
            free_vram = get_gpu_free_vram()
            # Prefer the GPU with the fewest running tasks first (load balance)
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
