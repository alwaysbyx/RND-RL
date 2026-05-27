#!/usr/bin/env python3
"""Run PPO experiments across Gym and ManiSkill-state environments."""
import os
import subprocess
import sys
import time
from collections import defaultdict

# Gym environment configurations
GYM_ENV_CONFIGS = [
    # {"env_id": "HalfCheetah-v4"},
    # {"env_id": "Hopper-v4"},
    # {"env_id": "Ant-v4"},
    # {"env_id": "Walker2d-v4"},
    {"env_id": "Humanoid-v4"},
]

# ManiSkill state environment configurations
MANISKILL_STATE_ENV_CONFIGS = [
    # {"env_id": "PushCube-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=4"},
    # {"env_id": "PickCube-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=4"},
    # {"env_id": "PickCubeSO100-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=8"},
    # {"env_id": "PushT-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=16 --gamma=0.99 --num-eval-steps=100"},
    {"env_id": "StackCube-v1", "num_envs": 4096, "total_timesteps": 200_000_000, "extra": ""},
    # {"env_id": "RollBall-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=16 --num-eval-steps=80 --gamma=0.95"},
    # {"env_id": "PullCube-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=4"},
    # {"env_id": "PokeCube-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=4"},
    # {"env_id": "LiftPegUpright-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=4"},
    # {"env_id": "PickSingleYCB-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=16"},
]

# Methods to evaluate
METHOD_CONFIGS = [
    {"name": "discrete_true_residual_true", "discrete_action": True, "use_residual_blocks": True},
    {"name": "discrete_true_residual_false", "discrete_action": True, "use_residual_blocks": False},
    {"name": "discrete_false_residual_true", "discrete_action": False, "use_residual_blocks": True},
    {"name": "discrete_false_residual_false", "discrete_action": False, "use_residual_blocks": False},
]

# Seeds to run for each environment and method
SEEDS = [0, 1, 2, 3, 4]

# GPU and scheduling settings
GPUS = [4, 5, 6, 7]                # Which GPUs to use
UTILIZATION_THRESHOLD = 80          # Launch new task only if GPU util < this %
CHECK_INTERVAL = 60                 # Seconds between scheduling checks

# One shared W&B project for all runs
GLOBAL_COMMON_SETTINGS = {
    "wandb_project_name": "ablation_extend",
    "wandb_entity": None,  # TODO: set to your wandb entity
    "track": True,
}

TASK_FAMILIES = [
    {
        "name": "gym",
        "base_config": "configs/gym.yaml",
        "env_configs": GYM_ENV_CONFIGS,
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
        "env_configs": MANISKILL_STATE_ENV_CONFIGS,
        "common_settings": {},
    },
]


def get_gpu_utilization(gpu_ids):
    """Query GPU utilization (%) for the given GPU IDs. Returns {gpu_id: util%}."""
    try:
        id_str = ",".join(str(g) for g in gpu_ids)
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={id_str}",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        utils = {}
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            gpu_id = int(parts[0])
            mem_used = int(parts[2])
            mem_total = int(parts[3])
            mem_util = mem_used * 100 // mem_total if mem_total > 0 else 0
            utils[gpu_id] = mem_util
        return utils
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: nvidia-smi query failed: {e}")
        return {g: 0 for g in gpu_ids}


def parse_extra_args(extra_str: str):
    """Parse extra command-line arguments into Hydra overrides."""
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
    """Format a key-value pair for Hydra command-line override."""
    if isinstance(value, bool):
        return f"{key}={str(value).lower()}"
    return f"{key}={value}"


def build_overrides(task_family: dict, env_config: dict, method_config: dict, seed: int):
    """Build Hydra overrides for a single run."""
    overrides = [f"env_id={env_config['env_id']}", f"seed={seed}"]

    # Method settings
    overrides.append(format_override("discrete_action", method_config["discrete_action"]))
    overrides.append(format_override("use_residual_blocks", method_config["use_residual_blocks"]))

    # Environment-specific settings
    for key, value in env_config.items():
        if key in {"env_id", "extra"}:
            continue
        overrides.append(format_override(key, value))

    # Environment-specific extra argument string
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


def launch_experiment(task_family: dict, env_config: dict, method_config: dict, seed: int, gpu_id: int):
    """Launch a single experiment as a non-blocking subprocess. Returns (task_name, Popen, gpu_id)."""
    family_name = task_family["name"]
    env_id = env_config["env_id"]
    method_name = method_config["name"]
    task_name = f"{family_name}:{env_id}:{method_name}:seed{seed}"

    overrides = build_overrides(task_family, env_config, method_config, seed)
    cmd = [
        sys.executable,
        "train.py",
        "--config",
        task_family["base_config"],
        "--config_overrides",
    ]
    cmd.extend(overrides)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_dir = os.path.join(project_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{task_name.replace(':', '_')}.log")

    print(f"[LAUNCH] {task_name} -> GPU {gpu_id}  (log: {log_file})")
    fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, cwd=project_dir, env=env, stdout=fh, stderr=subprocess.STDOUT)

    return task_name, proc, gpu_id, fh


def build_tasks():
    """Build all task tuples."""
    tasks = []
    for task_family in TASK_FAMILIES:
        for env_config in task_family["env_configs"]:
            for method_config in METHOD_CONFIGS:
                for seed in SEEDS:
                    tasks.append((task_family, env_config, method_config, seed))
    return tasks


def main():
    """Run all experiments with dynamic GPU scheduling based on utilization."""
    tasks = build_tasks()
    total = len(tasks)
    pending = list(tasks)  # Tasks waiting to be launched

    print("=" * 80)
    print("Dynamic GPU Scheduler")
    print("=" * 80)
    print(f"Task families: {[f['name'] for f in TASK_FAMILIES]}")
    print(f"Methods: {[m['name'] for m in METHOD_CONFIGS]}")
    print(f"Seeds per env/method: {len(SEEDS)}")
    print(f"Total experiments: {total}")
    print(f"GPUs: {GPUS}")
    print(f"Utilization threshold: {UTILIZATION_THRESHOLD}%")
    print(f"Check interval: {CHECK_INTERVAL}s")
    print(f"W&B project: {GLOBAL_COMMON_SETTINGS['wandb_project_name']}")
    print()

    # running: list of (task_name, Popen, gpu_id, file_handle)
    running = []
    results = []  # (family, env, method, seed, success)
    start_time = time.time()

    while pending or running:
        # 1) Check for finished processes
        still_running = []
        for task_name, proc, gpu_id, fh in running:
            ret = proc.poll()
            if ret is not None:
                fh.close()
                success = ret == 0
                # Parse task_name back to components
                parts = task_name.split(":")
                results.append((parts[0], parts[1], parts[2], parts[3], success))
                status = "SUCCESS" if success else f"FAILED (exit {ret})"
                elapsed = time.time() - start_time
                print(f"[{status}] {task_name} | GPU {gpu_id} | "
                      f"Progress: {len(results)}/{total} | "
                      f"Elapsed: {elapsed/60:.1f}min")
            else:
                still_running.append((task_name, proc, gpu_id, fh))
        running = still_running

        # 2) If there are pending tasks, check GPU utilization and launch
        if pending:
            utils = get_gpu_utilization(GPUS)
            # Find GPUs below threshold
            free_gpus = [g for g in GPUS if utils.get(g, 0) < UTILIZATION_THRESHOLD]

            if free_gpus:
                # Count how many tasks are already running on each free GPU
                running_per_gpu = defaultdict(int)
                for _, _, gid, _ in running:
                    running_per_gpu[gid] += 1

                # Pick the GPU with fewest running tasks (load balance)
                free_gpus.sort(key=lambda g: running_per_gpu[g])

                # Launch one task on the least-loaded free GPU
                gpu_id = free_gpus[0]
                task_family, env_config, method_config, seed = pending.pop(0)
                info = launch_experiment(task_family, env_config, method_config, seed, gpu_id)
                running.append(info)
                print(f"  [STATUS] Running: {len(running)} | Pending: {len(pending)} | "
                      f"GPU utils: {{{', '.join(f'{g}: {utils.get(g,0)}%' for g in GPUS)}}}")

                # If multiple GPUs are free, try to launch more without waiting
                # (but re-query utilization next cycle)
                continue  # Skip sleep, re-check immediately
            else:
                print(f"  [WAIT] All GPUs busy (utils: {{{', '.join(f'{g}: {utils.get(g,0)}%' for g in GPUS)}}})"
                      f" | Running: {len(running)} | Pending: {len(pending)}")

        # 3) Sleep before next check
        if pending or running:
            time.sleep(CHECK_INTERVAL)

    # Summary
    print("\n" + "=" * 80)
    print("Experiment Summary")
    print("=" * 80)

    grouped = defaultdict(list)
    for family_name, env_id, method_name, seed_str, success in results:
        grouped[(family_name, env_id, method_name)].append((seed_str, success))

    for task_family in TASK_FAMILIES:
        family_name = task_family["name"]
        print(f"\n[{family_name}]")
        for env_config in task_family["env_configs"]:
            env_id = env_config["env_id"]
            for method_config in METHOD_CONFIGS:
                method_name = method_config["name"]
                key = (family_name, env_id, method_name)
                seed_results = grouped.get(key, [])
                success_count = sum(1 for _, s in seed_results if s)
                print(f"  {env_id} | {method_name}: {success_count}/{len(SEEDS)} successful")

    num_success = sum(1 for *_, s in results if s)
    total_time = time.time() - start_time
    print(f"\nTotal: {num_success}/{total} experiments completed successfully")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")

    failed = [r for r in results if not r[4]]
    if failed:
        print("\nFailed runs:")
        for f, e, m, s, _ in failed:
            print(f"  {f}:{e}:{m}:{s}")


if __name__ == "__main__":
    main()
