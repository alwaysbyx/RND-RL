#!/usr/bin/env python3
"""Run PPO experiments across Gym and ManiSkill-state environments."""
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

# Gym environment configurations
GYM_ENV_CONFIGS = [
    {"env_id": "HalfCheetah-v4"},
    {"env_id": "Hopper-v4"},
    {"env_id": "Ant-v4"},
    {"env_id": "Walker2d-v4"},
    {"env_id": "Humanoid-v4"},
]

# ManiSkill state environment configurations
MANISKILL_STATE_ENV_CONFIGS = [
    # {"env_id": "PushCube-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=4"},
    # {"env_id": "PickCube-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=4"},
    # {"env_id": "PickCubeSO100-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=8"},
    # {"env_id": "PushT-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=16 --gamma=0.99 --num-eval-steps=100"},
    # {"env_id": "StackCube-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": ""},
    # {"env_id": "RollBall-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=16 --num-eval-steps=80 --gamma=0.95"},
    # {"env_id": "PullCube-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=4"},
    # {"env_id": "PokeCube-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=4"},
    # {"env_id": "LiftPegUpright-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=4"},
    {"env_id": "PickSingleYCB-v1", "num_envs": 4096, "total_timesteps": 50_000_000, "extra": "--num-steps=16"},
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

# GPU and concurrency settings
NUM_GPUS = 1
MAX_CONCURRENT_TASKS = 5

# One shared W&B project for all runs
GLOBAL_COMMON_SETTINGS = {
    "wandb_project_name": "gym_maniskill_state_final",
    "wandb_entity": "jif005-ucsd",
    "track": True,
}

TASK_FAMILIES = [
    # {
    #     "name": "gym",
    #     "base_config": "configs/gym.yaml",
    #     "env_configs": GYM_ENV_CONFIGS,
    #     "common_settings": {
    #         "total_timesteps": 5_000_000,
    #         "num_envs": 16,
    #         "num_steps": 1024,
    #         "num_bins": 41,
    #         "critic_width": 64,
    #         "critic_depth": 2,
    #     },
    # },
    {
        "name": "maniskill_state",
        "base_config": "configs/maniskill-state.yaml",
        "env_configs": MANISKILL_STATE_ENV_CONFIGS,
        "common_settings": {},
    },
]


def get_available_gpus():
    """Get list of available GPU IDs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_count = len(result.stdout.strip().split("\n"))
        return list(range(gpu_count))
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: nvidia-smi not found, assuming no GPUs available")
        return []


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


def run_experiment(task_family: dict, env_config: dict, method_config: dict, seed: int, gpu_id: int = None):
    """Run a single experiment."""
    family_name = task_family["name"]
    env_id = env_config["env_id"]
    method_name = method_config["name"]
    task_name = f"{family_name}:{env_id}:{method_name}:seed{seed}"
    gpu_str = f" (GPU {gpu_id})" if gpu_id is not None else ""

    print("=" * 80)
    print(f"Running experiment: {task_name}{gpu_str}")
    print("=" * 80)

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
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"Command: {' '.join(cmd)}")
    if gpu_id is not None:
        print(f"Using GPU: {gpu_id}")
    print()

    try:
        subprocess.run(cmd, check=True, cwd=project_dir, env=env)
        print(f"SUCCESS: {task_name}{gpu_str}\n")
        return (family_name, env_id, method_name, seed, True)
    except subprocess.CalledProcessError as err:
        print(f"FAILED: {task_name}{gpu_str}")
        print(f"Error: {err}\n")
        return (family_name, env_id, method_name, seed, False)


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
    """Run all experiments with GPU and concurrency management."""
    tasks = build_tasks()
    total_experiments = len(tasks)

    available_gpus = get_available_gpus()
    if NUM_GPUS is not None:
        available_gpus = available_gpus[1:2]

    print("Starting combined Gym + ManiSkill-state experiments")
    print(f"Task families: {[family['name'] for family in TASK_FAMILIES]}")
    print(f"Methods: {[method['name'] for method in METHOD_CONFIGS]}")
    print(f"Seeds per env/method: {len(SEEDS)}")
    print(f"Total experiments: {total_experiments}")
    print(f"Shared W&B project: {GLOBAL_COMMON_SETTINGS['wandb_project_name']}")
    print(f"Available GPUs: {len(available_gpus)} ({available_gpus if available_gpus else 'None'})")
    print(f"Max concurrent tasks: {MAX_CONCURRENT_TASKS}")
    print()

    gpu_queue = Queue()
    if available_gpus:
        for i in range(len(tasks)):
            gpu_queue.put(available_gpus[i % len(available_gpus)])
    else:
        for _ in range(len(tasks)):
            gpu_queue.put(None)

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS) as executor:
        future_to_task = {}
        for task_family, env_config, method_config, seed in tasks:
            gpu_id = gpu_queue.get()
            future = executor.submit(
                run_experiment,
                task_family,
                env_config,
                method_config,
                seed,
                gpu_id,
            )
            future_to_task[future] = (task_family["name"], env_config["env_id"], method_config["name"], seed)

        completed = 0
        for future in as_completed(future_to_task):
            family_name, env_id, method_name, seed = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as err:  # pragma: no cover - defensive catch
                print(f"Exception for {family_name}:{env_id}:{method_name}:seed{seed}: {err}")
                results.append((family_name, env_id, method_name, seed, False))

            completed += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / completed if completed > 0 else 0
            remaining = total_experiments - completed
            eta = avg_time * remaining if remaining > 0 else 0
            print(f"Progress: {completed}/{total_experiments} completed (ETA: {eta/60:.1f} minutes)")

    print("\n" + "=" * 80)
    print("Experiment Summary")
    print("=" * 80)

    grouped = defaultdict(list)
    for family_name, env_id, method_name, seed, success in results:
        grouped[(family_name, env_id, method_name)].append((seed, success))

    for task_family in TASK_FAMILIES:
        family_name = task_family["name"]
        print(f"\n[{family_name}]")
        for env_config in task_family["env_configs"]:
            env_id = env_config["env_id"]
            for method_config in METHOD_CONFIGS:
                method_name = method_config["name"]
                key = (family_name, env_id, method_name)
                seed_results = grouped.get(key, [])
                success_count = sum(1 for _, success in seed_results if success)
                print(f"  {env_id} | {method_name}: {success_count}/{len(SEEDS)} successful")

    num_success = sum(1 for _, _, _, _, success in results if success)
    total_time = time.time() - start_time
    print(f"\nTotal: {num_success}/{total_experiments} experiments completed successfully")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")

    failed_runs = [(f, e, m, s) for f, e, m, s, ok in results if not ok]
    if failed_runs:
        print("\nFailed runs:")
        for family_name, env_id, method_name, seed in failed_runs:
            print(f"  {family_name}:{env_id}:{method_name}:seed{seed}")


if __name__ == "__main__":
    main()
