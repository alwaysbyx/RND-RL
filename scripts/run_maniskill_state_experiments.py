#!/usr/bin/env python3
"""Script to run PPO experiments on multiple ManiSkill state environments."""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

# Environment configurations
ENV_CONFIGS = [
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

# Seeds to run for each environment
SEEDS = [0, 1, 2, 3, 4]

# GPU and concurrency settings
NUM_GPUS = 2  # Number of GPUs to use (set to None to use all available)
MAX_CONCURRENT_TASKS = 4  # Maximum number of tasks to run simultaneously

# Common settings for all environments
COMMON_SETTINGS = {
    "wandb_project_name": "maniskill_state_final",
    "wandb_entity": "jif005-ucsd",
    "track": True,
}

# Base config file
BASE_CONFIG = "configs/maniskill-state.yaml"


def get_available_gpus():
    """Get list of available GPU IDs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--list-gpus"],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_count = len(result.stdout.strip().split('\n'))
        return list(range(gpu_count))
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: nvidia-smi not found, assuming no GPUs available")
        return []


def parse_extra_args(extra_str: str):
    """Parse extra command-line arguments string into key-value pairs.
    
    Supports both formats:
    - --key=value
    - --key value
    """
    if not extra_str or not extra_str.strip():
        return {}
    
    overrides = {}
    parts = extra_str.strip().split()
    i = 0
    while i < len(parts):
        arg = parts[i]
        if arg.startswith("--"):
            # Handle --key=value format
            if "=" in arg:
                key, value = arg[2:].split("=", 1)  # Remove "--" and split on first "="
            else:
                key = arg[2:]  # Remove "--"
                # Check if next part is a value (not starting with --)
                if i + 1 < len(parts) and not parts[i + 1].startswith("--"):
                    value = parts[i + 1]
                    i += 1  # Skip the value in next iteration
                else:
                    # Boolean flag
                    overrides[key.replace("-", "_")] = True
                    i += 1
                    continue
            
            # Convert to appropriate type
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string
            overrides[key.replace("-", "_")] = value
        i += 1
    return overrides


def run_experiment(env_config: dict, seed: int, common_settings: dict, gpu_id: int = None):
    """Run a single experiment for the given environment and seed."""
    env_id = env_config["env_id"]
    task_name = f"{env_id}_seed{seed}"
    gpu_str = f" (GPU {gpu_id})" if gpu_id is not None else ""
    print("=" * 50)
    print(f"Running experiment for: {task_name}{gpu_str}")
    print("=" * 50)
    
    # Build overrides list
    overrides = [f"env_id={env_id}", f"seed={seed}"]
    
    # Add environment-specific settings
    if "num_envs" in env_config:
        overrides.append(f"num_envs={env_config['num_envs']}")
    if "total_timesteps" in env_config:
        overrides.append(f"total_timesteps={env_config['total_timesteps']}")
    
    # Parse and add extra arguments
    if "extra" in env_config:
        extra_overrides = parse_extra_args(env_config["extra"])
        for key, value in extra_overrides.items():
            if isinstance(value, bool):
                overrides.append(f"{key}={str(value).lower()}")
            else:
                overrides.append(f"{key}={value}")
    
    # Add common settings
    for key, value in common_settings.items():
        if isinstance(value, bool):
            overrides.append(f"{key}={str(value).lower()}")
        else:
            overrides.append(f"{key}={value}")
    
    # Get project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # Build command
    cmd = [
        sys.executable,
        "train.py",
        "--config", BASE_CONFIG,
        "--config_overrides"
    ]
    # Add overrides as separate arguments
    cmd.extend(overrides)
    
    # Set CUDA_VISIBLE_DEVICES if GPU is specified
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"Command: {' '.join(cmd)}")
    if gpu_id is not None:
        print(f"Using GPU: {gpu_id}")
    print()
    
    # Run the experiment
    try:
        result = subprocess.run(
            cmd,
            check=True,
            cwd=project_dir,
            env=env
        )
        print(f"\n✓ Completed experiment for: {task_name}{gpu_str}\n")
        return (env_id, seed, True)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed experiment for: {task_name}{gpu_str}")
        print(f"Error: {e}\n")
        return (env_id, seed, False)


def main():
    """Run all experiments with GPU and concurrency management."""
    total_experiments = len(ENV_CONFIGS) * len(SEEDS)
    
    # Get available GPUs
    available_gpus = get_available_gpus()
    if NUM_GPUS is not None:
        available_gpus = available_gpus[:NUM_GPUS]
    
    print("Starting batch experiments for ManiSkill state environments")
    print(f"Total environments: {len(ENV_CONFIGS)}")
    print(f"Seeds per environment: {len(SEEDS)}")
    print(f"Total experiments: {total_experiments}")
    print(f"Available GPUs: {len(available_gpus)} ({available_gpus if available_gpus else 'None'})")
    print(f"Max concurrent tasks: {MAX_CONCURRENT_TASKS}")
    print()
    
    # Create task queue
    tasks = []
    for env_config in ENV_CONFIGS:
        env_id = env_config["env_id"]
        for seed in SEEDS:
            tasks.append((env_config, seed))
    
    # GPU assignment queue (round-robin)
    gpu_queue = Queue()
    if available_gpus:
        # Fill queue with GPU IDs in round-robin fashion
        for i in range(len(tasks)):
            gpu_queue.put(available_gpus[i % len(available_gpus)])
    else:
        # No GPUs available, use None for all tasks
        for _ in range(len(tasks)):
            gpu_queue.put(None)
    
    # Run experiments with concurrency control
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS) as executor:
        # Submit all tasks
        future_to_task = {}
        for env_config, seed in tasks:
            gpu_id = gpu_queue.get()
            future = executor.submit(run_experiment, env_config, seed, COMMON_SETTINGS, gpu_id)
            future_to_task[future] = (env_config["env_id"], seed, gpu_id)
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_task):
            env_id, seed, gpu_id = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / completed if completed > 0 else 0
                remaining = total_experiments - completed
                eta = avg_time * remaining if remaining > 0 else 0
                print(f"Progress: {completed}/{total_experiments} completed "
                      f"(ETA: {eta/60:.1f} minutes)")
            except Exception as e:
                print(f"Exception for {env_id} seed={seed}: {e}")
                results.append((env_id, seed, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("Experiment Summary")
    print("=" * 50)
    
    # Group by environment
    for env_config in ENV_CONFIGS:
        env_id = env_config["env_id"]
        env_results = [(s, success) for eid, s, success in results if eid == env_id]
        print(f"\n{env_id}:")
        for seed, success in env_results:
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"  seed={seed}: {status}")
    
    # Count successes
    num_success = sum(1 for _, _, success in results if success)
    total_time = time.time() - start_time
    print(f"\nTotal: {num_success}/{total_experiments} experiments completed successfully")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")


if __name__ == "__main__":
    main()
