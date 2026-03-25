#!/usr/bin/env python3
"""Script to run PPO experiments on multiple Gymnasium environments."""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

# Environment configurations
ENV_CONFIGS = [
    {"env_id": "Humanoid-v4"},
]

# Seeds to run for each environment
SEEDS = [0, 1, 2, 3, 4]

# Bin sweep
NUM_BINS = [2, 3, 5]

# GPU and concurrency settings
GPU_ID = 0                  # Physical GPU to use
MAX_CONCURRENT_TASKS = 1    # One job at a time on this card

# Common settings for all environments
COMMON_SETTINGS = {
    "total_timesteps": 5_000_000,
    "num_envs": 16,
    "num_steps": 1024,
    "critic_width": 64,
    "critic_depth": 2,
    "wandb_project_name": "rnd-bin-sweep",
    "wandb_entity": "jif005-ucsd",
    "track": True,
}

# Base config file
BASE_CONFIG = "configs/gym.yaml"


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


def run_experiment(env_id: str, seed: int, num_bins: int, common_settings: dict, gpu_id: int = None):
    """Run a single experiment for the given environment and seed."""
    task_name = f"{env_id}_bins{num_bins}_seed{seed}"
    gpu_str = f" (GPU {gpu_id})" if gpu_id is not None else ""
    print("=" * 50)
    print(f"Running experiment for: {task_name}{gpu_str}")
    print("=" * 50)
    
    # Build overrides list
    overrides = [f"env_id={env_id}", f"seed={seed}", f"num_bins={num_bins}"]
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
    total_experiments = len(ENV_CONFIGS) * len(NUM_BINS) * len(SEEDS)

    print("Starting batch experiments for Gymnasium environments")
    print(f"Environments: {[c['env_id'] for c in ENV_CONFIGS]}")
    print(f"Bins: {NUM_BINS}, Seeds: {SEEDS}")
    print(f"Total experiments: {total_experiments}, GPU: {GPU_ID}, Max concurrent: {MAX_CONCURRENT_TASKS}")
    print()

    # Create task list
    tasks = []
    for env_config in ENV_CONFIGS:
        for num_bins in NUM_BINS:
            for seed in SEEDS:
                tasks.append((env_config["env_id"], seed, num_bins))

    # Run experiments with concurrency control
    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS) as executor:
        # Submit all tasks
        future_to_task = {}
        for env_id, seed, num_bins in tasks:
            future = executor.submit(run_experiment, env_id, seed, num_bins, COMMON_SETTINGS, GPU_ID)
            future_to_task[future] = (env_id, seed, GPU_ID)
        
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

    for env_config in ENV_CONFIGS:
        env_id = env_config["env_id"]
        env_results = [(s, success) for eid, s, success in results if eid == env_id]
        print(f"\n{env_id}:")
        for seed, success in env_results:
            status = "SUCCESS" if success else "FAILED"
            print(f"  seed={seed}: {status}")
    
    # Count successes
    num_success = sum(1 for _, _, success in results if success)
    total_time = time.time() - start_time
    print(f"\nTotal: {num_success}/{total_experiments} experiments completed successfully")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")


if __name__ == "__main__":
    main()
