#!/usr/bin/env python3
"""Script to run PPO experiments on multiple ManiSkill state environments."""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Environment configurations
ENV_CONFIGS = [
    {"env_id": "StackCube-v1", "num_envs": 1024, "total_timesteps": 50_000_000, "extra": ""},
]

# Seeds to run for each environment
SEEDS = [0, 1, 2, 3, 4]

# Bin sweep
NUM_BINS = [2, 3, 5]

# GPU and concurrency settings
GPU_IDS = [0, 1]            # Physical GPUs to use (round-robin)
MAX_CONCURRENT_TASKS = 2    # One job per card

# Common settings for all environments
COMMON_SETTINGS = {
    "wandb_project_name": "rnd-bin-sweep",
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


def run_experiment(env_config: dict, seed: int, num_bins: int, common_settings: dict, gpu_id: int = None):
    """Run a single experiment for the given environment and seed."""
    env_id = env_config["env_id"]
    task_name = f"{env_id}_bins{num_bins}_seed{seed}"
    gpu_str = f" (GPU {gpu_id})" if gpu_id is not None else ""
    print("=" * 50)
    print(f"Running experiment for: {task_name}{gpu_str}")
    print("=" * 50)
    
    # Build overrides list
    overrides = [f"env_id={env_id}", f"seed={seed}", f"num_bins={num_bins}"]
    
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
    total_experiments = len(ENV_CONFIGS) * len(NUM_BINS) * len(SEEDS)

    print("Starting batch experiments for ManiSkill state environments")
    print(f"Environments: {[c['env_id'] for c in ENV_CONFIGS]}")
    print(f"Bins: {NUM_BINS}, Seeds: {SEEDS}")
    print(f"Total experiments: {total_experiments}, GPUs: {GPU_IDS}, Max concurrent: {MAX_CONCURRENT_TASKS}")
    print()

    # Create task list with round-robin GPU assignment
    tasks = []
    for i, (env_config, num_bins, seed) in enumerate(
        (e, b, s) for e in ENV_CONFIGS for b in NUM_BINS for s in SEEDS
    ):
        gpu_id = GPU_IDS[i % len(GPU_IDS)]
        tasks.append((env_config, seed, num_bins, gpu_id))

    # Run experiments with concurrency control
    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS) as executor:
        # Submit all tasks
        future_to_task = {}
        for env_config, seed, num_bins, gpu_id in tasks:
            future = executor.submit(run_experiment, env_config, seed, num_bins, COMMON_SETTINGS, gpu_id)
            future_to_task[future] = (env_config["env_id"], seed, gpu_id)

        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_task):
            env_id, seed, _ = future_to_task[future]
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

    num_success = sum(1 for _, _, success in results if success)
    total_time = time.time() - start_time
    print(f"\nTotal: {num_success}/{total_experiments} experiments completed successfully")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")


if __name__ == "__main__":
    main()
