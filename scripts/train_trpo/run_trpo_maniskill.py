#!/usr/bin/env python3
"""Run TRPO experiments across Gym and ManiSkill-state environments.

Experiment matrix:
  - 15 environments (5 Gym + 10 ManiSkill-state)
  - 4 actor variants (discrete/continuous × residual/MLP)
  - 2 max_kl values (0.01, 0.001)
  - 5 seeds
  = 600 total runs

Usage:
  python scripts/train_trpo/run_trpo_experiments.py                # run all
  python scripts/train_trpo/run_trpo_experiments.py --phase gym     # gym only
  python scripts/train_trpo/run_trpo_experiments.py --phase maniskill  # maniskill only
  python scripts/train_trpo/run_trpo_experiments.py --dry-run       # print tasks without running
"""
import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

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

# ---------------------------------------------------------------------------
# Actor variants & TRPO hyperparameter sweep
# ---------------------------------------------------------------------------

ACTOR_VARIANTS = [
    {"name": "discrete_residual", "discrete_action": True, "use_residual_blocks": True},
    {"name": "discrete_mlp", "discrete_action": True, "use_residual_blocks": False},
    {"name": "continuous_residual", "discrete_action": False, "use_residual_blocks": True},
    {"name": "continuous_mlp", "discrete_action": False, "use_residual_blocks": False},
]

MAX_KL_VALUES = [0.01]

SEEDS = [0, 1, 2, 3, 4]

# ---------------------------------------------------------------------------
# GPU & concurrency settings
# ---------------------------------------------------------------------------

NUM_GPUS = 4  # Number of GPUs to use
GPU_OFFSET = 4  # First GPU index (using GPUs 4,5,6,7)


# RTX PRO 6000 Blackwell: 96 GB VRAM per GPU
CONCURRENT_PER_GPU = {
    "gym": 20,            # ~2-3 GB each → 20 × 3 = 60 GB
    "maniskill_state": 12,  # ~10 GB each → 8 × 10 = 80 GB
}

# ---------------------------------------------------------------------------
# Task families
# ---------------------------------------------------------------------------

GLOBAL_COMMON_SETTINGS = {
    "wandb_project_name": "trpo_maniskill_experiments",
    "wandb_entity": "jif005-ucsd",
    "track": True,
}

TASK_FAMILIES = {
    # "gym": {
    #     "base_config": "configs/gym.yaml",
    #     "env_configs": GYM_ENV_CONFIGS,
    #     "common_settings": {
    #         "total_timesteps": 20_000_000,
    #         "num_envs": 16,
    #         "num_steps": 1024,
    #         "num_bins": 41,
    #         "critic_width": 64,
    #         "critic_depth": 2,
    #     },
    # },
    "maniskill_state": {
        "base_config": "configs/maniskill-state.yaml",
        "env_configs": MANISKILL_STATE_ENV_CONFIGS,
        "common_settings": {},
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    """Parse extra command-line arguments into Hydra overrides dict."""
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
    """Format a key-value pair as a Hydra override string."""
    if isinstance(value, bool):
        return f"{key}={str(value).lower()}"
    return f"{key}={value}"


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


def hydra_override(key, value):
    """Return a Hydra override, using +key=val for keys absent from YAML."""
    formatted = format_override(key, value)
    if key not in _YAML_KEYS:
        return f"+{formatted}"
    return formatted


def build_overrides(task_family: dict, env_config: dict, actor_variant: dict,
                    max_kl: float, seed: int):
    """Build the list of Hydra override strings for a single run."""
    overrides = [
        hydra_override("env_id", env_config["env_id"]),
        hydra_override("seed", seed),
        hydra_override("discrete_action", actor_variant["discrete_action"]),
        hydra_override("use_residual_blocks", actor_variant["use_residual_blocks"]),
        hydra_override("max_kl", max_kl),
        hydra_override("wandb_group", "TRPO"),
    ]

    # Environment-specific numeric overrides
    for key, value in env_config.items():
        if key in {"env_id", "extra"}:
            continue
        overrides.append(hydra_override(key, value))

    # Environment-specific extra-arg string (e.g. "--num-steps=4")
    if "extra" in env_config:
        for key, value in parse_extra_args(env_config["extra"]).items():
            overrides.append(hydra_override(key, value))

    # Task-family common settings
    for key, value in task_family["common_settings"].items():
        overrides.append(hydra_override(key, value))

    # Global shared settings
    for key, value in GLOBAL_COMMON_SETTINGS.items():
        overrides.append(hydra_override(key, value))

    return overrides


def build_tasks(families_to_run):
    """Build a list of (family_name, task_family, env_config, actor_variant, max_kl, seed) tuples."""
    tasks = []
    for family_name in families_to_run:
        task_family = TASK_FAMILIES[family_name]
        for env_config in task_family["env_configs"]:
            for actor_variant in ACTOR_VARIANTS:
                for max_kl in MAX_KL_VALUES:
                    for seed in SEEDS:
                        tasks.append((family_name, task_family, env_config,
                                      actor_variant, max_kl, seed))
    return tasks


def run_experiment(task_family, env_config, actor_variant, max_kl, seed,
                   gpu_id=None):
    """Run a single TRPO experiment."""
    env_id = env_config["env_id"]
    variant_name = actor_variant["name"]
    task_name = f"{env_id}:{variant_name}:kl{max_kl}:seed{seed}"
    gpu_str = f" (GPU {gpu_id})" if gpu_id is not None else ""

    print(f"[START] {task_name}{gpu_str}")

    overrides = build_overrides(task_family, env_config, actor_variant, max_kl, seed)

    # Resolve project directory (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))

    cmd = [
        sys.executable,
        "train_trpo.py",
        "--config",
        task_family["base_config"],
        "--config_overrides",
    ]
    cmd.extend(overrides)

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        subprocess.run(cmd, check=True, cwd=project_dir, env=env)
        print(f"[DONE]  {task_name}{gpu_str}")
        return (env_id, variant_name, max_kl, seed, True)
    except subprocess.CalledProcessError as err:
        print(f"[FAIL]  {task_name}{gpu_str} — {err}")
        return (env_id, variant_name, max_kl, seed, False)


def run_phase(family_name, tasks, available_gpus):
    """Run all tasks for one task family with appropriate concurrency."""
    per_gpu = CONCURRENT_PER_GPU.get(family_name, 3)
    max_concurrent = len(available_gpus) * per_gpu if available_gpus else 1
    total = len(tasks)

    print(f"\n{'=' * 70}")
    print(f"Phase: {family_name}  |  {total} runs  |  "
          f"{len(available_gpus)} GPUs × {per_gpu}/GPU = {max_concurrent} concurrent")
    print(f"{'=' * 70}\n")

    # Round-robin GPU assignment
    gpu_queue = Queue()
    if available_gpus:
        for i in range(total):
            gpu_queue.put(available_gpus[i % len(available_gpus)])
    else:
        for _ in range(total):
            gpu_queue.put(None)

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_task = {}
        for _, task_family, env_config, actor_variant, max_kl, seed in tasks:
            gpu_id = gpu_queue.get()
            future = executor.submit(
                run_experiment, task_family, env_config,
                actor_variant, max_kl, seed, gpu_id,
            )
            future_to_task[future] = (env_config["env_id"], actor_variant["name"],
                                      max_kl, seed)

        completed = 0
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                results.append(result)
            except Exception as err:
                info = future_to_task[future]
                print(f"[ERROR] {info}: {err}")
                results.append((*info, False))

            completed += 1
            elapsed = time.time() - start_time
            eta = elapsed / completed * (total - completed) if completed else 0
            print(f"  Progress: {completed}/{total}  "
                  f"(elapsed {elapsed/60:.0f}m, ETA {eta/60:.0f}m)")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run TRPO experiments")
    parser.add_argument("--phase", choices=["gym", "maniskill", "all"],
                        default="all", help="Which phase to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print task list without running")
    parser.add_argument("--num-gpus", type=int, default=NUM_GPUS,
                        help="Number of GPUs to use")
    parser.add_argument("--gpu-offset", type=int, default=GPU_OFFSET,
                        help="First GPU index")
    cli_args = parser.parse_args()

    families_to_run = {
        "all": ["gym", "maniskill_state"],
        "gym": ["gym"],
        "maniskill": ["maniskill_state"],
    }[cli_args.phase]

    all_tasks = build_tasks(families_to_run)

    # GPU setup
    available_gpus = get_available_gpus()
    available_gpus = available_gpus[cli_args.gpu_offset:
                                    cli_args.gpu_offset + cli_args.num_gpus]

    # Summary
    counts = defaultdict(int)
    for fn, *_ in all_tasks:
        counts[fn] += 1

    print("=" * 70)
    print("TRPO Experiment Batch")
    print("=" * 70)
    print(f"Phases:       {families_to_run}")
    print(f"Actor variants: {len(ACTOR_VARIANTS)}")
    print(f"max_kl values:  {MAX_KL_VALUES}")
    print(f"Seeds:          {SEEDS}")
    for fn, n in counts.items():
        envs = len(TASK_FAMILIES[fn]["env_configs"])
        per_gpu = CONCURRENT_PER_GPU.get(fn, 3)
        print(f"  {fn}: {envs} envs × {len(ACTOR_VARIANTS)} variants "
              f"× {len(MAX_KL_VALUES)} kl × {len(SEEDS)} seeds = {n} runs  "
              f"({per_gpu}/GPU)")
    print(f"Total runs:   {len(all_tasks)}")
    print(f"GPUs:         {available_gpus if available_gpus else 'None'}")
    print()

    if cli_args.dry_run:
        print("Dry-run task list:")
        for i, (fn, tf, ec, av, kl, s) in enumerate(all_tasks, 1):
            print(f"  {i:4d}. {fn}:{ec['env_id']}:{av['name']}:kl{kl}:seed{s}")
        print(f"\nTotal: {len(all_tasks)} tasks")
        return

    # Run each family as a separate phase
    all_results = []
    global_start = time.time()

    for family_name in families_to_run:
        phase_tasks = [(fn, tf, ec, av, kl, s)
                       for fn, tf, ec, av, kl, s in all_tasks
                       if fn == family_name]
        if phase_tasks:
            results = run_phase(family_name, phase_tasks, available_gpus)
            all_results.extend(results)

    # Final summary
    total_time = time.time() - global_start
    num_success = sum(1 for *_, ok in all_results if ok)
    print(f"\n{'=' * 70}")
    print("Final Summary")
    print(f"{'=' * 70}")
    print(f"Succeeded: {num_success}/{len(all_results)}")
    print(f"Wall time: {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")

    failed = [r for r in all_results if not r[-1]]
    if failed:
        print(f"\nFailed runs ({len(failed)}):")
        for env_id, variant, kl, seed, _ in failed:
            print(f"  {env_id}:{variant}:kl{kl}:seed{seed}")


if __name__ == "__main__":
    main()
