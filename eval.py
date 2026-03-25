"""Evaluation script for PPO agents."""
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import tyro
from hydra import initialize, compose
from omegaconf import OmegaConf

# Project imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rnd import Agent
from envs import make_gym_env, make_maniskill_state_env, make_maniskill_rgb_env


@dataclass
class EvalArgs:
    """Evaluation arguments."""
    config: Optional[str] = None
    """Path to YAML config file"""
    checkpoint: str = ""
    """Path to model checkpoint file"""
    eval_episodes: int = 10
    """Number of episodes to evaluate"""
    num_eval_envs: int = 8
    """Number of parallel evaluation environments"""
    num_eval_steps: int = 1000
    """Maximum number of steps per evaluation"""
    cuda: bool = True
    """Use CUDA if available"""


def load_config_hydra(config_path: str, config_name: str = None, overrides: list = None) -> dict:
    """Load configuration using Hydra."""
    # Normalize path to handle both relative and absolute
    if os.path.isabs(config_path):
        # Convert absolute path to relative from current working directory
        abs_path = os.path.abspath(config_path)
        cwd = os.getcwd()
        try:
            rel_path = os.path.relpath(abs_path, cwd)
        except ValueError:
            # Windows: different drives, use basename approach
            rel_path = os.path.basename(abs_path)
    else:
        rel_path = config_path
    
    # Extract directory and config name from relative path
    config_dir = os.path.dirname(rel_path) if os.path.dirname(rel_path) else "."
    if config_name is None:
        config_name = os.path.splitext(os.path.basename(rel_path))[0]
    
    # Initialize Hydra and compose config
    with initialize(version_base=None, config_path=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides or [])
        # Resolve all resolvers
        OmegaConf.resolve(cfg)
    
    # Convert OmegaConf to dict
    return OmegaConf.to_container(cfg, resolve=True)


def make_eval_envs(config: dict, num_eval_envs: int):
    """Create evaluation environments based on config."""
    env_type = config.get("env_type", "gym")
    
    if env_type == "gym":
        eval_envs = make_gym_env(
            env_id=config["env_id"],
            num_envs=num_eval_envs,
            seed=config.get("seed", 1),
            capture_video=False,
            run_name=None,
            gamma=config.get("gamma", 0.99),
        )
        return eval_envs, None, None, {}
    
    elif env_type == "maniskill-state":
        eval_envs, _, max_episode_steps, env_kwargs = make_maniskill_state_env(
            env_id=config["env_id"],
            num_envs=1,  # Single env for eval
            num_eval_envs=num_eval_envs,
            seed=config.get("seed", 1),
            control_mode=config.get("control_mode", "pd_joint_delta_pos"),
            reconfiguration_freq=None,
            eval_reconfiguration_freq=config.get("eval_reconfiguration_freq", 1),
            partial_reset=False,
            eval_partial_reset=False,
            capture_video=False,
            save_trajectory=False,
            run_name=None,
            save_train_video_freq=None,
            num_steps=config.get("num_steps", 50),
            num_eval_steps=config.get("num_eval_steps", 50),
            evaluate=False,
        )
        return eval_envs, None, max_episode_steps, env_kwargs
    
    elif env_type == "maniskill-rgb":
        eval_envs, _, max_episode_steps, env_kwargs = make_maniskill_rgb_env(
            env_id=config["env_id"],
            num_envs=1,  # Single env for eval
            num_eval_envs=num_eval_envs,
            seed=config.get("seed", 1),
            control_mode=config.get("control_mode", "pd_joint_delta_pos"),
            reconfiguration_freq=None,
            eval_reconfiguration_freq=config.get("eval_reconfiguration_freq", 1),
            partial_reset=False,
            eval_partial_reset=False,
            capture_video=False,
            save_trajectory=False,
            run_name=None,
            save_train_video_freq=None,
            num_steps=config.get("num_steps", 50),
            num_eval_steps=config.get("num_eval_steps", 50),
            evaluate=False,
            include_state=config.get("include_state", True),
            render_mode=config.get("render_mode", "all"),
        )
        return eval_envs, None, max_episode_steps, env_kwargs
    else:
        raise ValueError(f"Unknown env_type: {env_type}")


def run_evaluation(eval_envs, agent, env_type: str, num_eval_steps: int, device: torch.device, 
                   logger=None, global_step: int = 0, num_eval_envs: int = 1, seed: int = None):
    """Run evaluation loop during training."""
    eval_metrics = defaultdict(list)
    num_episodes = 0
    
    eval_obs, _ = eval_envs.reset(seed=seed)
    
    # Convert obs to tensor for gym environments
    if env_type == "gym":
        eval_obs = torch.Tensor(eval_obs).to(device)
    
    for _ in range(num_eval_steps):
        with torch.no_grad():
            # Use get_action_and_value for consistency
            eval_action, _, _, _ = agent.get_action_and_value(eval_obs)
            
            # Step environment
            if env_type == "gym":
                next_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(
                    eval_action.cpu().numpy()
                )
                next_obs = torch.Tensor(next_obs).to(device)
            else:
                next_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_action)
            
            # Handle final_info
            if "final_info" in eval_infos:
                if env_type == "gym":
                    # For gym environments, final_info is a list
                    for info in eval_infos["final_info"]:
                        if info is not None and "episode" in info:
                            num_episodes += 1
                            for k, v in info["episode"].items():
                                eval_metrics[k].append(v)
                            print(f"eval_episode={num_episodes}, episodic_return={info['episode'].get('r', 'N/A')}")
                else:
                    # For ManiSkill environments, final_info is a dict with mask
                    mask = eval_infos["_final_info"]
                    if isinstance(mask, torch.Tensor):
                        num_episodes += mask.sum().item()
                    else:
                        num_episodes += sum(mask) if isinstance(mask, (list, np.ndarray)) else int(mask)
                    
                    for k, v in eval_infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v)
            
            eval_obs = next_obs
    
    print(f"Evaluated {num_eval_steps * num_eval_envs} steps resulting in {num_episodes} episodes")
    
    # Compute and print metrics
    for k, v in eval_metrics.items():
        if len(v) > 0:
            if isinstance(v[0], torch.Tensor):
                mean = torch.stack(v).float().mean().item()
            else:
                mean = float(np.mean(v))
            if logger is not None:
                logger.add_scalar(f"eval/{k}", mean, global_step)
            print(f"eval_{k}_mean={mean}")
    
    return eval_metrics


def evaluate(eval_envs, agent, args: EvalArgs, config: dict, device: torch.device):
    """Run evaluation loop for standalone evaluation script."""
    env_type = config.get("env_type", "gym")
    seed = config.get("seed", 1)
    eval_metrics = defaultdict(list)
    num_episodes = 0
    
    eval_obs, _ = eval_envs.reset(seed=seed)
    
    # Convert obs to tensor for gym environments
    if env_type == "gym":
        eval_obs = torch.Tensor(eval_obs).to(device)
    
    for step in range(args.num_eval_steps):
        with torch.no_grad():
            # Use get_action_and_value for consistency
            eval_action, _, _, _ = agent.get_action_and_value(eval_obs)
            
            # Step environment
            if env_type == "gym":
                next_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(
                    eval_action.cpu().numpy()
                )
                next_obs = torch.Tensor(next_obs).to(device)
            else:
                next_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_action)
            
            # Handle final_info
            if "final_info" in eval_infos:
                if env_type == "gym":
                    # For gym environments, final_info is a list
                    for info in eval_infos["final_info"]:
                        if info is not None and "episode" in info:
                            num_episodes += 1
                            for k, v in info["episode"].items():
                                eval_metrics[k].append(v)
                            print(f"eval_episode={num_episodes}, episodic_return={info['episode'].get('r', 'N/A')}")
                            
                            # Stop if we've collected enough episodes
                            if num_episodes >= args.eval_episodes:
                                break
                else:
                    # For ManiSkill environments, final_info is a dict with mask
                    mask = eval_infos["_final_info"]
                    if isinstance(mask, torch.Tensor):
                        num_episodes += mask.sum().item()
                    else:
                        num_episodes += sum(mask) if isinstance(mask, (list, np.ndarray)) else int(mask)
                    
                    for k, v in eval_infos["final_info"]["episode"].items():
                        eval_metrics[k].append(v)
            
            eval_obs = next_obs
            
            # Stop if we've collected enough episodes
            if num_episodes >= args.eval_episodes:
                break
    
    print(f"Evaluated {step + 1} steps resulting in {num_episodes} episodes")
    
    # Compute and print metrics
    for k, v in eval_metrics.items():
        if len(v) > 0:
            if isinstance(v[0], torch.Tensor):
                mean = torch.stack(v).float().mean().item()
            else:
                mean = float(np.mean(v))
            print(f"eval_{k}_mean={mean}")
    
    return eval_metrics


if __name__ == "__main__":
    args = tyro.cli(EvalArgs)
    
    if not args.config:
        raise ValueError("--config is required")
    if not args.checkpoint:
        raise ValueError("--checkpoint is required")
    
    # Load config
    config = load_config_hydra(args.config)
    
    # Set random seeds for reproducibility
    seed = config.get("seed", 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_deterministic = config.get("torch_deterministic", True)
    torch.backends.cudnn.deterministic = torch_deterministic
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}, seed: {seed}")
    
    # Create evaluation environments
    eval_envs, _, max_episode_steps, env_kwargs = make_eval_envs(config, args.num_eval_envs)
    
    # Get observation and action dimensions
    env_type = config.get("env_type", "gym")
    if env_type == "gym":
        n_obs = int(np.array(eval_envs.single_observation_space.shape).prod())
        n_act = int(np.prod(eval_envs.single_action_space.shape))
        sample_obs = None
    elif env_type == "maniskill-state":
        n_obs = int(torch.prod(torch.tensor(eval_envs.single_observation_space.shape)))
        n_act = int(torch.prod(torch.tensor(eval_envs.single_action_space.shape)))
        sample_obs = None
    elif env_type == "maniskill-rgb":
        next_obs, _ = eval_envs.reset(seed=seed)
        sample_obs = next_obs
        n_act = int(np.prod(eval_envs.unwrapped.single_action_space.shape))
        n_obs = None
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
    
    # Create a simple args-like object for agent initialization
    class SimpleArgs:
        def __init__(self, config_dict):
            for k, v in config_dict.items():
                setattr(self, k, v)
    
    train_args = SimpleArgs(config)
    
    action_space = eval_envs.single_action_space if hasattr(eval_envs, 'single_action_space') else eval_envs.unwrapped.single_action_space
    algorithm = config.get("algorithm", "ppo")
    if algorithm == "ppocma":
        from ppocma import PPOCMAAgent
        agent = PPOCMAAgent(
            n_obs=n_obs,
            n_act=n_act,
            action_space=action_space,
            args=train_args,
            device=device,
            sample_obs=sample_obs,
        ).to(device)
    else:
        agent = Agent(
            n_obs=n_obs,
            n_act=n_act,
            action_space=action_space,
            args=train_args,
            device=device,
            sample_obs=sample_obs,
        ).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()
    
    # Run evaluation
    print(f"Starting evaluation with {args.eval_episodes} episodes...")
    stime = time.perf_counter()
    eval_metrics = evaluate(eval_envs, agent, args, config, device)
    eval_time = time.perf_counter() - stime
    
    print(f"\nEvaluation completed in {eval_time:.2f} seconds")
    
    eval_envs.close()
