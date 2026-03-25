"""Unified training script for PPO across Gym and ManiSkill environments."""
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from hydra import initialize, compose
from omegaconf import OmegaConf

# Project imports
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rnd import Agent
from envs import make_gym_env, make_maniskill_state_env, make_maniskill_rgb_env
from envs.maniskill_rgb_env import DictArray
from eval import run_evaluation
import wandb


@dataclass
class Args:
    """Training arguments."""
    config: Optional[str] = None
    """Path to YAML config file"""
    config_overrides: Optional[list] = None
    """List of Hydra overrides (e.g., ['num_envs=128', 'learning_rate=1e-4'])"""
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "PPO_Scale"
    wandb_entity: Optional[str] = "alwaysbb"
    wandb_group: str = "PPO"
    capture_video: bool = False
    save_trajectory: bool = False
    save_model: bool = False
    evaluate: bool = False
    checkpoint: Optional[str] = None

    # Environment (will be overridden by config)
    env_type: str = "gym"  # "gym", "maniskill-state", "maniskill-rgb"
    env_id: str = "HalfCheetah-v4"
    num_envs: int = 64
    num_eval_envs: int = 8
    num_steps: int = 32
    num_eval_steps: int = 32
    partial_reset: bool = True
    eval_partial_reset: bool = False
    reconfiguration_freq: Optional[int] = None
    eval_reconfiguration_freq: int = 1
    control_mode: Optional[str] = "pd_joint_delta_pos"
    include_state: bool = True
    render_mode: str = "all"

    # Algorithm
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 64
    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    reward_scale: float = 1.0
    finite_horizon_gae: bool = False
    eval_freq: int = 1
    save_train_video_freq: Optional[int] = None

    # Network architecture
    discrete_action: bool = False
    num_bins: int = 41
    actor_width: int = 256
    actor_depth: int = 3
    critic_width: int = 256
    critic_depth: int = 3
    use_residual_blocks: bool = False

    # Runtime-computed
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


class Logger:
    """Unified logger for wandb and tensorboard."""
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


def load_config_hydra(config_path: str, config_name: str = None, overrides: list = None) -> Args:
    """Load configuration using Hydra and convert directly to Args dataclass."""
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
    
    # Convert OmegaConf to dict and then to Args dataclass
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    return Args(**config_dict)


def make_envs(args: Args):
    """Create training and evaluation environments based on env_type."""
    if args.env_type == "gym":
        envs = make_gym_env(
            env_id=args.env_id,
            num_envs=args.num_envs if not args.evaluate else 1,
            seed=args.seed,
            capture_video=args.capture_video,
            run_name=None,  # Will be set later
            gamma=args.gamma,
        )
        eval_envs = make_gym_env(
            env_id=args.env_id,
            num_envs=args.num_eval_envs,
            seed=args.seed,
            capture_video=args.capture_video,
            run_name=None,
            gamma=args.gamma,
        )
        max_episode_steps = None
        env_kwargs = {}
        return envs, eval_envs, max_episode_steps, env_kwargs

    elif args.env_type == "maniskill-state":
        return make_maniskill_state_env(
            env_id=args.env_id,
            num_envs=args.num_envs,
            num_eval_envs=args.num_eval_envs,
            seed=args.seed,
            control_mode=args.control_mode,
            reconfiguration_freq=args.reconfiguration_freq,
            eval_reconfiguration_freq=args.eval_reconfiguration_freq,
            partial_reset=args.partial_reset,
            eval_partial_reset=args.eval_partial_reset,
            capture_video=args.capture_video,
            save_trajectory=args.save_trajectory,
            run_name=None,  # Will be set later
            save_train_video_freq=args.save_train_video_freq,
            num_steps=args.num_steps,
            num_eval_steps=args.num_eval_steps,
            evaluate=args.evaluate,
        )

    elif args.env_type == "maniskill-rgb":
        return make_maniskill_rgb_env(
            env_id=args.env_id,
            num_envs=args.num_envs,
            num_eval_envs=args.num_eval_envs,
            seed=args.seed,
            control_mode=args.control_mode,
            reconfiguration_freq=args.reconfiguration_freq,
            eval_reconfiguration_freq=args.eval_reconfiguration_freq,
            partial_reset=args.partial_reset,
            eval_partial_reset=args.eval_partial_reset,
            capture_video=args.capture_video,
            save_trajectory=args.save_trajectory,
            run_name=None,  # Will be set later
            save_train_video_freq=args.save_train_video_freq,
            num_steps=args.num_steps,
            num_eval_steps=args.num_eval_steps,
            evaluate=args.evaluate,
            include_state=args.include_state,
            render_mode=args.render_mode,
        )
    else:
        raise ValueError(f"Unknown env_type: {args.env_type}")


def get_obs_action_dims(envs, args, sample_obs=None):
    """Get observation and action dimensions based on environment type."""
    if args.env_type == "gym":
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        act_dim = int(np.prod(envs.single_action_space.shape))
        return obs_dim, act_dim
    elif args.env_type == "maniskill-state":
        n_act = int(torch.prod(torch.tensor(envs.single_action_space.shape)))
        n_obs = int(torch.prod(torch.tensor(envs.single_observation_space.shape)))
        return n_obs, n_act
    elif args.env_type == "maniskill-rgb":
        n_act = int(np.prod(envs.unwrapped.single_action_space.shape))
        # obs_dim will be computed from feature extractor in Agent
        return None, n_act
    else:
        raise ValueError(f"Unknown env_type: {args.env_type}")


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Load config if provided
    if args.config:
        overrides = args.config_overrides if args.config_overrides else []
        args = load_config_hydra(args.config, overrides=overrides)

    # Compute runtime values
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # Generate run name
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        residual_suffix = "_residual" if args.use_residual_blocks else ""
        discrete_suffix = "_discrete" if args.discrete_action else "_continuous"
        run_name = (
            f"{args.env_id}__{args.exp_name}{discrete_suffix}{residual_suffix}__{args.seed}_"
            f"aw{args.actor_width}_ad{args.actor_depth}_cw{args.critic_width}_cd{args.critic_depth}__"
            f"{int(time.time())}"
        )
    else:
        run_name = args.exp_name

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")
    # Create environments
    envs, eval_envs, max_episode_steps, env_kwargs = make_envs(args)

    # Get observation and action dimensions
    sample_obs = None
    if args.env_type == "maniskill-rgb":
        next_obs, _ = envs.reset(seed=args.seed)
        sample_obs = next_obs
        n_obs, n_act = get_obs_action_dims(envs, args, sample_obs)
    else:
        n_obs, n_act = get_obs_action_dims(envs, args)

    # Setup logger
    logger = None
    if not args.evaluate:
        if args.track:
            wandb.login(key="6eb16696cec55f88c62b7bbc82a5d16284c915cf")
            config_dict = vars(args)
            config_dict["env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_envs,
                env_id=args.env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps if max_episode_steps else None,
                partial_reset=args.partial_reset if args.env_type != "gym" else None,
            )
            config_dict["eval_env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_eval_envs,
                env_id=args.env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps if max_episode_steps else None,
                partial_reset=args.eval_partial_reset if args.env_type != "gym" else None,
            )
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config_dict,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["ppo", "scale"],
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        logger = Logger(log_wandb=False, tensorboard=None)

    # Create agent
    action_space = envs.single_action_space if hasattr(envs, 'single_action_space') else envs.unwrapped.single_action_space
    agent = Agent(
        n_obs=n_obs,
        n_act=n_act,
        action_space=action_space,
        args=args,
        device=device,
        sample_obs=sample_obs,
    ).to(device)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Storage setup
    if args.env_type == "maniskill-rgb":
        obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device=device)
    else:
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)

    actions = torch.zeros((args.num_steps, args.num_envs) + action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start training
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    if args.env_type == "gym":
        next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.float32)

    cumulative_times = defaultdict(float)

    for iteration in range(1, args.num_iterations + 1):
        agent.eval()
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)

        # Evaluation
        if iteration % args.eval_freq == 0:
            stime = time.perf_counter()
            eval_metrics = run_evaluation(
                eval_envs=eval_envs,
                agent=agent,
                env_type=args.env_type,
                num_eval_steps=args.num_eval_steps,
                device=device,
                logger=logger,
                global_step=global_step,
                num_eval_envs=args.num_eval_envs,
                seed=args.seed,
            )
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break

        if args.save_model and iteration % args.eval_freq == 0:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")

        # Anneal learning rate
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Rollout
        rollout_time = time.perf_counter()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute environment step
            if args.env_type == "gym":
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = torch.tensor(np.logical_or(terminations, truncations), device=device, dtype=torch.float32)
                next_obs = torch.Tensor(next_obs).to(device)
            else:
                next_obs, reward, terminations, truncations, infos = envs.step(action)
                next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            
            # Handle rewards
            if args.env_type == "gym":
                rewards[step] = torch.tensor(reward, device=device, dtype=torch.float32).view(-1) * args.reward_scale
            else:
                rewards[step] = reward.view(-1) * args.reward_scale

            # Log training metrics
            if "final_info" in infos:
                if args.env_type == "gym":
                    # Gym: final_info is a list of info dicts
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            for k, v in info["episode"].items():
                                if logger is not None:
                                    logger.add_scalar(f"train/{k}", v, global_step)
                                    
                else:
                    # ManiSkill: final_info is a dict with episode and mask
                    final_info = infos["final_info"]
                    done_mask = infos["_final_info"]
                    
                    # Ensure done_mask is bool tensor
                    if isinstance(done_mask, torch.Tensor):
                        if done_mask.dtype != torch.bool:
                            done_mask = done_mask.bool()
                    else:
                        done_mask = torch.tensor(done_mask, device=device, dtype=torch.bool)
                    
                    for k, v in final_info["episode"].items():
                        if logger is not None:
                            logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                    
                    # Store final values for GAE bootstrap
                    if "final_observation" in infos:
                        final_obs = infos["final_observation"]
                        # Handle both dict (RGB) and tensor (state) observations
                        if isinstance(final_obs, dict):
                            # ManiSkill RGB: apply done_mask to each key
                            for k in final_obs:
                                final_obs[k] = final_obs[k][done_mask]
                        else:
                            # ManiSkill State: apply done_mask directly
                            final_obs = final_obs[done_mask]
                        
                        with torch.no_grad():
                            final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(final_obs).view(-1)
                
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time

        # GAE computation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                
                if args.env_type == "gym":
                    # Simple GAE for gym environments
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                else:
                    # ManiSkill: handle final_values for bootstrap
                    real_next_values = nextnonterminal * nextvalues + final_values[t]
                    if args.finite_horizon_gae:
                        if t == args.num_steps - 1:
                            lam_coef_sum = 0.0
                            reward_term_sum = 0.0
                            value_term_sum = 0.0
                        lam_coef_sum = lam_coef_sum * nextnonterminal
                        reward_term_sum = reward_term_sum * nextnonterminal
                        value_term_sum = value_term_sum * nextnonterminal
                        lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                        reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                        value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values
                        advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                    else:
                        delta = rewards[t] + args.gamma * real_next_values - values[t]
                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten batch
        if args.env_type == "maniskill-rgb":
            b_obs = obs.reshape((-1,))
        else:
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Update
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.perf_counter()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if logger is not None:
            logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            logger.add_scalar("losses/explained_variance", explained_var, global_step)
            logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            logger.add_scalar("time/step", global_step, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)

        if iteration % 10 == 0:
            print(f"Iteration: {iteration}, global_step: {global_step}, SPS: {int(global_step / (time.time() - start_time))}")

    if not args.evaluate:
        if args.save_model:
            model_path = f"runs/{run_name}/final_ckpt.pt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        if logger is not None:
            logger.close()

    envs.close()
    eval_envs.close()
