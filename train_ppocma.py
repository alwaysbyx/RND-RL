"""Training script for PPO-CMA across Gym and ManiSkill environments.

Implements Algorithm 2 from:
  Hämäläinen et al., "PPO-CMA: Proximal Policy Optimization with Covariance Matrix Adaptation"
  IEEE MLSP 2020. arXiv:1810.02541

Reuses environment, GAE, logging, and evaluation infrastructure from train.py.
"""
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

from ppocma import PPOCMAAgent, HistoryBuffer
from envs import make_gym_env, make_maniskill_state_env, make_maniskill_rgb_env
from envs.maniskill_rgb_env import DictArray
from eval import run_evaluation
import wandb


@dataclass
class PPOCMAArgs:
    """PPO-CMA training arguments."""
    config: Optional[str] = None
    config_overrides: Optional[list] = None
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "PPO_Scale"
    wandb_entity: Optional[str] = "alwaysbb"
    wandb_group: str = "PPO-CMA"
    capture_video: bool = False
    save_trajectory: bool = False
    save_model: bool = False
    evaluate: bool = False
    checkpoint: Optional[str] = None
    algorithm: str = "ppocma"

    # Environment
    env_type: str = "gym"
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

    # Algorithm — shared
    total_timesteps: int = 1000000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 64
    norm_adv: bool = False
    max_grad_norm: float = 0.5
    reward_scale: float = 1.0
    finite_horizon_gae: bool = False
    eval_freq: int = 1
    save_train_video_freq: Optional[int] = None

    # PPO-CMA specific
    mean_lr: float = 3e-4
    var_lr: float = 3e-4
    critic_lr: float = 3e-4
    learning_rate: float = 3e-4  # fallback / compatibility
    anneal_lr: bool = False
    history_buffer_size: int = 5  # H
    variance_train_steps: int = 100  # K for variance
    mean_train_steps: int = 100  # K for mean
    critic_train_steps: int = 100  # K for critic
    lower_std_limit: float = 0.01
    critic_loss_type: str = "l1"  # "l1" or "l2"
    use_mirroring: bool = True  # mirror negative advantages; False = clip to zero

    # Network architecture
    discrete_action: bool = False  # PPO-CMA is continuous only
    ppocma_actor_width: int = 128
    ppocma_actor_depth: int = 2
    actor_width: int = 128  # fallback
    actor_depth: int = 2  # fallback
    critic_width: int = 128
    critic_depth: int = 2
    use_residual_blocks: bool = False
    num_bins: int = 41  # unused, kept for config compatibility

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


def load_config_hydra(config_path: str, config_name: str = None, overrides: list = None) -> PPOCMAArgs:
    """Load configuration using Hydra and convert to PPOCMAArgs dataclass."""
    if os.path.isabs(config_path):
        abs_path = os.path.abspath(config_path)
        cwd = os.getcwd()
        try:
            rel_path = os.path.relpath(abs_path, cwd)
        except ValueError:
            rel_path = os.path.basename(abs_path)
    else:
        rel_path = config_path

    config_dir = os.path.dirname(rel_path) if os.path.dirname(rel_path) else "."
    if config_name is None:
        config_name = os.path.splitext(os.path.basename(rel_path))[0]

    with initialize(version_base=None, config_path=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides or [])
        OmegaConf.resolve(cfg)

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    return PPOCMAArgs(**config_dict)


def make_envs(args: PPOCMAArgs):
    """Create training and evaluation environments based on env_type."""
    if args.env_type == "gym":
        envs = make_gym_env(
            env_id=args.env_id,
            num_envs=args.num_envs if not args.evaluate else 1,
            seed=args.seed,
            capture_video=args.capture_video,
            run_name=None,
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
        return envs, eval_envs, None, {}

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
            run_name=None,
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
            run_name=None,
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
        return None, n_act
    else:
        raise ValueError(f"Unknown env_type: {args.env_type}")


if __name__ == "__main__":
    args = tyro.cli(PPOCMAArgs)

    if args.config:
        overrides = args.config_overrides if args.config_overrides else []
        args = load_config_hydra(args.config, overrides=overrides)

    # Compute runtime values
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # Generate run name
    if args.exp_name is None:
        args.exp_name = "ppocma"
        run_name = (
            f"{args.env_id}__ppocma_H{args.history_buffer_size}_K{args.mean_train_steps}"
            f"__{args.seed}__"
            f"aw{args.ppocma_actor_width}_ad{args.ppocma_actor_depth}"
            f"_cw{args.critic_width}_cd{args.critic_depth}__"
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
                tags=["ppocma", "scale"],
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        logger = Logger(log_wandb=False, tensorboard=None)

    # Create PPO-CMA agent
    action_space = envs.single_action_space if hasattr(envs, 'single_action_space') else envs.unwrapped.single_action_space
    agent = PPOCMAAgent(
        n_obs=n_obs,
        n_act=n_act,
        action_space=action_space,
        args=args,
        device=device,
        sample_obs=sample_obs,
    ).to(device)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    # Three separate optimizers (Section 5, rank-μ update)
    mean_optimizer = optim.Adam(agent.mean_parameters(), lr=args.mean_lr, eps=1e-5)
    var_optimizer = optim.Adam(agent.var_parameters(), lr=args.var_lr, eps=1e-5)
    critic_optimizer = optim.Adam(agent.critic_parameters(), lr=args.critic_lr, eps=1e-5)

    # History buffer for variance training (Section 5.1)
    history_buffer = HistoryBuffer(max_size=args.history_buffer_size)

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

        # Anneal learning rate (optional, not used by default in PPO-CMA)
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            mean_optimizer.param_groups[0]["lr"] = frac * args.mean_lr
            var_optimizer.param_groups[0]["lr"] = frac * args.var_lr
            critic_optimizer.param_groups[0]["lr"] = frac * args.critic_lr

        # =====================================================================
        # Rollout (same as PPO)
        # =====================================================================
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
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            for k, v in info["episode"].items():
                                if logger is not None:
                                    logger.add_scalar(f"train/{k}", v, global_step)
                else:
                    final_info = infos["final_info"]
                    done_mask = infos["_final_info"]
                    if isinstance(done_mask, torch.Tensor):
                        if done_mask.dtype != torch.bool:
                            done_mask = done_mask.bool()
                    else:
                        done_mask = torch.tensor(done_mask, device=device, dtype=torch.bool)

                    for k, v in final_info["episode"].items():
                        if logger is not None:
                            logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

                    if "final_observation" in infos:
                        final_obs = infos["final_observation"]
                        if isinstance(final_obs, dict):
                            for k in final_obs:
                                final_obs[k] = final_obs[k][done_mask]
                        else:
                            final_obs = final_obs[done_mask]
                        with torch.no_grad():
                            final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(final_obs).view(-1)

        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time

        # =====================================================================
        # Algorithm 2, Step 6: Train critic (before GAE, using current iteration)
        # =====================================================================
        agent.train()
        update_time = time.perf_counter()

        # We need value targets for critic training. Use simple TD(lambda) returns
        # computed with the OLD value estimates (before critic update).
        # First compute GAE to get returns, then train critic on those returns.

        # =====================================================================
        # GAE computation (same as PPO)
        # =====================================================================
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
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                else:
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
        b_actions = actions.reshape((-1,) + action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # =====================================================================
        # Algorithm 2, Step 6: Train critic
        # =====================================================================
        b_inds = np.arange(args.batch_size)
        for k in range(args.critic_train_steps):
            np.random.shuffle(b_inds)
            mb_inds = b_inds[:args.minibatch_size]
            newvalue = agent.get_value(b_obs[mb_inds]).view(-1)
            if args.critic_loss_type == "l1":
                v_loss = torch.abs(newvalue - b_returns[mb_inds]).mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
            critic_optimizer.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(agent.critic_parameters(), args.max_grad_norm)
            critic_optimizer.step()

        # =====================================================================
        # Algorithm 2, Step 8: Mirror or clip negative advantages
        # =====================================================================
        with torch.no_grad():
            b_means, b_stds, _ = agent.get_mean_and_std(b_obs)

        if args.use_mirroring:
            b_mir_actions, b_mir_advantages = PPOCMAAgent.mirror_actions(
                b_actions, b_means, b_advantages, b_stds
            )
        else:
            # Clip negative advantages to zero
            b_mir_actions = b_actions.clone()
            b_mir_advantages = torch.clamp(b_advantages, min=0.0)

        # =====================================================================
        # Append to history buffer (for variance training)
        # =====================================================================
        history_buffer.append({
            "obs": b_obs.detach(),
            "actions": b_mir_actions.detach(),
            "advantages": b_mir_advantages.detach(),
        })

        # =====================================================================
        # Algorithm 2, Step 9: Train variance network (history buffer, past H iterations)
        # =====================================================================
        hist_data = history_buffer.get_all()
        hist_obs = hist_data["obs"]
        hist_actions = hist_data["actions"]
        hist_advantages = hist_data["advantages"]
        hist_size = hist_obs.shape[0]
        hist_inds = np.arange(hist_size)

        for k in range(args.variance_train_steps):
            np.random.shuffle(hist_inds)
            mb_inds = hist_inds[:args.minibatch_size]
            var_loss = agent.compute_var_loss(
                hist_obs[mb_inds],
                hist_actions[mb_inds],
                hist_advantages[mb_inds],
            )
            var_optimizer.zero_grad()
            var_loss.backward()
            nn.utils.clip_grad_norm_(agent.var_parameters(), args.max_grad_norm)
            var_optimizer.step()

        # =====================================================================
        # Algorithm 2, Step 10: Train mean network (current iteration only)
        # =====================================================================
        for k in range(args.mean_train_steps):
            np.random.shuffle(b_inds)
            mb_inds = b_inds[:args.minibatch_size]
            mean_loss = agent.compute_mean_loss(
                b_obs[mb_inds],
                b_mir_actions[mb_inds],
                b_mir_advantages[mb_inds],
            )
            mean_optimizer.zero_grad()
            mean_loss.backward()
            nn.utils.clip_grad_norm_(agent.mean_parameters(), args.max_grad_norm)
            mean_optimizer.step()

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # =====================================================================
        # Logging
        # =====================================================================
        if logger is not None:
            logger.add_scalar("charts/learning_rate_mean", mean_optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("charts/learning_rate_var", var_optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("charts/learning_rate_critic", critic_optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
            logger.add_scalar("losses/mean_loss", mean_loss.item(), global_step)
            logger.add_scalar("losses/var_loss", var_loss.item(), global_step)
            logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            logger.add_scalar("time/step", global_step, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
            # PPO-CMA specific metrics
            logger.add_scalar("ppocma/history_buffer_size", len(history_buffer), global_step)
            logger.add_scalar("ppocma/mean_advantage", b_advantages.mean().item(), global_step)
            logger.add_scalar("ppocma/positive_advantage_frac",
                              (b_advantages >= 0).float().mean().item(), global_step)
            with torch.no_grad():
                _, stds, _ = agent.get_mean_and_std(b_obs[:min(1024, b_obs.shape[0])])
                logger.add_scalar("ppocma/mean_std", stds.mean().item(), global_step)
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)

        if iteration % 10 == 0:
            print(f"Iteration: {iteration}, global_step: {global_step}, "
                  f"SPS: {int(global_step / (time.time() - start_time))}, "
                  f"hist_buf: {len(history_buffer)}/{args.history_buffer_size}")

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
