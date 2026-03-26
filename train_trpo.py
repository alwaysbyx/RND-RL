"""Unified training script for TRPO across Gym and ManiSkill environments.

Mirrors train.py (PPO) but replaces the clipped-loss update with TRPO's
natural gradient via conjugate gradient + Fisher-vector product + line search.
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
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
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
from train import Logger, make_envs, get_obs_action_dims
import wandb


@dataclass
class Args:
    """Training arguments for TRPO."""
    config: Optional[str] = None
    """Path to YAML config file"""
    config_overrides: Optional[list] = None
    """List of Hydra overrides (e.g., ['num_envs=128', 'learning_rate=1e-4'])"""
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "TRPO_Scale"
    wandb_entity: Optional[str] = "alwaysbb"
    wandb_group: str = "TRPO"
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

    # Algorithm (shared with PPO for config compatibility)
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

    # TRPO-specific
    max_kl: float = 0.01
    """Trust region KL divergence constraint"""
    damping: float = 0.1
    """Fisher matrix damping for numerical stability"""
    cg_iters: int = 10
    """Conjugate gradient iterations"""
    backtrack_iters: int = 10
    """Line search backtracking iterations"""
    backtrack_coef: float = 0.8
    """Line search backtracking coefficient"""
    vf_lr: float = 1e-3
    """Critic learning rate (separate from actor)"""
    vf_iters: int = 5
    """Critic update iterations per batch"""

    # Runtime-computed
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


def load_config_hydra(config_path: str, config_name: str = None, overrides: list = None) -> Args:
    """Load configuration using Hydra and convert directly to Args dataclass."""
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
    return Args(**config_dict)


# ---------------------------------------------------------------------------
# TRPO helper functions
# ---------------------------------------------------------------------------

def get_actor_params(agent):
    """Collect actor parameters (actor_mean + actor_logstd + feature_net for RGB)."""
    params = list(agent.actor_mean.parameters())
    if not agent.discrete_action:
        params.append(agent.actor_logstd)
    if agent.use_rgb and agent.feature_net is not None:
        params += list(agent.feature_net.parameters())
    return params


def get_critic_params(agent):
    """Collect critic parameters (critic + feature_net for RGB)."""
    params = list(agent.critic.parameters())
    if agent.use_rgb and agent.feature_net is not None:
        params += list(agent.feature_net.parameters())
    return params


def get_policy_distribution(agent, obs):
    """Get the policy distribution for given observations.

    Mirrors rnd.py Agent's internal distribution logic so we can compute
    analytical KL divergence for the Fisher-vector product.
    """
    features = agent.get_features(obs)

    if agent.discrete_action:
        logits = agent.actor_mean(features).reshape(-1, agent.n_act, agent.num_bins)
        probs = torch.softmax(logits, dim=-1)
        return Categorical(probs=probs)
    else:
        if agent.use_residual_blocks:
            action_mean = agent.actor_mean(features)
        else:
            action_mean_raw = agent.actor_mean(features)
            action_mean = torch.tanh(action_mean_raw) * agent.action_scale
        action_logstd = agent.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return Normal(action_mean, action_std)


def compute_kl(old_dist, new_dist, agent):
    """Mean KL(old || new), summed over action dims, averaged over batch."""
    kl = kl_divergence(old_dist, new_dist)  # (batch,) or (batch, n_act) or (batch, n_act, num_bins)
    if agent.discrete_action:
        # Categorical KL: shape (batch, n_act)
        return kl.sum(dim=-1).mean()
    else:
        # Normal KL: shape (batch, n_act)
        return kl.sum(dim=-1).mean()


def get_flat_params(params):
    """Flatten a list of parameters into a single 1D vector."""
    return torch.cat([p.data.reshape(-1) for p in params])


def set_flat_params(params, flat_vector):
    """Set parameters from a flat 1D vector."""
    offset = 0
    for p in params:
        numel = p.numel()
        p.data.copy_(flat_vector[offset:offset + numel].reshape(p.shape))
        offset += numel


def flat_grad_from_list(grads):
    """Flatten a list of gradient tensors into a single 1D vector."""
    return torch.cat([g.reshape(-1) for g in grads if g is not None])


def conjugate_gradient(fvp_fn, b, nsteps=10, residual_tol=1e-10):
    """Solve Hx = b via conjugate gradient, where H is applied implicitly by fvp_fn."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = r.dot(r)

    for _ in range(nsteps):
        Ap = fvp_fn(p)
        pAp = p.dot(Ap)
        if pAp <= 0:
            break
        alpha = rdotr / (pAp + 1e-8)
        x += alpha * p
        r -= alpha * Ap
        new_rdotr = r.dot(r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr

    return x, rdotr


def trpo_update(agent, b_obs, b_actions, b_advantages, b_logprobs, args):
    """Perform a single TRPO policy update on the full batch.

    Returns a dict of metrics for logging.
    """
    actor_params = get_actor_params(agent)

    # 1. Get old distribution (detached for KL reference)
    with torch.no_grad():
        old_dist = get_policy_distribution(agent, b_obs)
        if agent.discrete_action:
            old_dist_detached = Categorical(probs=old_dist.probs.detach().clone())
        else:
            old_dist_detached = Normal(old_dist.loc.detach().clone(),
                                       old_dist.scale.detach().clone())

    # 2. Forward pass with gradients
    new_dist = get_policy_distribution(agent, b_obs)

    # Compute new log probs for the collected actions
    if agent.discrete_action:
        action_bins = agent.action_bins.to(b_actions.device)
        diff = b_actions.unsqueeze(-1) - action_bins.view(1, 1, -1)
        actions_idx = diff.abs().argmin(dim=-1)
        new_logprobs = new_dist.log_prob(actions_idx).sum(dim=-1)
    else:
        new_logprobs = new_dist.log_prob(b_actions).sum(dim=-1)

    # Importance sampling ratio
    ratio = torch.exp(new_logprobs - b_logprobs)

    # Normalize advantages
    if args.norm_adv:
        b_adv_norm = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
    else:
        b_adv_norm = b_advantages

    # Surrogate loss (negative because we want to maximize)
    surrogate_loss = -(ratio * b_adv_norm).mean()

    # 3. Policy gradient
    grads = torch.autograd.grad(surrogate_loss, actor_params, retain_graph=True)
    loss_grad = flat_grad_from_list(grads)

    # If gradient is essentially zero, skip update
    if loss_grad.norm() < 1e-10:
        return {
            "surrogate_loss": surrogate_loss.item(),
            "kl_divergence": 0.0,
            "surrogate_loss_improvement": 0.0,
            "step_size": 0.0,
            "backtrack_steps": 0,
            "cg_residual": 0.0,
            "max_ratio": ratio.max().item(),
            "approx_kl": 0.0,
            "old_approx_kl": 0.0,
            "entropy": new_dist.entropy().sum(dim=-1).mean().item(),
        }

    # 4. Fisher-vector product via KL Hessian
    def fvp_fn(v):
        kl = compute_kl(old_dist_detached, new_dist, agent)
        kl_grads = torch.autograd.grad(kl, actor_params, create_graph=True)
        flat_kl_grad = flat_grad_from_list(kl_grads)
        kl_v = flat_kl_grad.dot(v)
        kl_v_grads = torch.autograd.grad(kl_v, actor_params, retain_graph=True)
        flat_kl_v_grad = flat_grad_from_list(kl_v_grads)
        return flat_kl_v_grad + args.damping * v

    # 5. Conjugate gradient
    step_dir, cg_residual = conjugate_gradient(fvp_fn, loss_grad, nsteps=args.cg_iters)

    # 6. Compute max step size: sqrt(2 * max_kl / (s^T H s))
    shs = step_dir.dot(fvp_fn(step_dir))
    if shs <= 0:
        shs = torch.tensor(1e-8)
    max_step = torch.sqrt(2 * args.max_kl / (shs + 1e-8))
    full_step = max_step * step_dir

    # 7. Line search with backtracking
    old_params = get_flat_params(actor_params)
    old_loss = surrogate_loss.item()

    backtrack_steps = 0
    success = False
    actual_step_size = 0.0
    new_loss_val = old_loss

    for i in range(args.backtrack_iters):
        step_frac = args.backtrack_coef ** i
        new_params = old_params - step_frac * full_step
        set_flat_params(actor_params, new_params)

        with torch.no_grad():
            new_dist_check = get_policy_distribution(agent, b_obs)

            if agent.discrete_action:
                new_logprobs_check = new_dist_check.log_prob(actions_idx).sum(dim=-1)
            else:
                new_logprobs_check = new_dist_check.log_prob(b_actions).sum(dim=-1)

            ratio_check = torch.exp(new_logprobs_check - b_logprobs)
            new_loss = -(ratio_check * b_adv_norm).mean()
            kl_check = compute_kl(old_dist_detached, new_dist_check, agent)

        if kl_check <= args.max_kl and new_loss <= old_loss:
            success = True
            backtrack_steps = i
            actual_step_size = (step_frac * max_step).item()
            new_loss_val = new_loss.item()
            break

    if not success:
        # Revert to old parameters
        set_flat_params(actor_params, old_params)
        backtrack_steps = args.backtrack_iters
        actual_step_size = 0.0
        new_loss_val = old_loss

    # 8. Compute final metrics
    with torch.no_grad():
        final_dist = get_policy_distribution(agent, b_obs)
        final_kl = compute_kl(old_dist_detached, final_dist, agent)

        if agent.discrete_action:
            final_logprobs = final_dist.log_prob(actions_idx).sum(dim=-1)
        else:
            final_logprobs = final_dist.log_prob(b_actions).sum(dim=-1)

        logratio = final_logprobs - b_logprobs
        approx_kl = ((torch.exp(logratio) - 1) - logratio).mean()
        old_approx_kl = (-logratio).mean()
        entropy = final_dist.entropy().sum(dim=-1).mean()
        max_ratio = torch.exp(logratio).max()

    return {
        "surrogate_loss": old_loss,
        "kl_divergence": final_kl.item(),
        "surrogate_loss_improvement": old_loss - new_loss_val,
        "step_size": actual_step_size,
        "backtrack_steps": backtrack_steps,
        "cg_residual": cg_residual.item() if isinstance(cg_residual, torch.Tensor) else cg_residual,
        "max_ratio": max_ratio.item(),
        "approx_kl": approx_kl.item(),
        "old_approx_kl": old_approx_kl.item(),
        "entropy": entropy.item(),
    }


def update_critic(agent, b_obs, b_returns, args, critic_optimizer):
    """Update critic with minibatch SGD (Adam), returns final value loss."""
    batch_size = b_returns.shape[0]
    minibatch_size = max(1, batch_size // args.num_minibatches)
    v_loss = torch.tensor(0.0)

    for _ in range(args.vf_iters):
        b_inds = np.arange(batch_size)
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]
            value_pred = agent.get_value(b_obs[mb_inds]).view(-1)
            v_loss = 0.5 * ((value_pred - b_returns[mb_inds]) ** 2).mean()
            critic_optimizer.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(get_critic_params(agent), args.max_grad_norm)
            critic_optimizer.step()

    return v_loss.item()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

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
                tags=["trpo", "scale"],
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

    # Separate critic optimizer (TRPO uses natural gradient for actor, Adam for critic)
    critic_optimizer = optim.Adam(get_critic_params(agent), lr=args.vf_lr, eps=1e-5)

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

        # Anneal critic learning rate
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.vf_lr
            critic_optimizer.param_groups[0]["lr"] = lrnow

        # Rollout (identical to train.py)
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

        # GAE computation (identical to train.py)
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

        # Flatten batch (identical to train.py)
        if args.env_type == "maniskill-rgb":
            b_obs = obs.reshape((-1,))
        else:
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # ---- TRPO Update ----
        agent.train()
        update_time = time.perf_counter()

        # Policy update via natural gradient (full batch)
        trpo_metrics = trpo_update(
            agent=agent,
            b_obs=b_obs,
            b_actions=b_actions,
            b_advantages=b_advantages,
            b_logprobs=b_logprobs,
            args=args,
        )

        # Critic update (minibatch SGD with Adam)
        v_loss = update_critic(
            agent=agent,
            b_obs=b_obs,
            b_returns=b_returns,
            args=args,
            critic_optimizer=critic_optimizer,
        )

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # Logging
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if logger is not None:
            # Shared metrics (consistent with PPO train.py)
            logger.add_scalar("charts/learning_rate", critic_optimizer.param_groups[0]["lr"], global_step)
            logger.add_scalar("losses/value_loss", v_loss, global_step)
            logger.add_scalar("losses/policy_loss", trpo_metrics["surrogate_loss"], global_step)
            logger.add_scalar("losses/entropy", trpo_metrics["entropy"], global_step)
            logger.add_scalar("losses/old_approx_kl", trpo_metrics["old_approx_kl"], global_step)
            logger.add_scalar("losses/approx_kl", trpo_metrics["approx_kl"], global_step)
            logger.add_scalar("losses/explained_variance", explained_var, global_step)
            logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # TRPO-specific metrics
            logger.add_scalar("losses/kl_divergence", trpo_metrics["kl_divergence"], global_step)
            logger.add_scalar("trpo/surrogate_loss_improvement", trpo_metrics["surrogate_loss_improvement"], global_step)
            logger.add_scalar("trpo/step_size", trpo_metrics["step_size"], global_step)
            logger.add_scalar("trpo/backtrack_steps", trpo_metrics["backtrack_steps"], global_step)
            logger.add_scalar("trpo/cg_residual", trpo_metrics["cg_residual"], global_step)
            logger.add_scalar("trpo/max_ratio", trpo_metrics["max_ratio"], global_step)

            # Time metrics
            logger.add_scalar("time/step", global_step, global_step)
            logger.add_scalar("time/update_time", update_time, global_step)
            logger.add_scalar("time/rollout_time", rollout_time, global_step)
            logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)

        if iteration % 10 == 0:
            print(f"Iteration: {iteration}, global_step: {global_step}, "
                  f"SPS: {int(global_step / (time.time() - start_time))}, "
                  f"KL: {trpo_metrics['kl_divergence']:.4f}, "
                  f"backtrack: {trpo_metrics['backtrack_steps']}")

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
