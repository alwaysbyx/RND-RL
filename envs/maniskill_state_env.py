"""ManiSkill state-based environment wrapper."""
import gymnasium as gym
import torch
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


def make_maniskill_state_env(
    env_id: str,
    num_envs: int,
    num_eval_envs: int,
    seed: int,
    control_mode: str = "pd_joint_delta_pos",
    reconfiguration_freq: int = None,
    eval_reconfiguration_freq: int = 1,
    partial_reset: bool = True,
    eval_partial_reset: bool = False,
    capture_video: bool = False,
    save_trajectory: bool = False,
    run_name: str = None,
    save_train_video_freq: int = None,
    num_steps: int = 50,
    num_eval_steps: int = 50,
    evaluate: bool = False,
):
    """Create ManiSkill state-based vectorized environment."""
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
    if control_mode is not None:
        env_kwargs["control_mode"] = control_mode

    envs = gym.make(
        env_id,
        num_envs=num_envs if not evaluate else 1,
        reconfiguration_freq=reconfiguration_freq,
        **env_kwargs
    )
    eval_envs = gym.make(
        env_id,
        num_envs=num_eval_envs,
        reconfiguration_freq=eval_reconfiguration_freq,
        **env_kwargs
    )

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    if capture_video or save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos" if run_name else "runs/videos"
        if save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // num_steps) % save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_name}/train_videos" if run_name else "runs/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=num_steps,
                video_fps=30,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=save_trajectory,
            save_video=capture_video,
            trajectory_name="trajectory",
            max_steps_per_video=num_eval_steps,
            video_fps=30,
        )

    envs = ManiSkillVectorEnv(envs, num_envs, ignore_terminations=not partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(
        eval_envs,
        num_eval_envs,
        ignore_terminations=not eval_partial_reset,
        record_metrics=True,
    )

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    return envs, eval_envs, max_episode_steps, env_kwargs


def get_maniskill_state_obs_action_dims(envs):
    """Get observation and action dimensions for ManiSkill state environments."""
    n_act = int(torch.prod(torch.tensor(envs.single_action_space.shape)))
    n_obs = int(torch.prod(torch.tensor(envs.single_observation_space.shape)))
    return n_obs, n_act
