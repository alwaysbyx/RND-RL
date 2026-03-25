"""ManiSkill RGB-based environment wrapper."""
import gymnasium as gym
import numpy as np
import torch
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


class DictArray(object):
    """Dictionary-based array for storing nested observations."""
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {
            k: v[index] for k, v in self.data.items()
        }

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k, v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)

    def numel(self):
        """Get total number of elements."""
        return sum(v.numel() if hasattr(v, 'numel') else len(v) for v in self.data.values())


def make_maniskill_rgb_env(
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
    include_state: bool = True,
    render_mode: str = "all",
):
    """Create ManiSkill RGB-based vectorized environment."""
    env_kwargs = dict(obs_mode="rgb", render_mode=render_mode, sim_backend="physx_cuda")
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

    # Flatten RGBD observations
    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=include_state)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=include_state)

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


def get_maniskill_rgb_obs_action_dims(envs, sample_obs):
    """Get observation and action dimensions for ManiSkill RGB environments."""
    n_act = int(np.prod(envs.unwrapped.single_action_space.shape))
    # For RGB, we need to use the feature extractor to get the latent size
    # This will be computed in the Agent class
    return None, n_act  # obs_dim will be computed from feature extractor
