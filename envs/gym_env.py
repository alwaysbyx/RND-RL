"""Gymnasium environment wrapper."""
import gymnasium as gym
import numpy as np
import torch


def make_gym_env(env_id: str, num_envs: int, seed: int, capture_video: bool = False, run_name: str = None, gamma: float = 0.99):
    """Create a Gymnasium vectorized environment with standard wrappers."""
    def thunk():
        if capture_video and run_name is not None:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    envs = gym.vector.SyncVectorEnv([thunk for _ in range(num_envs)])
    return envs


def get_gym_obs_action_dims(envs):
    """Get observation and action dimensions for Gym environments."""
    obs_dim = int(np.array(envs.single_observation_space.shape).prod())
    act_dim = int(np.prod(envs.single_action_space.shape))
    return obs_dim, act_dim
