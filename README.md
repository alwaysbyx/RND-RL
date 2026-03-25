# RND-RL

Unified PPO implementation for Gymnasium and ManiSkill environments with support for discrete/continuous actions and residual/MLP architectures.

## Project Structure

```
RND-RL/
├── envs/                          # Environment wrappers
│   ├── gym_env.py                 # Gymnasium environment wrapper
│   ├── maniskill_state_env.py     # ManiSkill state-based environment
│   └── maniskill_rgb_env.py       # ManiSkill RGB-based environment
├── configs/                       # YAML configuration files
├── scripts/                       # Experiment launch scripts
├── train.py                       # Unified training script
├── eval.py                        # Evaluation script
└── rnd.py                         # RND network architectures
```

## Installation

```bash
pip install hydra-core omegaconf gymnasium torch tyro wandb tensorboard
pip install mani-skill2  # for ManiSkill environments
```

## Usage

```bash
# Run with config file
python train.py --config configs/gym.yaml

# Override parameters
python train.py --config configs/gym.yaml --config_overrides num_envs=128 learning_rate=1e-4 track=true

# Direct command-line arguments
python train.py --env_type gym --env_id HalfCheetah-v4 --num_envs 64 --track
```
