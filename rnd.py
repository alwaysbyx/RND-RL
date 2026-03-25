"""Neural network architectures for PPO agents."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize a layer with orthogonal weights and constant bias."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def build_network(input_dim, output_dim, hidden_dim, depth, device=None, output_std=None):
    """Build MLP with Tanh activations, used for both actor and critic."""
    layers = []
    layers.append(layer_init(nn.Linear(input_dim, hidden_dim, device=device)))
    layers.append(nn.Tanh())
    for _ in range(depth - 1):
        layers.append(layer_init(nn.Linear(hidden_dim, hidden_dim, device=device)))
        layers.append(nn.Tanh())
    if output_std is not None:
        layers.append(layer_init(nn.Linear(hidden_dim, output_dim, device=device), std=output_std))
    else:
        layers.append(layer_init(nn.Linear(hidden_dim, output_dim, device=device)))
    return nn.Sequential(*layers)


class ResidualBlockFFN(nn.Module):
    """Residual FFN block used for Simba-style residual actor."""

    def __init__(self, hidden_dim, dtype=torch.float32, device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        self.norm = nn.LayerNorm(hidden_dim, dtype=dtype)
        if device is not None:
            self.norm = self.norm.to(device)

        self.ffn_up = nn.Linear(hidden_dim, hidden_dim * 4, dtype=dtype, device=device)
        self.ffn_down = nn.Linear(hidden_dim * 4, hidden_dim, dtype=dtype, device=device)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.ffn_up.weight, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.ffn_down.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.ffn_up.bias)
        nn.init.zeros_(self.ffn_down.bias)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.ffn_up(x)
        x = nn.functional.relu(x)
        x = self.ffn_down(x)
        return x + residual


class ActorPolicy(nn.Module):
    """Residual-block actor as in ppo_scale (Simba-style)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_blocks: int,
        discrete_action: bool = True,
        dtype=torch.float32,
        action_scale: float = 1.0,
        device=None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dtype = dtype
        self.action_scale = action_scale
        self.discrete_action = discrete_action

        self.projection = nn.Linear(obs_dim, hidden_dim, dtype=dtype, device=device)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlockFFN(hidden_dim, dtype=dtype, device=device) for _ in range(num_blocks)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim, dtype=dtype)
        if device is not None:
            self.final_norm = self.final_norm.to(device)
        self.action_head = nn.Linear(hidden_dim, action_dim, dtype=dtype, device=device)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.projection.weight, gain=1.0)
        nn.init.zeros_(self.projection.bias)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, x):
        x = self.projection(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.final_norm(x)
        action_logits = self.action_head(x)
        if not self.discrete_action:
            action_logits = torch.tanh(action_logits) * self.action_scale
        return action_logits


class NatureCNN(nn.Module):
    """CNN feature extractor for RGB observations."""
    def __init__(self, sample_obs):
        super().__init__()

        extractors = {}

        self.out_features = 0
        feature_size = 256
        in_channels = sample_obs["rgb"].shape[-1]
        image_size = (sample_obs["rgb"].shape[1], sample_obs["rgb"].shape[2])

        # NatureCNN architecture to process images
        cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
                padding=0,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Figure out dimensions after flattening
        with torch.no_grad():
            n_flatten = cnn(sample_obs["rgb"].float().permute(0, 3, 1, 2).cpu()).shape[1]
            fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
        extractors["rgb"] = nn.Sequential(cnn, fc)
        self.out_features += feature_size

        if "state" in sample_obs:
            # For state data we simply pass it through a single linear layer
            state_size = sample_obs["state"].shape[-1]
            extractors["state"] = nn.Linear(state_size, 256)
            self.out_features += 256

        self.extractors = nn.ModuleDict(extractors)

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            obs = observations[key]
            if key == "rgb":
                obs = obs.float().permute(0, 3, 1, 2)
                obs = obs / 255
            encoded_tensor_list.append(extractor(obs))
        return torch.cat(encoded_tensor_list, dim=1)


class Agent(nn.Module):
    """Agent supporting continuous/discrete actions and residual/MLP architectures."""

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        action_space,
        args,
        device=None,
        sample_obs=None,  # For RGB environments
    ):
        super().__init__()
        self.discrete_action = args.discrete_action
        self.use_residual_blocks = args.use_residual_blocks
        self.device = device
        self.n_act = n_act
        self.use_rgb = sample_obs is not None

        # RGB feature extractor (if needed)
        if self.use_rgb:
            self.feature_net = NatureCNN(sample_obs=sample_obs)
            latent_size = self.feature_net.out_features
        else:
            self.feature_net = None
            latent_size = n_obs

        # Critic
        self.critic = build_network(
            input_dim=latent_size,
            output_dim=1,
            hidden_dim=args.critic_width,
            depth=args.critic_depth,
            device=device,
        )

        if self.discrete_action:
            # Discretize each action dimension into bins
            self.num_bins = args.num_bins
            low = action_space.low
            high = action_space.high
            uniform_action_bins = torch.linspace(low[0], high[0], self.num_bins, device=device)
            self.action_bins = uniform_action_bins

            act_dim = n_act * self.num_bins
            if self.use_residual_blocks:
                self.actor_mean = ActorPolicy(
                    obs_dim=latent_size,
                    action_dim=act_dim,
                    hidden_dim=args.actor_width,
                    num_blocks=args.actor_depth,
                    discrete_action=True,
                    action_scale=1.0,
                    device=device,
                )
            else:
                self.actor_mean = build_network(
                    input_dim=latent_size,
                    output_dim=act_dim,
                    hidden_dim=args.actor_width,
                    depth=args.actor_depth,
                    device=device,
                    output_std=0.01 * np.sqrt(2),
                )
        else:
            # Continuous actions (tanh * scale)
            low = action_space.low
            self.action_scale = float(abs(low[0])) if low is not None else 1.0

            if self.use_residual_blocks:
                self.actor_mean = ActorPolicy(
                    obs_dim=latent_size,
                    action_dim=n_act,
                    hidden_dim=args.actor_width,
                    num_blocks=args.actor_depth,
                    discrete_action=False,
                    action_scale=self.action_scale,
                    device=device,
                )
            else:
                self.actor_mean = build_network(
                    input_dim=latent_size,
                    output_dim=n_act,
                    hidden_dim=args.actor_width,
                    depth=args.actor_depth,
                    device=device,
                    output_std=0.01 * np.sqrt(2),
                )
            self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))

    def get_features(self, x):
        """Extract features from observations (for RGB)."""
        if self.use_rgb:
            return self.feature_net(x)
        return x

    def get_value(self, x):
        """Get value estimate."""
        if self.use_rgb:
            x = self.feature_net(x)
        return self.critic(x)

    def action_to_index(self, action_continuous: torch.Tensor, action_bins: torch.Tensor):
        """Convert continuous action to discrete bin index."""
        diff = action_continuous.unsqueeze(-1) - action_bins.view(1, 1, -1)
        return diff.abs().argmin(dim=-1)

    def _discrete_action_and_stats(self, obs, action=None):
        """Get discrete action and statistics."""
        if self.use_rgb:
            obs = self.feature_net(obs)
        logits = self.actor_mean(obs).reshape(-1, self.n_act, self.num_bins)
        probs_logits = torch.softmax(logits, dim=-1)
        dist = Categorical(probs=probs_logits)

        if action is None:
            actions_idx = dist.sample()
            action_bins = self.action_bins.to(actions_idx.device)
            action = action_bins[actions_idx]
        else:
            action_bins = self.action_bins.to(action.device)
            actions_idx = self.action_to_index(action, action_bins)

        logprob = dist.log_prob(actions_idx).sum(1)
        entropy = dist.entropy().sum(1)
        value = self.critic(obs)
        return action, logprob, entropy, value

    def _continuous_action_and_stats(self, obs, action=None):
        """Get continuous action and statistics."""
        if self.use_rgb:
            obs = self.feature_net(obs)
        if self.use_residual_blocks:
            action_mean = self.actor_mean(obs)
        else:
            action_mean_raw = self.actor_mean(obs)
            action_mean = torch.tanh(action_mean_raw) * self.action_scale
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        dist = Normal(action_mean, action_std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(1)
        entropy = dist.entropy().sum(1)
        value = self.critic(obs)
        return action, logprob, entropy, value

    def get_action(self, obs, deterministic: bool = False):
        """Get action (for inference)."""
        if self.discrete_action:
            if self.use_rgb:
                obs = self.feature_net(obs)
            logits = self.actor_mean(obs).reshape(-1, self.n_act, self.num_bins)
            probs_logits = torch.softmax(logits, dim=-1)
            dist = Categorical(probs=probs_logits)
            if deterministic:
                actions_idx = torch.argmax(probs_logits, dim=-1)
            else:
                actions_idx = dist.sample()
            action_bins = self.action_bins.to(actions_idx.device)
            action = action_bins[actions_idx]
            return action
        else:
            if self.use_rgb:
                obs = self.feature_net(obs)
            if self.use_residual_blocks:
                action_mean = self.actor_mean(obs)
            else:
                action_mean_raw = self.actor_mean(obs)
                action_mean = torch.tanh(action_mean_raw) * self.action_scale
            if deterministic:
                return action_mean
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            dist = Normal(action_mean, action_std)
            return dist.sample()

    def get_action_and_value(self, obs, action=None):
        """Get action and value (for training)."""
        if self.discrete_action:
            return self._discrete_action_and_stats(obs, action)
        else:
            return self._continuous_action_and_stats(obs, action)
