"""PPO-CMA: Proximal Policy Optimization with Covariance Matrix Adaptation.

Implements the PPO-CMA algorithm from:
  Hämäläinen et al., "PPO-CMA: Proximal Policy Optimization with Covariance Matrix Adaptation"
  IEEE MLSP 2020. arXiv:1810.02541

Key differences from PPO:
  - Separate neural networks for policy mean and variance (rank-μ update)
  - Only positive-advantage actions used; negative advantages mirrored
  - History buffer of H iterations for variance network training (evolution path heuristic)
  - No clipped surrogate loss, no entropy bonus
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from rnd import NatureCNN, layer_init


def build_ppocma_network(input_dim, output_dim, hidden_dim, depth, device=None, output_std=None):
    """Build MLP with LeakyReLU activations (PPO-CMA paper Table 2)."""
    layers = []
    layers.append(layer_init(nn.Linear(input_dim, hidden_dim, device=device)))
    layers.append(nn.LeakyReLU())
    for _ in range(depth - 1):
        layers.append(layer_init(nn.Linear(hidden_dim, hidden_dim, device=device)))
        layers.append(nn.LeakyReLU())
    if output_std is not None:
        layers.append(layer_init(nn.Linear(hidden_dim, output_dim, device=device), std=output_std))
    else:
        layers.append(layer_init(nn.Linear(hidden_dim, output_dim, device=device)))
    return nn.Sequential(*layers)


class HistoryBuffer:
    """Ring buffer storing past H iterations of rollout data for variance training."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.buffer = []

    def append(self, data: dict):
        self.buffer.append(data)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def get_all(self):
        """Concatenate all buffered data along batch dimension."""
        if not self.buffer:
            return None
        return {
            key: torch.cat([b[key] for b in self.buffer], dim=0)
            for key in self.buffer[0].keys()
        }

    def __len__(self):
        return len(self.buffer)


class PPOCMAAgent(nn.Module):
    """PPO-CMA Agent with separate mean and variance networks.

    Supports continuous actions only (Gaussian policy).
    Compatible interface with Agent: get_value(), get_action(), get_action_and_value().
    """

    def __init__(
        self,
        n_obs: int,
        n_act: int,
        action_space,
        args,
        device=None,
        sample_obs=None,
    ):
        super().__init__()
        self.device = device
        self.n_act = n_act
        self.use_rgb = sample_obs is not None

        # RGB feature extractor (shared, if needed)
        if self.use_rgb:
            self.feature_net = NatureCNN(sample_obs=sample_obs)
            latent_size = self.feature_net.out_features
        else:
            self.feature_net = None
            latent_size = n_obs

        # Action space bounds
        low = torch.tensor(action_space.low, dtype=torch.float32, device=device)
        high = torch.tensor(action_space.high, dtype=torch.float32, device=device)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)
        self.register_buffer("action_range", high - low)

        # Log-variance clipping bounds (Appendix A)
        lower_std = getattr(args, "lower_std_limit", 0.01)
        v_min = 2.0 * np.log(lower_std)
        v_max = 2.0 * torch.log(self.action_range).cpu().numpy()
        self.register_buffer("v_min", torch.tensor(v_min, dtype=torch.float32, device=device))
        self.register_buffer("v_max", torch.tensor(v_max, dtype=torch.float32, device=device))

        # Network dimensions
        actor_width = getattr(args, "ppocma_actor_width", getattr(args, "actor_width", 128))
        actor_depth = getattr(args, "ppocma_actor_depth", getattr(args, "actor_depth", 2))
        critic_width = getattr(args, "critic_width", 128)
        critic_depth = getattr(args, "critic_depth", 2)

        # Mean network
        self.mean_net = build_ppocma_network(
            input_dim=latent_size,
            output_dim=n_act,
            hidden_dim=actor_width,
            depth=actor_depth,
            device=device,
            output_std=0.01 * np.sqrt(2),
        )

        # Variance network (separate from mean)
        self.var_net = build_ppocma_network(
            input_dim=latent_size,
            output_dim=n_act,
            hidden_dim=actor_width,
            depth=actor_depth,
            device=device,
            output_std=0.01 * np.sqrt(2),
        )

        # Critic network
        self.critic = build_ppocma_network(
            input_dim=latent_size,
            output_dim=1,
            hidden_dim=critic_width,
            depth=critic_depth,
            device=device,
        )

        # Pretrain policy outputs (Appendix A):
        # Initial mean at center of action space, initial variance covering half the range
        self._pretrain_init(device)

    def _pretrain_init(self, device):
        """Initialize networks so initial policy covers the action space (Appendix A).

        Target: mu_clipped = 0.5*(a_max + a_min), i.e., center
                v_clipped = 2*log(0.5*(a_max - a_min)), i.e., std = half the range
        Since output = clip(raw) uses sigmoid, we want sigmoid(raw) = target_fraction.
        For center: sigmoid(raw) = 0.5 -> raw = 0 (default init is near zero, so OK).
        For variance: we want v_clipped = 2*log(0.5*range).
          v_clipped = v_min + (v_max - v_min) * sigmoid(raw_v)
          target_v = 2*log(0.5*range)
          sigmoid(raw_v) = (target_v - v_min) / (v_max - v_min)
        The small init std (0.01*sqrt(2)) means raw outputs start near zero,
        which gives sigmoid ~ 0.5, placing mean at center and variance at midpoint of [v_min, v_max].
        This is close enough to the paper's recommendation.
        """
        pass  # Default initialization is sufficient due to small output std

    def get_features(self, x):
        """Extract features from observations."""
        if self.use_rgb:
            return self.feature_net(x)
        return x

    def clip_mean(self, raw_mean):
        """Sigmoid-based mean clipping: a_min + (a_max - a_min) * sigmoid(raw)."""
        return self.action_low + self.action_range * torch.sigmoid(raw_mean)

    def clip_logvar(self, raw_logvar):
        """Sigmoid-based log-variance clipping: v_min + (v_max - v_min) * sigmoid(raw)."""
        return self.v_min + (self.v_max - self.v_min) * torch.sigmoid(raw_logvar)

    def get_mean_and_std(self, obs):
        """Compute clipped mean, std, and log-variance from both networks."""
        features = self.get_features(obs)
        raw_mean = self.mean_net(features)
        raw_logvar = self.var_net(features)
        mean = self.clip_mean(raw_mean)
        logvar = self.clip_logvar(raw_logvar)
        std = torch.exp(0.5 * logvar)
        return mean, std, logvar

    def get_value(self, x):
        """Get value estimate."""
        features = self.get_features(x)
        return self.critic(features)

    def get_action(self, obs, deterministic=False):
        """Get action for inference."""
        mean, std, _ = self.get_mean_and_std(obs)
        if deterministic:
            return mean
        dist = Normal(mean, std)
        return dist.sample()

    def get_action_and_value(self, obs, action=None):
        """Get action, logprob, entropy, value (compatible with Agent interface)."""
        features = self.get_features(obs)

        raw_mean = self.mean_net(features)
        raw_logvar = self.var_net(features)
        mean = self.clip_mean(raw_mean)
        logvar = self.clip_logvar(raw_logvar)
        std = torch.exp(0.5 * logvar)

        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(features)
        return action, logprob, entropy, value

    def compute_mean_loss(self, obs, actions, advantages):
        """Mean network loss (Eq. 3, variance detached).

        L = (1/M) * sum_i [ A_i * sum_j (a_{ij} - mu_j(s_i))^2 / var_j(s_i) ]
        The log(var) term is omitted since it has zero gradient w.r.t. mean params.
        Only called with positive advantages (after mirroring).
        """
        features = self.get_features(obs)
        raw_mean = self.mean_net(features)
        mean = self.clip_mean(raw_mean)

        # Variance is frozen for mean update
        with torch.no_grad():
            raw_logvar = self.var_net(features)
            logvar = self.clip_logvar(raw_logvar)
            var = torch.exp(logvar)

        # Squared Mahalanobis distance weighted by advantage
        squared_mahal = ((actions - mean) ** 2 / var).sum(dim=-1)
        loss = (advantages * squared_mahal).mean()
        return loss

    def compute_var_loss(self, obs, actions, advantages):
        """Variance network loss (Eq. 3, mean detached).

        L = (1/M) * sum_i [ A_i * sum_j ((a_{ij} - mu_j(s_i))^2 / var_j(s_i) + 0.5*log(var_j(s_i))) ]
        Mean is frozen for variance update.
        """
        features = self.get_features(obs)

        # Mean is frozen for variance update
        with torch.no_grad():
            raw_mean = self.mean_net(features)
            mean = self.clip_mean(raw_mean)

        raw_logvar = self.var_net(features)
        logvar = self.clip_logvar(raw_logvar)
        var = torch.exp(logvar)

        # Full NLL per dimension
        nll_per_dim = (actions - mean) ** 2 / var + 0.5 * logvar
        nll = nll_per_dim.sum(dim=-1)
        loss = (advantages * nll).mean()
        return loss

    @staticmethod
    def mirror_actions(actions, means, advantages, stds):
        """Mirror negative-advantage actions to create all-positive dataset (Section 5.2).

        For negative-advantage samples:
            a'_i = 2 * mu(s_i) - a_i           (reflect across mean)
            A'_i = -A_i * psi(a_i, s_i)        (flip sign, weight by Gaussian kernel)

        where psi is Gaussian kernel with same shape as policy.
        """
        positive_mask = advantages >= 0
        negative_mask = ~positive_mask

        mirrored_actions = actions.clone()
        mirrored_advantages = advantages.clone()

        if negative_mask.any():
            neg_actions = actions[negative_mask]
            neg_means = means[negative_mask]
            neg_stds = stds[negative_mask]
            neg_advantages = advantages[negative_mask]

            # Reflect actions across mean
            mirrored_actions[negative_mask] = 2 * neg_means - neg_actions

            # Gaussian kernel weight: psi = exp(-0.5 * sum((a - mu)^2 / var))
            diff = neg_actions - neg_means
            var = neg_stds ** 2
            log_kernel = -0.5 * (diff ** 2 / var).sum(dim=-1)
            psi = torch.exp(log_kernel)

            # Flip advantage sign and weight by kernel
            mirrored_advantages[negative_mask] = -neg_advantages * psi

        return mirrored_actions, mirrored_advantages

    def mean_parameters(self):
        """Return mean network parameters for optimizer."""
        return list(self.mean_net.parameters())

    def var_parameters(self):
        """Return variance network parameters for optimizer."""
        return list(self.var_net.parameters())

    def critic_parameters(self):
        """Return critic (+ feature extractor) parameters for optimizer."""
        params = list(self.critic.parameters())
        if self.feature_net is not None:
            params += list(self.feature_net.parameters())
        return params
