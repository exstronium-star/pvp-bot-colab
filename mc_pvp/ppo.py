"""PPO with a hybrid action space.

Continuous (tanh-squashed Gaussian): yaw delta, pitch delta, forward, strafe.
Discrete (Bernoulli): jump, sprint, attack.

Kept deliberately small - this box is 4 CPU cores and a GTX 750, so the policy
that actually gets trained is a 128x128 MLP.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

CONT_DIM = 4   # yaw, pitch, forward, strafe
DISC_DIM = 11  # jump, sprint, attack, block, swap, eat, spl_heal/str/spd/fir/weave
LOG_STD_MIN, LOG_STD_MAX = -3.0, 0.0


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, CONT_DIM)
        self.log_std = nn.Parameter(torch.full((CONT_DIM,), -1.2))
        self.logits = nn.Linear(hidden, DISC_DIM)
        self.value = nn.Linear(hidden, 1)

        nn.init.orthogonal_(self.trunk[0].weight, np.sqrt(2))
        nn.init.orthogonal_(self.trunk[2].weight, np.sqrt(2))
        nn.init.orthogonal_(self.mu.weight, 0.01)
        nn.init.zeros_(self.mu.bias)
        nn.init.orthogonal_(self.logits.weight, 0.01)
        nn.init.zeros_(self.logits.bias)
        nn.init.orthogonal_(self.value.weight, 1.0)

    def forward(self, obs):
        h = self.trunk(obs)
        mu = torch.tanh(self.mu(h))
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std, self.logits(h), self.value(h).squeeze(-1)

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        mu, log_std, logits, value = self.forward(obs)
        std = log_std.exp()
        u = mu if deterministic else mu + torch.randn_like(mu) * std
        cont = torch.tanh(u)
        cont_logp = _gaussian_logp(u, mu, std) - torch.log(1 - cont.pow(2) + 1e-6)
        cont_logp = cont_logp.sum(-1)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        dist = torch.distributions.Bernoulli(logits=logits)
        disc = (dist.probs > 0.5).float() if deterministic else dist.sample()
        disc_logp = dist.log_prob(disc).sum(-1)

        return cont, disc, cont_logp + disc_logp, value

    def evaluate(self, obs, cont, disc):
        mu, log_std, logits, value = self.forward(obs)
        std = log_std.exp()
        u = torch.atanh(torch.clamp(cont, -0.999999, 0.999999))
        cont_logp = _gaussian_logp(u, mu, std) - torch.log(1 - cont.pow(2) + 1e-6)
        cont_logp = cont_logp.sum(-1)

        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        dist = torch.distributions.Bernoulli(logits=logits)
        disc_logp = dist.log_prob(disc).sum(-1)

        entropy = (_gaussian_entropy(std).sum(-1) + dist.entropy().sum(-1)).mean()
        return cont_logp + disc_logp, entropy, value, mu


def _gaussian_logp(u, mu, std):
    return -0.5 * (((u - mu) / std) ** 2) - std.log() - 0.5 * np.log(2 * np.pi)


def _gaussian_entropy(std):
    return std.log() + 0.5 * np.log(2 * np.pi * np.e)


class PPO:
    def __init__(self, obs_dim, lr=3e-4, clip=0.2, epochs=4, batch=512,
                 gamma=0.99, lam=0.95, ent_coef=0.01, vf_coef=0.5, hidden=128):
        self.policy = ActorCritic(obs_dim, hidden)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.clip = clip
        self.epochs = epochs
        self.batch = batch
        self.gamma = gamma
        self.lam = lam
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef

    def update(self, obs, cont, disc, logp, adv, ret, aux=None, aux_coef=0.0):
        n = obs.shape[0]
        idx = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "clip_frac": 0.0, "aux_loss": 0.0}
        batches = 0

        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch):
                mb = idx[start:start + self.batch]
                new_logp, entropy, value, mu = self.policy.evaluate(
                    obs[mb], cont[mb], disc[mb])
                ratio = torch.exp(new_logp - logp[mb])
                a = adv[mb]

                s1 = ratio * a
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * a
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = 0.5 * (value - ret[mb]).pow(2).mean()
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                aux_loss = torch.zeros((), dtype=torch.float32, device=loss.device)
                if aux is not None and aux_coef > 0.0:
                    # Regress the *mean* yaw/pitch output onto a perfect aim
                    # delta. Only the action mean is supervised, so exploration
                    # noise and the PPO ratio stay intact.
                    aux_loss = (mu[:, :2] - aux[mb]).pow(2).mean()
                    loss = loss + aux_coef * aux_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.opt.step()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["clip_frac"] += ((ratio - 1).abs() > self.clip).float().mean().item()
                stats["aux_loss"] += aux_loss.item()
                batches += 1

        return {k: v / max(batches, 1) for k, v in stats.items()}

    def snapshot(self) -> dict:
        return {k: v.detach().clone() for k, v in self.policy.state_dict().items()}

    @staticmethod
    def load(path: str, obs_dim: int) -> ActorCritic:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model" in sd and "trunk.0.weight" not in sd:
            sd = sd["model"]  # full ckpt.pt passed by mistake: use its model
        ref = ActorCritic(obs_dim)
        ref_sd = ref.state_dict()
        # same widening rule as train.py:build_net: old 46x10 / 23x3 brains
        # load into the current shape, new rows train fresh.
        own = {}
        for k, v in sd.items():
            if k in ref_sd and ref_sd[k].shape == tuple(v.shape):
                own[k] = v
        for k in ("trunk.0.weight", "logits.weight", "logits.bias"):
            if k in own or k not in sd:
                continue
            old = sd[k]
            new_shape = ref_sd[k].shape
            if k == "trunk.0.weight":
                w = torch.zeros(new_shape, dtype=old.dtype)
                w[:, :old.shape[1]] = old
                own[k] = w
            elif k == "logits.weight":
                w = torch.zeros(new_shape, dtype=old.dtype)
                w[:old.shape[0]] = old
                own[k] = w
            elif k == "logits.bias":
                b = torch.zeros(new_shape, dtype=old.dtype)
                b[:old.shape[0]] = old
                own[k] = b
        net = ActorCritic(obs_dim)
        net.load_state_dict({**ref_sd, **own}, strict=True)
        net.eval()
        return net
