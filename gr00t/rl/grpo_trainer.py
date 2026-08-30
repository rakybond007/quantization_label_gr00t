"""Group-Relative Policy Optimization (GRPO) for the merge gate.

Per training iteration:
  - Sample K episodes from the same env seed (group).
  - Compute group-relative advantage on episode-level reward.
  - PPO-clip update on per-chunk gate log-probs, with KL anchor + entropy bonus.

Episode reward:
  R = +1 if success else -1
    + alpha_len * (1 - total_env_steps / max_env_steps)
    + alpha_merge * (num_merges / max(num_chunks, 1))
"""

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def episode_reward(ep, max_env_steps=1500, alpha_len=0.1, alpha_merge=0.05):
    """Combine binary success + length + merge-bonus into a scalar."""
    base = 1.0 if ep.success else -1.0
    len_bonus = alpha_len * (1.0 - ep.total_env_steps / max(max_env_steps, 1))
    n_chunks = len(ep.chunks)
    merge_ratio = sum(c.decision for c in ep.chunks) / max(n_chunks, 1)
    merge_bonus = alpha_merge * merge_ratio
    return base + len_bonus + merge_bonus


@dataclass
class GRPOLoss:
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    entropy: torch.Tensor
    total: torch.Tensor


def grpo_loss(gate, episodes, advantages_per_ep,
              clip_eps: float = 0.2, kl_weight: float = 0.05,
              entropy_weight: float = 0.01, prior_logit: float = -2.0,
              device: torch.device = None) -> GRPOLoss:
    """Compute the per-step PPO-clip loss + KL anchor + entropy bonus.

    advantages_per_ep: list of scalars (one per episode, same len as episodes).

    Each chunk's old log-prob and the group's per-episode advantage define
    its contribution. Re-evaluates the gate at the recorded vl_features
    to get fresh log-probs (with grad).
    """
    pol_terms = []
    kl_terms = []
    ent_terms = []
    for ep, A in zip(episodes, advantages_per_ep):
        if not ep.chunks:
            continue
        feats = torch.stack([c.vl_features for c in ep.chunks]).to(device)  # (N, T, D)
        old_lp = torch.tensor([c.log_prob for c in ep.chunks], device=device)
        decisions = torch.tensor([c.decision for c in ep.chunks], device=device)

        new_logit = gate(feats)
        new_prob = torch.sigmoid(new_logit)
        new_lp = decisions * torch.log(new_prob.clamp(min=1e-8)) \
               + (1 - decisions) * torch.log((1 - new_prob).clamp(min=1e-8))

        ratio = torch.exp(new_lp - old_lp)
        adv = torch.full_like(new_lp, A)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        policy_t = -torch.min(unclipped, clipped).mean()
        pol_terms.append(policy_t)

        # KL anchor: pull current Bernoulli toward the prior Bernoulli(sigmoid(prior_logit))
        prior_p = torch.sigmoid(torch.tensor(prior_logit, device=device))
        kl_t = (new_prob * (torch.log(new_prob.clamp(min=1e-8)) - torch.log(prior_p.clamp(min=1e-8)))
                + (1 - new_prob) * (torch.log((1 - new_prob).clamp(min=1e-8))
                                     - torch.log((1 - prior_p).clamp(min=1e-8)))).mean()
        kl_terms.append(kl_t)

        # Entropy of Bernoulli
        ent_t = -(new_prob * torch.log(new_prob.clamp(min=1e-8))
                   + (1 - new_prob) * torch.log((1 - new_prob).clamp(min=1e-8))).mean()
        ent_terms.append(ent_t)

    if not pol_terms:
        zero = torch.zeros((), device=device, requires_grad=True)
        return GRPOLoss(zero, zero, zero, zero)

    pol = torch.stack(pol_terms).mean()
    kl = torch.stack(kl_terms).mean()
    ent = torch.stack(ent_terms).mean()
    total = pol + kl_weight * kl - entropy_weight * ent
    return GRPOLoss(pol, kl, ent, total)


def group_advantages(rewards: List[float]) -> List[float]:
    """Group-relative advantage: (R_i - mean) / (std + eps)."""
    import numpy as np
    arr = np.asarray(rewards, dtype=float)
    if arr.size == 0:
        return []
    mu = arr.mean()
    sd = arr.std()
    if sd < 1e-6:
        return [0.0] * len(arr)
    return ((arr - mu) / sd).tolist()
