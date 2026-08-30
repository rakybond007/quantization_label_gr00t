"""Per-chunk merge gate. Conditions on pooled VLM features (and optional
extras) to predict P(merge 16->8) for the current action chunk."""

import torch
import torch.nn as nn


class GateMLP(nn.Module):
    """Predicts a single Bernoulli logit per call: merge or not.

    Input convention:
        features: (B, T, D) or (B, D). If 3D, mean-pool over T.
        extras (optional): (B, E) — additional scalars (e.g. heads-disagreement).

    Initialized so the bias of the final layer biases the prior toward
    'no merge' (the baseline-safe default).
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256,
                 extras_dim: int = 0, init_logit: float = -2.0):
        super().__init__()
        in_dim = feature_dim + extras_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_logit)

    def forward(self, features: torch.Tensor, extras: torch.Tensor = None) -> torch.Tensor:
        if features.dim() == 3:
            features = features.mean(dim=1)
        elif features.dim() == 1:
            features = features.unsqueeze(0)
        x = features
        if extras is not None:
            if extras.dim() == 1:
                extras = extras.unsqueeze(0)
            x = torch.cat([x, extras], dim=-1)
        return self.net(x).squeeze(-1)  # (B,)

    def sample(self, features, extras=None, deterministic: bool = False):
        """Returns (decision tensor int64 in {0,1}, log_prob, prob)."""
        logit = self.forward(features, extras)
        prob = torch.sigmoid(logit)
        if deterministic:
            decision = (prob > 0.5).long()
        else:
            decision = torch.bernoulli(prob).long()
        # Bernoulli log-prob of decision under prob
        log_prob = decision * torch.log(prob.clamp(min=1e-8)) \
                 + (1 - decision) * torch.log((1 - prob).clamp(min=1e-8))
        return decision, log_prob, prob
