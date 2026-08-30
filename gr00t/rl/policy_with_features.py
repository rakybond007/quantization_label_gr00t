"""Capture pooled VLM backbone features alongside the normal Gr00tPolicy action
output by monkey-patching `action_head.process_backbone_output` with a wrapper
that stashes the post-processed features on the policy.

This avoids duplicating Gr00tPolicy's transform/batch handling — we just call
the existing `policy.get_action(obs)` and read `policy._last_vl_features` after.
"""

from typing import Tuple

import numpy as np
import torch


def install_vl_capture(policy) -> None:
    """Idempotent: wraps action_head.process_backbone_output so each call also
    stores its result as `policy._last_vl_features` (a CPU float tensor)."""
    if getattr(policy, "_vl_capture_installed", False):
        return
    ah = policy.model.action_head
    orig = ah.process_backbone_output

    def wrapped(backbone_output):
        out = orig(backbone_output)
        # backbone_features: (B, T, D)
        try:
            policy._last_vl_features = out.backbone_features.detach().float().cpu()
        except Exception:
            policy._last_vl_features = None
        return out

    ah.process_backbone_output = wrapped
    policy._vl_capture_installed = True


def policy_forward(policy, observations: dict) -> Tuple[dict, torch.Tensor]:
    """Run policy.get_action and return (unnormalised_action_dict, vl_features).

    vl_features shape: (B, T, D) on CPU. If single-obs (no batch), B=1.
    """
    install_vl_capture(policy)
    policy._last_vl_features = None
    action = policy.get_action(observations)
    vl = policy._last_vl_features
    if vl is None:
        raise RuntimeError("VL features not captured — check process_backbone_output hook.")
    return action, vl
