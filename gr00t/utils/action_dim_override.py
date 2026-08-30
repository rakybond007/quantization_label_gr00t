"""Helper to override the action_dim of a loaded GR00T-N1.5 model.

When the data pipeline requires a wider action vector than the pretrained base
(e.g. dual-arm dexjoco with 44-dim actions vs. the base model's 32-dim), the
action-head's input encoder and output decoders no longer fit. This helper
preserves the pretrained backbone + DiT body and reinitialises ONLY the
action-dim-sensitive submodules at the new dimension.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from gr00t.model.action_head.flow_matching_action_head import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


def _new_decoder(ah, new_dim: int) -> CategorySpecificMLP:
    return CategorySpecificMLP(
        num_categories=ah.config.max_num_embodiments,
        input_dim=ah.hidden_size,
        hidden_dim=ah.hidden_size,
        output_dim=new_dim,
    )


def override_action_dim(model, new_dim: int) -> None:
    """Resize the action head to ``new_dim`` in-place.

    Replaces:
      - action_encoder (input projection action_dim -> hidden)
      - action_decoder (main output projection)
      - any present optional decoders: m8/m4/n8/aux multi-horizon
      - any present MoE expert decoders on fair_moe heads
    All replacements are random-initialised at ``new_dim`` and moved to the
    same device/dtype as the original module.
    """
    ah = model.action_head
    old_dim = int(ah.action_dim)
    if new_dim == old_dim:
        return

    device = next(ah.parameters()).device
    dtype = next(ah.parameters()).dtype
    print(f"[override_action_dim] {old_dim} -> {new_dim} (device={device}, dtype={dtype})")

    # 1) config bookkeeping (both model-level and action-head-level)
    ah.action_dim = new_dim
    ah.config.action_dim = new_dim
    model.action_dim = new_dim
    model.config.action_dim = new_dim
    model.config.action_head_cfg["action_dim"] = new_dim

    def _replace(name: str, builder):
        if hasattr(ah, name) and getattr(ah, name) is not None:
            new = builder().to(device=device, dtype=dtype)
            setattr(ah, name, new)
            print(f"  reinit ah.{name}")

    # 2) input encoder
    _replace(
        "action_encoder",
        lambda: MultiEmbodimentActionEncoder(
            action_dim=new_dim,
            hidden_size=ah.input_embedding_dim,
            num_embodiments=ah.config.max_num_embodiments,
        ),
    )

    # 3) main + optional decoders
    for name in [
        "action_decoder",
        "m8_action_decoder",
        "m4_action_decoder",
        "m4_full_action_decoder",
        "m2_action_decoder",
        "m2_full_action_decoder",
        "n8_action_decoder",
    ]:
        _replace(name, lambda n=name: _new_decoder(ah, new_dim))

    # 4) aux multi-horizon decoders (nn.ModuleDict)
    if hasattr(ah, "aux_action_decoders") and len(ah.aux_action_decoders) > 0:
        new_aux = nn.ModuleDict()
        for k in list(ah.aux_action_decoders.keys()):
            new_aux[k] = _new_decoder(ah, new_dim).to(device=device, dtype=dtype)
        ah.aux_action_decoders = new_aux
        print(f"  reinit ah.aux_action_decoders ({list(new_aux.keys())})")

    # 5) MoE expert decoders (fair_moe heads). ExpertHead has .proj: Linear(body_dim, action_dim).
    if hasattr(ah, "moe_experts") and ah.moe_experts is not None:
        for i, expert in enumerate(ah.moe_experts):
            if hasattr(expert, "proj"):
                body_dim = expert.proj.in_features
                new_proj = nn.Linear(body_dim, new_dim).to(device=device, dtype=dtype)
                expert.proj = new_proj
                print(f"  reinit ah.moe_experts[{i}].proj")

    # 6) refine MLPs that use 8 * action_dim (if any)
    if hasattr(ah, "_refine_mlp") and ah._refine_mlp is not None:
        # rebuild structure with new dim — only relevant if user enables refine path
        print("  [warn] refine MLP present; consider disabling refine for dual-arm or extend this helper")

    print(f"[override_action_dim] done")
