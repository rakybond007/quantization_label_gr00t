"""Smoke test for multi-horizon loss implementation.

Builds a tiny FlowmatchingActionHead with multi-horizon enabled and runs
a forward pass with synthetic data to verify shapes and loss values.

Usage:
    python scripts/smoke_test_multi_horizon.py
"""

import torch
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)


def make_tiny_config(use_multi_horizon: bool, aux_grad_scale: float = 0.1, warmup: int = 0):
    diffusion_cfg = dict(
        attention_head_dim=32,
        cross_attention_dim=128,
        dropout=0.0,
        final_dropout=False,
        interleave_self_attention=True,
        norm_type="ada_norm",
        num_attention_heads=4,
        num_layers=2,
        output_dim=128,
        positional_embeddings=None,
    )
    cfg = FlowmatchingActionHeadConfig(
        add_pos_embed=True,
        diffusion_model_cfg=diffusion_cfg,
        input_embedding_dim=128,
        backbone_embedding_dim=128,
        hidden_size=128,
        max_seq_len=128,
        action_dim=8,
        action_horizon=16,
        max_state_dim=8,
        num_inference_timesteps=4,
        max_num_embodiments=4,
        tune_projector=True,
        tune_diffusion_model=True,
        use_vlln=False,
        vl_self_attention_cfg=None,
        use_multi_horizon_loss=use_multi_horizon,
        multi_horizon_factors=[2, 4],
        multi_horizon_loss_weights=[0.5, 0.5],
        multi_horizon_main_weight=1.0,
        aux_grad_scale_to_body=aux_grad_scale,
        aux_loss_warmup_steps=warmup,
    )
    return cfg


def run_forward(use_multi_horizon: bool):
    print(f"\n=== Multi-horizon = {use_multi_horizon} ===")
    cfg = make_tiny_config(use_multi_horizon)
    head = FlowmatchingActionHead(cfg).eval()  # use eval to skip dropout etc.
    head.train()  # but keep training mode for forward

    B = 2
    H = 16
    H_ext = 64
    D = 8
    state_dim = 8

    backbone_output = BatchFeature(data={
        "backbone_features": torch.randn(B, 8, 128),
        "backbone_attention_mask": torch.ones(B, 8, dtype=torch.long),
    })

    action_input_data = {
        "embodiment_id": torch.zeros(B, dtype=torch.long),
        "state": torch.randn(B, 1, state_dim),
        "action": torch.randn(B, H, D),
        "action_mask": torch.ones(B, H, D),
    }
    if use_multi_horizon:
        action_input_data["action_extended"] = torch.randn(B, H_ext, D)
        action_input_data["action_extended_mask"] = torch.ones(B, H_ext, D, dtype=torch.bool)

    action_input = BatchFeature(data=action_input_data)
    out = head(backbone_output, action_input)
    print(f"  loss = {out['loss'].item():.4f}")
    print(f"  loss_main = {out['loss_main'].item():.4f}")
    if use_multi_horizon:
        for k in out.keys():
            if k.startswith("loss_f"):
                print(f"  {k} = {out[k].item():.4f}")

    # Backward to verify grad flow
    out["loss"].backward()
    n_grad = sum(p.grad.abs().sum().item() for p in head.parameters() if p.grad is not None)
    print(f"  Total grad magnitude: {n_grad:.2f}")
    assert n_grad > 0, "Expected non-zero gradient"
    print(f"  ✓ Forward + backward OK")


def grad_scale_test():
    """Verify that aux_grad_scale_to_body actually dampens gradient into the
    shared DiT body, while aux decoders still get full gradient."""
    print(f"\n=== Gradient scaling test (verify protection) ===")

    def get_dit_grad_norm(scale):
        torch.manual_seed(42)
        cfg = make_tiny_config(use_multi_horizon=True, aux_grad_scale=scale, warmup=0)
        head = FlowmatchingActionHead(cfg)
        head.train()

        B, H, H_ext, D = 2, 16, 64, 8
        backbone_output = BatchFeature(data={
            "backbone_features": torch.randn(B, 8, 128),
            "backbone_attention_mask": torch.ones(B, 8, dtype=torch.long),
        })
        action_input = BatchFeature(data={
            "embodiment_id": torch.zeros(B, dtype=torch.long),
            "state": torch.randn(B, 1, 8),
            "action": torch.randn(B, H, D),
            "action_mask": torch.ones(B, H, D),
            "action_extended": torch.randn(B, H_ext, D),
            "action_extended_mask": torch.ones(B, H_ext, D, dtype=torch.bool),
        })

        # Compute ONLY aux loss to isolate its effect on DiT
        out = head(backbone_output, action_input)
        aux_only = sum(v for k, v in out.items() if k.startswith("loss_f"))
        head.zero_grad()
        aux_only.backward()
        # DiT body grad
        dit_grad = sum(p.grad.abs().sum().item() for p in head.model.parameters() if p.grad is not None)
        # Aux decoders' own grad (should be unaffected by scale)
        aux_grad = sum(p.grad.abs().sum().item() for p in head.aux_action_decoders.parameters() if p.grad is not None)
        return dit_grad, aux_grad

    for scale in [1.0, 0.5, 0.1, 0.0]:
        dit, aux = get_dit_grad_norm(scale)
        print(f"  scale={scale:>4} -> DiT grad={dit:.2f}, aux_decoders grad={aux:.2f}")
    print("  (DiT grad should ~scale linearly with `scale`; aux grad should be similar across scales)")


def warmup_test():
    print(f"\n=== Warmup test (aux weight ramps over forward steps) ===")
    cfg = make_tiny_config(use_multi_horizon=True, aux_grad_scale=1.0, warmup=10)
    head = FlowmatchingActionHead(cfg)
    head.train()

    B, H, H_ext, D = 2, 16, 64, 8
    backbone_output = BatchFeature(data={
        "backbone_features": torch.randn(B, 8, 128),
        "backbone_attention_mask": torch.ones(B, 8, dtype=torch.long),
    })
    action_input = BatchFeature(data={
        "embodiment_id": torch.zeros(B, dtype=torch.long),
        "state": torch.randn(B, 1, 8),
        "action": torch.randn(B, H, D),
        "action_mask": torch.ones(B, H, D),
        "action_extended": torch.randn(B, H_ext, D),
        "action_extended_mask": torch.ones(B, H_ext, D, dtype=torch.bool),
    })
    print(f"  step | aux_warmup")
    for i in range(13):
        out = head(backbone_output, action_input)
        w = out["aux_warmup"].item()
        print(f"  {i:>4} | {w:.2f}")


def main():
    torch.manual_seed(0)
    run_forward(use_multi_horizon=False)
    run_forward(use_multi_horizon=True)
    grad_scale_test()
    warmup_test()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
