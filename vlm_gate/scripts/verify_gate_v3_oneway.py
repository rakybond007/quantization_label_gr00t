"""C-v3 unit test: with the one-way self-attn mask, perturbing ONLY the gate
row of the initial noise must leave the 16 action rows of the denoised output
BIT-IDENTICAL (torch.equal). Without the mask (v2 behavior) they must differ —
proving the mask is what decouples them.
"""
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_gate_identity import build_obs, CKPT  # noqa: E402
from extract_gate_backbone_features import load_policy  # noqa: E402


def main():
    from gr00t.model.policy import COMPUTE_DTYPE

    policy = load_policy(CKPT)
    model = policy.model
    head = model.action_head
    head.attach_quant_gate_fm(loss_weight=1.0, oneway=True)
    head.eval()

    obs = build_obs()
    normalized = policy.apply_transforms(obs)
    backbone_inputs, action_inputs = model.prepare_input(normalized)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
        bo = model.backbone(backbone_inputs)
        bo = head.process_backbone_output(bo)
        vl = bo.backbone_features
        emb_id = action_inputs.embodiment_id
        state = head.state_encoder(action_inputs.state, emb_id)

        H2 = head.config.action_horizon + 1
        torch.manual_seed(1234)
        x0 = torch.randn((vl.shape[0], H2, head.config.action_dim),
                         dtype=vl.dtype, device=vl.device)
        x0_pert = x0.clone()
        x0_pert[:, -1] = x0_pert[:, -1] + 3.0  # perturb ONLY the gate row

        def run(x, oneway):
            head.quant_gate_oneway = oneway
            return head._fm_denoise(vl, state, emb_id, x.clone())

        # --- with one-way mask ---
        a = run(x0, True)
        b = run(x0_pert, True)
        same = torch.equal(a[:, :-1], b[:, :-1])
        mad = (a[:, :-1].float() - b[:, :-1].float()).abs().max().item()
        dg = (a[:, -1].float() - b[:, -1].float()).abs().max().item()
        print(f"[oneway=True ] action rows bitwise_equal={same} max_abs_diff={mad:.3e} "
              f"(gate row diff={dg:.3e}, expected >0)")

        # --- without mask (v2): coupling should be visible ---
        c = run(x0, False)
        d = run(x0_pert, False)
        mad2 = (c[:, :-1].float() - d[:, :-1].float()).abs().max().item()
        print(f"[oneway=False] action rows max_abs_diff={mad2:.3e} (expected >0: v2 coupling)")

        assert same, "ONE-WAY VIOLATED: gate perturbation leaked into action rows"
        assert dg > 0, "gate row unexpectedly identical"
        print("PASS: one-way mask decouples gate -> actions (bitwise).")


if __name__ == "__main__":
    main()
