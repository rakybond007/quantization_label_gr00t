"""C-v4 unit test: with one-way mask + pixel injection, perturbing the gate
row of the initial noise AND the gate pixel input must leave the 16 action
rows BIT-IDENTICAL. Sanity: gate row itself must change (pixels are used)."""
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
    head.attach_quant_gate_fm(loss_weight=1.0, oneway=True, pixel=True)
    head.quant_gate_pixel_encoder.to(device=model.device, dtype=head.dtype)
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
        x0_pert[:, -1] += 3.0

        px_a = torch.randint(0, 256, (1, 3, 112, 112, 3), dtype=torch.uint8,
                             device=vl.device)
        px_b = torch.randint(0, 256, (1, 3, 112, 112, 3), dtype=torch.uint8,
                             device=vl.device)

        a = head._fm_denoise(vl, state, emb_id, x0.clone(), gate_pixels=px_a)
        b = head._fm_denoise(vl, state, emb_id, x0_pert.clone(), gate_pixels=px_b)

        same = torch.equal(a[:, :-1], b[:, :-1])
        mad = (a[:, :-1].float() - b[:, :-1].float()).abs().max().item()
        dg = (a[:, -1].float() - b[:, -1].float()).abs().max().item()
        print(f"[oneway+pixel] action rows bitwise_equal={same} max_abs_diff={mad:.3e} "
              f"(gate row diff={dg:.3e}, expected >0)")

        # pixel-only perturbation must also change the gate row (pixels used)
        c = head._fm_denoise(vl, state, emb_id, x0.clone(), gate_pixels=px_b)
        dpx = (a[:, -1].float() - c[:, -1].float()).abs().max().item()
        same2 = torch.equal(a[:, :-1], c[:, :-1])
        print(f"[pixel-only  ] action rows bitwise_equal={same2}, gate row diff={dpx:.3e} "
              f"(expected >0: pixels reach the gate)")

        assert same and same2, "ONE-WAY VIOLATED: perturbation leaked into actions"
        assert dg > 0 and dpx > 0, "gate row insensitive to inputs"
        print("PASS: pixel-injected gate is one-way (actions bitwise invariant).")


if __name__ == "__main__":
    main()
