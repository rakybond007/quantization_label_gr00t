"""Milestone-1 verification for the quantizability gate token (plan C).

Checks, on a real robocasa observation with identical seeds, that:
  (a) baseline action_head.get_action (no gate attached)
  (b) action_head.get_action after attach_quant_gate  (path untouched)
  (c) action_head.get_action_and_gate (gate stream runs at final step)
all produce BIT-IDENTICAL action chunks (torch.equal, i.e. allclose atol=0).

The gate token is a strictly one-way parallel stream: no state/action token
ever attends to it, and the base DiT pass is executed unmodified.
"""
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_gate_backbone_features import (  # noqa: E402
    VIEW_KEYS, VIDEO_KEYS, STATE_KEYS, LANG_KEY, read_frames, load_policy,
)

DS = "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
CKPT = os.path.expanduser(
    "~/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000")


def build_obs():
    info = json.load(open(os.path.join(DS, "meta", "info.json")))
    mod = json.load(open(os.path.join(DS, "meta", "modality.json")))["state"]
    import pandas as pd
    ep, fr = 0, 32
    chunk = ep // info.get("chunks_size", 1000)
    obs = {}
    for vk, dk in zip(VIEW_KEYS, VIDEO_KEYS):
        mp4 = os.path.join(DS, info["video_path"].format(
            episode_chunk=chunk, video_key=vk, episode_index=ep))
        f = read_frames(mp4, [fr])[fr]
        obs[dk] = f[None, None]  # (1,1,H,W,3)
    pq = pd.read_parquet(os.path.join(DS, info["data_path"].format(
        episode_chunk=chunk, episode_index=ep)), columns=["observation.state"])
    state = np.stack(pq["observation.state"].to_numpy())[fr]
    for sk in STATE_KEYS:
        sl = mod[sk.split("state.")[1]]
        obs[sk] = state[sl["start"]:sl["end"]][None, None].astype(np.float64)
    tasks = [json.loads(l) for l in open(os.path.join(DS, "meta", "tasks.jsonl"))]
    obs[LANG_KEY] = [[tasks[0]["task"]]]
    return obs


def main():
    from gr00t.model.policy import COMPUTE_DTYPE

    policy = load_policy(CKPT)
    model = policy.model
    head = model.action_head
    obs = build_obs()

    normalized = policy.apply_transforms(obs)
    backbone_inputs, action_inputs = model.prepare_input(normalized)

    def run(fn):
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE):
            bo = model.backbone(backbone_inputs)
            bo_copy = type(bo)(data={k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v))
                                     for k, v in bo.items()})
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            return fn(bo_copy)

    # (a) baseline, gate not attached
    out_a = run(lambda bo: head.get_action(bo, action_inputs))["action_pred"]

    # attach gate token
    head.attach_quant_gate(loss_weight=0.1)
    head.to(device=out_a.device)
    head.eval()

    # (b) gate attached, normal get_action path
    out_b = run(lambda bo: head.get_action(bo, action_inputs))["action_pred"]

    # (c) gate attached, get_action_and_gate path
    res_c = run(lambda bo: head.get_action_and_gate(bo, action_inputs))
    out_c, gate_logit, gate_prob = res_c["action_pred"], res_c["gate_logit"], res_c["gate_prob"]

    def report(name, x, y):
        same = torch.equal(x, y)
        mad = (x.float() - y.float()).abs().max().item()
        print(f"[{name}] bitwise_equal={same} max_abs_diff={mad:.3e} shape={tuple(x.shape)}")
        return same

    ok1 = report("a-vs-b: attach must not change get_action", out_a, out_b)
    ok2 = report("a-vs-c: gate stream must not touch actions", out_a, out_c)
    print(f"[gate] logit={gate_logit.float().item():+.4f} prob={gate_prob.item():.4f}")
    assert ok1 and ok2, "ACTION IDENTITY VIOLATED"
    assert torch.allclose(out_a, out_c, atol=0.0, rtol=0.0)
    print("PASS: action outputs are bit-identical with the gate token attached.")


if __name__ == "__main__":
    main()
