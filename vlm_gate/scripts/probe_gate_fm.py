"""Probe C-v2 fm_token gate: denoised g_hat variability + comparison to p_yes.

Loads a checkpoint via Gr00tPolicy, runs head=main_gate on several labeled
frames, prints (episode, frame, p_pred, p_yes). Asserts p_pred is not constant.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_gate_backbone_features import (  # noqa: E402
    VIEW_KEYS, VIDEO_KEYS, STATE_KEYS, LANG_KEY, read_frames, load_policy,
)

DS = "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
LABELS = "/rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_robocasa_cosmos_v1/labels/full_merged.parquet"


def build_obs(info, mod, ep, fr, task):
    chunk = ep // info.get("chunks_size", 1000)
    obs = {}
    for vk, dk in zip(VIEW_KEYS, VIDEO_KEYS):
        mp4 = os.path.join(DS, info["video_path"].format(
            episode_chunk=chunk, video_key=vk, episode_index=ep))
        obs[dk] = read_frames(mp4, [fr])[fr][None, None]
    pq = pd.read_parquet(os.path.join(DS, info["data_path"].format(
        episode_chunk=chunk, episode_index=ep)), columns=["observation.state"])
    state = np.stack(pq["observation.state"].to_numpy())[min(fr, len(pq) - 1)]
    for sk in STATE_KEYS:
        sl = mod[sk.split("state.")[1]]
        obs[sk] = state[sl["start"]:sl["end"]][None, None].astype(np.float64)
    obs[LANG_KEY] = [[task]]
    return obs


def main():
    ckpt = sys.argv[1]
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    info = json.load(open(os.path.join(DS, "meta", "info.json")))
    mod = json.load(open(os.path.join(DS, "meta", "modality.json")))["state"]
    lab = pd.read_parquet(LABELS)
    # Pick frames spread over label range (mix of low/high p_yes)
    lab = lab[lab.episode_index < 6].sort_values("p_yes")
    idxs = np.linspace(0, len(lab) - 1, n_frames).astype(int)
    rows = lab.iloc[idxs]

    policy = load_policy(ckpt)
    policy.inference_head = "main_gate"
    preds, ys, act0 = [], [], []
    for _, r in rows.iterrows():
        obs = build_obs(info, mod, int(r.episode_index), int(r.frame_index), str(r.task))
        torch.manual_seed(7)
        out = policy.get_action(obs)
        p = float(np.asarray(out["_gate_prob"]).ravel()[0])
        a_key = next(k for k in out if k.startswith("action."))
        a0 = float(np.asarray(out[a_key]).ravel()[0])
        preds.append(p); ys.append(float(r.p_yes)); act0.append(a0)
        print(f"ep{int(r.episode_index):4d} fr{int(r.frame_index):4d} "
              f"p_pred={p:.6f} p_yes={r.p_yes:.4f} act0={a0:+.4f}")
    preds, ys, act0 = np.asarray(preds), np.asarray(ys), np.asarray(act0)
    print(f"p_pred: mean={preds.mean():.6f} std={preds.std():.6f} "
          f"range=[{preds.min():.6f},{preds.max():.6f}] nunique={len(np.unique(preds.round(6)))}")
    print(f"act0 (same-seed cross-frame): std={act0.std():.6f}")
    if preds.std() > 1e-6 and ys.std() > 0:
        print(f"corr(p_pred, p_yes) = {np.corrcoef(preds, ys)[0,1]:.4f}")
    assert preds.std() > 1e-6, "CONSTANT gate prediction"
    print("PROBE PASS: denoised gate is not constant")


if __name__ == "__main__":
    main()
