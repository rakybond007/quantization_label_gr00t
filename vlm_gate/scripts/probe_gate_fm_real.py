"""Probe fm_token gate on the REAL DROID pnp dataset (2 views, joint state).

Same idea as probe_gate_fm.py: load checkpoint via Gr00tPolicy
(real_droid_joint config), run head=main_gate on labeled frames spread over
the p_yes range, print (episode, frame, p_pred, p_yes); assert non-constant.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_gate_backbone_features import read_frames  # noqa: E402

DS = "/sjw_alinlab/home/hojin2/taekwan/Isaac-GR00T/Data/human_data/MoSS/lerobot/pnp_objects"
LABELS = ("/rlwrld-unified-checkpoints/hojin2/checkpoints/"
          "gate_distill_real_droid_pnp_v1/labels/real_luna_full.parquet")
VIEW_KEYS = ["observation.images.exterior_image_1_left",
             "observation.images.wrist_image_left"]
VIDEO_KEYS = ["video.exterior_image_1_left", "video.wrist_image_left"]
STATE_KEYS = ["state.joint_pos_abs", "state.gripper_close"]
LANG_KEY = "annotation.human.action.task_description"


def load_policy(model_path):
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy
    dc = DATA_CONFIG_MAP["real_droid_joint"]
    return Gr00tPolicy(
        model_path=model_path,
        modality_config=dc.modality_config(),
        modality_transform=dc.transform(backbone_model_type="eagle"),
        embodiment_tag="new_embodiment",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


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
    lab = lab[lab.episode_index < 6].sort_values("p_yes")
    idxs = np.linspace(0, len(lab) - 1, n_frames).astype(int)
    rows = lab.iloc[idxs]

    policy = load_policy(ckpt)
    policy.inference_head = "main_gate"
    preds, ys = [], []
    for _, r in rows.iterrows():
        obs = build_obs(info, mod, int(r.episode_index), int(r.frame_index), str(r.task))
        torch.manual_seed(7)
        out = policy.get_action(obs)
        p = float(np.asarray(out["_gate_prob"]).ravel()[0])
        preds.append(p)
        ys.append(float(r.p_yes))
        print(f"ep={int(r.episode_index):3d} f={int(r.frame_index):4d} "
              f"p_pred={p:.3f} p_yes={r.p_yes:.2f}")
    preds = np.array(preds)
    print(f"[probe] p_pred std={preds.std():.4f} range=({preds.min():.3f},{preds.max():.3f}) "
          f"corr={np.corrcoef(preds, ys)[0,1]:.3f}")
    assert preds.std() > 1e-3, "gate output is constant"
    print("[probe] OK: gate output varies across frames")


if __name__ == "__main__":
    main()
