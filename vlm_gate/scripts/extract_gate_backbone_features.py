"""Extract frozen GR00T-N1.5 backbone features for gate-B distillation.

Loads the robocasa-finetuned GR00T checkpoint via Gr00tPolicy (same
single_panda_gripper transforms as eval), runs ONLY the Eagle backbone
(vision tower + truncated LLM + eagle_linear, see
gr00t/model/backbone/eagle_backbone.py) on the 3-view frames listed in the
labels parquet, and saves one masked-mean-pooled feature vector per frame.

Output per shard: features_shard{K}.npy (float16, (N, D)) + index_shard{K}.parquet
(row, episode_index, frame_index) — same pairing convention as
build_gate_frame_cache.py so train_gate_head.py can join on (episode, frame).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

VIEW_KEYS = ["observation.images.left_view", "observation.images.right_view",
             "observation.images.wrist_view"]
VIDEO_KEYS = ["video.left_view", "video.right_view", "video.wrist_view"]
STATE_KEYS = ["state.end_effector_position_relative", "state.end_effector_rotation_relative",
              "state.gripper_qpos", "state.base_position", "state.base_rotation"]
LANG_KEY = "annotation.human.action.task_description"


def read_frames(mp4_path, idxs):
    import av
    out = {}
    want = sorted(set(idxs))
    with av.open(mp4_path) as c:
        i = 0
        for frame in c.decode(video=0):
            if i in want:
                out[i] = frame.to_ndarray(format="rgb24")
                if len(out) == len(want):
                    break
            i += 1
    return out


def load_policy(model_path):
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy
    dc = DATA_CONFIG_MAP["single_panda_gripper"]
    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=dc.modality_config(),
        modality_transform=dc.transform(backbone_model_type="eagle"),
        embodiment_tag="new_embodiment",
        backbone_model_type="eagle",
    )
    policy.model.eval()
    return policy


@torch.no_grad()
def backbone_features(policy, obs_batch):
    """obs_batch: batched obs dict (B,T,...) -> (B, D) float32 numpy."""
    normalized = policy.apply_transforms(obs_batch)
    backbone_inputs, _ = policy.model.prepare_input(normalized)
    out = policy.model.backbone(backbone_inputs)
    feats = out["backbone_features"].float()          # (B, S, D)
    mask = out["backbone_attention_mask"].float().unsqueeze(-1)  # (B, S, 1)
    pooled = (feats * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    return pooled.cpu().numpy()


def patch_backbone_layers(bb, layers):
    """forward_eagle 을 감싸 지정 레이어들의 raw hidden state 를 함께 남긴다.
    12층 위는 이미 삭제돼 있으므로 후보는 액션 헤드가 직접 쓰지 않는 아래 레이어들이다.
    전 레이어가 어차피 계산되므로 추출에 추가 연산이 없다."""
    def wrapped(vl_input):
        ei = {k.removeprefix("eagle_"): v for k, v in vl_input.items() if k.startswith("eagle_")}
        ei.pop("image_sizes", None)
        out = bb.eagle_model(**ei, output_hidden_states=True, return_dict=True)
        bb._probe_hidden = {L: out.hidden_states[L] for L in layers}
        return bb.eagle_linear(out.hidden_states[bb.select_layer]), ei["attention_mask"]
    bb.forward_eagle = wrapped


@torch.no_grad()
def layer_features(policy, obs_batch, layers):
    """레이어별 masked-mean 풀링 특징. -> {L: (B, D)}"""
    normalized = policy.apply_transforms(obs_batch)
    bi, _ = policy.model.prepare_input(normalized)
    out = policy.model.backbone(bi)
    mask = out["backbone_attention_mask"].float().unsqueeze(-1)
    bb = policy.model.backbone
    res = {}
    for L in layers:
        h = bb._probe_hidden[L].float()
        res[L] = ((h * mask).sum(1) / mask.sum(1).clamp(min=1.0)).cpu().numpy()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", default="", help="쉼표 구분 레이어 인덱스. 주면 레이어별로 따로 저장한다.")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-episodes", type=int, default=0, help="smoke: limit episodes")
    args = p.parse_args()

    ds = args.dataset_path
    info = json.load(open(os.path.join(ds, "meta", "info.json")))
    cs = info.get("chunks_size", 1000)
    mod = json.load(open(os.path.join(ds, "meta", "modality.json")))["state"]

    lab = pd.read_parquet(args.labels)
    eps = sorted(lab["episode_index"].unique())
    eps = [e for e in eps if e % args.num_shards == args.shard]
    if args.max_episodes:
        eps = eps[: args.max_episodes]
    lab = lab[lab["episode_index"].isin(eps)].sort_values(["episode_index", "frame_index"])
    n = len(lab)
    os.makedirs(args.out_dir, exist_ok=True)
    ix_path = os.path.join(args.out_dir, f"index_shard{args.shard}.parquet")
    ft_path = os.path.join(args.out_dir, f"features_shard{args.shard}.npy")
    if os.path.exists(ix_path) and os.path.exists(ft_path):
        if len(pd.read_parquet(ix_path)) == n:
            print(f"[feat] shard {args.shard} already complete ({n} rows), skip")
            return

    policy = load_policy(args.model_path)
    LAYERS = [int(x) for x in args.layers.split(',')] if args.layers else []
    if LAYERS:
        patch_backbone_layers(policy.model.backbone, LAYERS)
        print(f'[feat] 레이어별 추출: {LAYERS}')
    mmL = {}
    mm = None  # allocate after first batch (feature dim discovered at runtime)
    row = 0
    rows_ix = []
    import time
    t0 = time.time()
    for ei, g in lab.groupby("episode_index"):
        idxs = g["frame_index"].tolist()
        chunk = ei // cs
        views = {}
        for vk in VIEW_KEYS:
            mp4 = os.path.join(ds, info["video_path"].format(
                episode_chunk=chunk, video_key=vk, episode_index=ei))
            views[vk] = read_frames(mp4, idxs)
        ep_pq = pd.read_parquet(os.path.join(ds, info["data_path"].format(
            episode_chunk=chunk, episode_index=ei)), columns=["observation.state"])
        state_all = np.stack(ep_pq["observation.state"].to_numpy())  # (T, 53)
        tasks = g["task"].tolist() if "task" in g else [""] * len(idxs)

        for b0 in range(0, len(idxs), args.batch_size):
            bidx = idxs[b0:b0 + args.batch_size]
            btask = tasks[b0:b0 + args.batch_size]
            B = len(bidx)
            obs = {}
            for vk, dk in zip(VIEW_KEYS, VIDEO_KEYS):
                ims = []
                for fi in bidx:
                    im = views[vk].get(fi)
                    if im is None:
                        im = np.zeros((256, 256, 3), np.uint8)
                    ims.append(im)
                obs[dk] = np.stack(ims)[:, None]  # (B, 1, H, W, 3)
            for sk in STATE_KEYS:
                sl = mod[sk.split("state.")[1]]
                rows = np.stack([state_all[min(fi, len(state_all) - 1)] for fi in bidx])
                obs[sk] = rows[:, sl["start"]:sl["end"]][:, None].astype(np.float64)
            obs[LANG_KEY] = [[t] for t in btask]
            if LAYERS:
                fl = layer_features(policy, obs, LAYERS)
                for L, v in fl.items():
                    if L not in mmL:
                        mmL[L] = np.lib.format.open_memmap(
                            ft_path.replace(".npy", f"_layer{L}.npy"), mode="w+",
                            dtype=np.float16, shape=(n, v.shape[1]))
                    mmL[L][row:row + B] = v.astype(np.float16)
                feats = next(iter(fl.values()))
            else:
                feats = backbone_features(policy, obs)
            if mm is None and not LAYERS:
                dim = feats.shape[1]
                print(f"[feat] feature dim = {dim}")
                mm = np.lib.format.open_memmap(ft_path, mode="w+", dtype=np.float16,
                                               shape=(n, dim))
            if mm is not None: mm[row:row + B] = feats.astype(np.float16)
            for k, fi in enumerate(bidx):
                rows_ix.append({"row": row + k, "episode_index": int(ei),
                                "frame_index": int(fi)})
            row += B
        el = time.time() - t0
        print(f"[feat] shard {args.shard}: ep {ei} done, {row}/{n} rows, "
              f"{row/max(el,1e-6):.1f} fps", flush=True)
    if mm is not None: mm.flush()
    for _m in mmL.values(): _m.flush()
    pd.DataFrame(rows_ix).to_parquet(ix_path, index=False)
    print(f"[feat] shard {args.shard} done: {row} rows -> {ft_path}")


if __name__ == "__main__":
    main()
