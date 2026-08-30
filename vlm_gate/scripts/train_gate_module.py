"""Train a lightweight gate module on VLM-labeled quantizability data.

Distills the (frozen VLM + best evolved guidance) gate into a small CNN so that
evaluation no longer needs the heavy judge. Reads the labels parquet produced
by label_gate_dataset.py plus the same LeRobot dataset frames.

Test-scale by design: episode-wise train/val split, BCE on soft P(YES) labels,
reports val AUC + agreement@tau against the VLM teacher.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

VIEW_KEYS = ["observation.images.left_view", "observation.images.right_view",
             "observation.images.wrist_view"]


def read_frames(mp4_path, idxs):
    try:
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
    except ImportError:
        import imageio
        r = imageio.get_reader(mp4_path)
        out = {}
        want = set(idxs)
        for i, f in enumerate(r):
            if i in want:
                out[i] = np.asarray(f)
                if len(out) == len(want):
                    break
        r.close()
        return out


class GateFrames(Dataset):
    def __init__(self, ds_root, labels: pd.DataFrame, res=128):
        self.res = res
        info = json.load(open(os.path.join(ds_root, "meta", "info.json")))
        cs = info.get("chunks_size", 1000)
        self.items = []
        cache = {}
        for ei, g in labels.groupby("episode_index"):
            idxs = g["frame_index"].tolist()
            chunk = ei // cs
            for vk in VIEW_KEYS:
                mp4 = os.path.join(ds_root, info["video_path"].format(
                    episode_chunk=chunk, video_key=vk, episode_index=ei))
                cache[(ei, vk)] = read_frames(mp4, idxs)
            for _, r in g.iterrows():
                self.items.append((ei, int(r["frame_index"]), float(r["p_yes"])))
        self.cache = cache

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ei, fi, y = self.items[i]
        imgs = [self.cache[(ei, vk)][fi] for vk in VIEW_KEYS]
        x = np.concatenate([self._prep(im) for im in imgs], axis=0)  # (9,res,res)
        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32)

    def _prep(self, im):
        import cv2
        im = cv2.resize(im, (self.res, self.res), interpolation=cv2.INTER_AREA)
        return (im.astype(np.float32) / 255.0).transpose(2, 0, 1)


class CachedGateFrames(Dataset):
    """Full-scale path: uint8 memmap shards from build_gate_frame_cache.py.

    act_cols: OFF BY DEFAULT and should stay that way for the deployed gate.
    The gate module is meant to run CONCURRENTLY with the action head's denoising
    (separate stream / MPS), so its latency is hidden. Feeding it the planned
    chunk would force it to wait for denoising to finish, turning a hidden cost
    into pure added latency. The teacher may look at the actions; the student
    sees images only and learns E[label | image]. A lower val AUC is the expected
    price of that, not a defect.
    """
    def __init__(self, cache_dir, labels: pd.DataFrame, task_emb=None, act_cols=None):
        import glob
        self.temb = task_emb  # {task_str: np.float32[384]} or None
        self.mms, self.lookup = [], {}
        for si, ix in enumerate(sorted(glob.glob(os.path.join(cache_dir, "index_shard*.parquet")))):
            sid = int(os.path.basename(ix).split("index_shard")[1].split(".")[0])
            mm = np.load(os.path.join(cache_dir, f"frames_shard{sid}.u8"), mmap_mode="r")
            k = len(self.mms); self.mms.append(mm)
            for r in pd.read_parquet(ix).itertuples():
                self.lookup[(r.episode_index, r.frame_index)] = (k, r.row)
        self.act_cols = list(act_cols or [])
        avals = (labels[self.act_cols].to_numpy(dtype=np.float32)
                 if self.act_cols else np.zeros((len(labels), 0), np.float32))
        self.items = []
        for i, r in enumerate(labels.itertuples()):
            key = (r.episode_index, r.frame_index)
            if key in self.lookup:
                self.items.append((*self.lookup[key], float(r.p_yes), r.task, avals[i]))
        print(f"[train] cache dataset: {len(self.items)}/{len(labels)} labeled frames matched")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        k, row, y, task, a = self.items[i]
        x = self.mms[k][row].astype(np.float32) / 255.0
        t = (torch.from_numpy(self.temb[task]) if self.temb is not None
             else torch.zeros(0))
        t = torch.cat([t, torch.from_numpy(np.ascontiguousarray(a))]) if len(a) else t
        return torch.from_numpy(x), t, torch.tensor([y], dtype=torch.float32)


class SmallGate(nn.Module):
    """Small CNN: 9ch (3 views) [+ task-instruction embedding] -> P(quantize)."""
    def __init__(self, ch=32, temb_dim=0):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.net = nn.Sequential(blk(9, ch), blk(ch, ch * 2), blk(ch * 2, ch * 4),
                                 blk(ch * 4, ch * 4), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Linear(ch * 4 + temb_dim, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, t=None):
        f = self.net(x).flatten(1)
        if t is not None and t.numel():
            f = torch.cat([f, t], dim=1)
        return self.head(f)


class DinoGate(nn.Module):
    """DINOv3 ViT-S/16 동결 + attention 풀링 + 지시문 조건화.

    SmallGate 는 3뷰를 첫 conv 에서 채널로 섞고 GlobalAvgPool 로 공간 정보를 버렸다.
    여기서는 뷰마다 따로 인코딩해 패치 토큰을 유지하고, 학습되는 질의 하나로
    attention 풀링한다. 인코더는 얼리므로 학습 파라미터는 풀링+헤드뿐이다.

    프로브(4x4 로 미리 뭉갠 특징)에서 val_AUC 0.829 로 SmallGate 0.734 를 앞섰다.
    여기서는 토큰을 뭉개지 않으므로 그 값이 하한이다.
    """
    DINO = "facebook/dinov3-vits16-pretrain-lvd1689m"

    def __init__(self, temb_dim=0, res=224, nheads=6):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(self.DINO)
        for p_ in self.enc.parameters():
            p_.requires_grad = False
        self.enc.eval()
        d = self.enc.config.hidden_size
        self.res = res
        self.d = d
        self.view_emb = nn.Parameter(torch.randn(3, 1, d) * 0.02)   # 뷰 구분
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.att = nn.MultiheadAttention(d, nheads, batch_first=True)
        self.temb_proj = nn.Linear(temb_dim, d) if temb_dim else None
        din = d * (2 if temb_dim else 1)
        self.head = nn.Sequential(nn.Linear(din, 256), nn.ReLU(),
                                  nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.enc.eval()          # 동결 인코더는 항상 eval
        return self

    def forward(self, x, t=None):
        B = x.shape[0]
        v = x.reshape(B * 3, 3, x.shape[-2], x.shape[-1])
        v = torch.nn.functional.interpolate(v, size=(self.res, self.res),
                                            mode="bilinear", align_corners=False)
        v = (v - self.mean) / self.std
        with torch.no_grad():
            h = self.enc(pixel_values=v).last_hidden_state          # (B*3, 1+reg+P, d)
        P = (self.res // self.enc.config.patch_size) ** 2
        tok = h[:, -P:, :].reshape(B, 3, P, self.d)                 # 패치 토큰만
        tok = (tok + self.view_emb.unsqueeze(0)).reshape(B, 3 * P, self.d).float()
        o, _ = self.att(self.q.expand(B, -1, -1).float(), tok, tok)
        f = o[:, 0]
        if self.temb_proj is not None and t is not None and t.numel():
            f = torch.cat([f, self.temb_proj(t)], dim=1)
        return self.head(f)


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l >= 0.5]
    neg = [s for s, l in zip(scores, labels) if l < 0.5]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.25, help="fraction of EPISODES held out")
    p.add_argument("--cache-dir", default="", help="memmap frame cache dir (full-scale path)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--wandb", default="", help="wandb run name (project gate-distill); empty=off")
    p.add_argument("--task-emb", default="", help="task_embeddings.npz (MiniLM); conditions on instruction")
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--encoder", default="smallgate", choices=["smallgate", "dinov3s"],
                   help="게이트 비전 인코더. dinov3s = DINOv3 ViT-S/16 동결 + attention 풀링.")
    p.add_argument("--act-cols", default="", help="comma-separated action descriptor columns "
                   "from the labels parquet, appended to the conditioning vector")
    args = p.parse_args()

    wb = None
    if args.wandb:
        import wandb as wb
        wb.init(project="gate-distill", name=args.wandb, config=vars(args))

    labels = pd.read_parquet(args.labels)
    eps = sorted(labels["episode_index"].unique())
    n_val = max(1, int(len(eps) * args.val_frac))
    val_eps = set(eps[-n_val:])
    tr = labels[~labels["episode_index"].isin(val_eps)]
    va = labels[labels["episode_index"].isin(val_eps)]
    print(f"[train] episodes train/val = {len(eps)-n_val}/{n_val}, "
          f"frames {len(tr)}/{len(va)}, label mean {labels['p_yes'].mean():.3f}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    temb = None
    if args.task_emb:
        z = np.load(args.task_emb, allow_pickle=True)
        temb = {t: e for t, e in zip(z["tasks"], z["emb"])}
        missing = set(labels["task"].unique()) - set(temb)
        assert not missing, f"tasks missing from embedding file: {list(missing)[:3]}"
    acols = [c for c in args.act_cols.split(",") if c]
    if acols:
        missing = [c for c in acols if c not in labels.columns]
        assert not missing, f"labels parquet lacks action columns: {missing}"
        print(f"[train] action descriptors: {acols}")
    if args.cache_dir:
        dtr = CachedGateFrames(args.cache_dir, tr, temb, acols)
        dva = CachedGateFrames(args.cache_dir, va, temb, acols)
    else:
        dtr = GateFrames(args.dataset_path, tr)
        dva = GateFrames(args.dataset_path, va)
    ltr = DataLoader(dtr, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)
    lva = DataLoader(dva, batch_size=args.bs, num_workers=args.num_workers)

    cond_dim = (384 if temb is not None else 0) + len(acols)
    if args.encoder == "dinov3s":
        model = DinoGate(temb_dim=cond_dim).to(dev)
        ntr = sum(x.numel() for x in model.parameters() if x.requires_grad)
        nall = sum(x.numel() for x in model.parameters())
        print(f"[train] DinoGate params: 학습 {ntr/1e6:.2f}M / 전체 {nall/1e6:.2f}M "
              f"(인코더 동결), device={dev}")
    else:
        model = SmallGate(temb_dim=cond_dim).to(dev)
        nparam = sum(x.numel() for x in model.parameters())
        print(f"[train] SmallGate params: {nparam/1e6:.2f}M, device={dev}")
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    for ep in range(args.epochs):
        model.train(); tl = 0.0; n = 0
        for batch in ltr:
            x, t, y = batch if len(batch) == 3 else (batch[0], None, batch[1])
            x, y = x.to(dev), y.to(dev)
            t = t.to(dev) if t is not None else None
            opt.zero_grad()
            loss = lossf(model(x, t), y)
            loss.backward(); opt.step()
            tl += loss.item() * len(x); n += len(x)
        model.eval(); scores, ys = [], []
        with torch.no_grad():
            for batch in lva:
                x, t, y = batch if len(batch) == 3 else (batch[0], None, batch[1])
                t = t.to(dev) if t is not None else None
                s = torch.sigmoid(model(x.to(dev), t)).cpu().numpy().ravel()
                scores += s.tolist(); ys += y.numpy().ravel().tolist()
        agree = np.mean([(s >= args.tau) == (y >= args.tau) for s, y in zip(scores, ys)]) if ys else float("nan")
        a = auc(scores, ys)
        print(f"[train] epoch {ep+1}/{args.epochs} loss={tl/max(n,1):.4f} "
              f"val_AUC={a:.3f} val_agree@tau={agree:.3f}", flush=True)
        if wb:
            wb.log({"epoch": ep + 1, "train_loss": tl / max(n, 1),
                    "val_auc": a, "val_agree_tau": agree})
        if not np.isnan(a) and a >= getattr(main, "_best_auc", -1.0):
            main._best_auc = a
            torch.save({"model": model.state_dict(), "res": 128, "views": VIEW_KEYS, "encoder": args.encoder,
                        "temb_dim": cond_dim, "act_cols": acols,
                        "task_emb_file": args.task_emb, "epoch": ep + 1, "val_auc": a},
                       os.path.join(args.out_dir, "gate_module_best.pt"))
    torch.save({"model": model.state_dict(), "res": 128, "views": VIEW_KEYS, "encoder": args.encoder,
                "temb_dim": cond_dim, "act_cols": acols,
                "task_emb_file": args.task_emb},
               os.path.join(args.out_dir, "gate_module.pt"))
    print(f"[train] saved -> {os.path.join(args.out_dir, 'gate_module.pt')}")


if __name__ == "__main__":
    main()
