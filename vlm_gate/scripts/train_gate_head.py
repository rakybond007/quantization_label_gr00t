"""Train a small MLP gate head on frozen GR00T backbone features (gate plan B).

Consumes features_shard*.npy + index_shard*.parquet from
extract_gate_backbone_features.py, joins with the teacher labels parquet on
(episode_index, frame_index), and trains dim->256->64->1 with BCE on soft
P(YES). Episode-wise train/val split, reports val AUC + agreement@tau,
optional wandb (project gate-distill).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class FeatureGate(nn.Module):
    def __init__(self, dim, hidden=(256, 64)):
        super().__init__()
        layers, d = [], dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FeatureDataset(Dataset):
    def __init__(self, features_dir, labels: pd.DataFrame):
        self.mms, lookup = [], {}
        for ix in sorted(glob.glob(os.path.join(features_dir, "index_shard*.parquet"))):
            sid = int(os.path.basename(ix).split("index_shard")[1].split(".")[0])
            mm = np.load(os.path.join(features_dir, f"features_shard{sid}.npy"), mmap_mode="r")
            k = len(self.mms)
            self.mms.append(mm)
            for r in pd.read_parquet(ix).itertuples():
                lookup[(r.episode_index, r.frame_index)] = (k, r.row)
        self.items = []
        for r in labels.itertuples():
            key = (r.episode_index, r.frame_index)
            if key in lookup:
                self.items.append((*lookup[key], float(r.p_yes)))
        self.dim = self.mms[0].shape[1]
        print(f"[head] matched {len(self.items)}/{len(labels)} frames, dim={self.dim}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        k, row, y = self.items[i]
        x = self.mms[k][row].astype(np.float32)
        return torch.from_numpy(x), torch.tensor([y], dtype=torch.float32)


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l >= 0.5]
    neg = [s for s, l in zip(scores, labels) if l < 0.5]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.25, help="fraction of EPISODES held out")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--wandb", default="", help="wandb run name (project gate-distill); empty=off")
    p.add_argument("--tau", type=float, default=0.5)
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
    print(f"[head] episodes train/val = {len(eps)-n_val}/{n_val}, "
          f"frames {len(tr)}/{len(va)}, label mean {labels['p_yes'].mean():.3f}")

    dtr = FeatureDataset(args.features_dir, tr)
    dva = FeatureDataset(args.features_dir, va)
    ltr = DataLoader(dtr, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)
    lva = DataLoader(dva, batch_size=args.bs, num_workers=args.num_workers)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = FeatureGate(dtr.dim).to(dev)
    nparam = sum(x.numel() for x in model.parameters())
    print(f"[head] FeatureGate params: {nparam/1e3:.1f}K, device={dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    best = -1.0
    for ep in range(args.epochs):
        model.train(); tl = 0.0; n = 0
        for x, y in ltr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward(); opt.step()
            tl += loss.item() * len(x); n += len(x)
        model.eval(); scores, ys = [], []
        with torch.no_grad():
            for x, y in lva:
                s = torch.sigmoid(model(x.to(dev))).cpu().numpy().ravel()
                scores += s.tolist(); ys += y.numpy().ravel().tolist()
        agree = np.mean([(s >= args.tau) == (y >= args.tau)
                         for s, y in zip(scores, ys)]) if ys else float("nan")
        a = auc(scores, ys)
        print(f"[head] epoch {ep+1}/{args.epochs} loss={tl/max(n,1):.4f} "
              f"val_AUC={a:.3f} val_agree@tau={agree:.3f}", flush=True)
        if wb:
            wb.log({"epoch": ep + 1, "train_loss": tl / max(n, 1),
                    "val_auc": a, "val_agree_tau": agree})
        if a == a and a > best:
            best = a
            torch.save({"model": model.state_dict(), "dim": dtr.dim,
                        "val_auc": a, "val_agree_tau": float(agree)},
                       os.path.join(args.out_dir, "gate_head_best.pt"))
    torch.save({"model": model.state_dict(), "dim": dtr.dim},
               os.path.join(args.out_dir, "gate_head_last.pt"))
    print(f"[head] saved -> {args.out_dir} (best val_AUC={best:.3f})")


if __name__ == "__main__":
    main()
