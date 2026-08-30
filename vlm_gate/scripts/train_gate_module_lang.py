"""A'-lang: gate module with frozen text-encoder instruction embeddings.

Same SmallGate CNN trunk/hparams as train_gate_module.py, but the 384-d
frozen MiniLM instruction embedding passes through ONE trainable linear
projection (384->32) before concat with the CNN feature. Supports holding
out entire task types (env names, mapped via meta/episodes.jsonl) to
measure unseen-task generalization: held-out episodes are excluded from
train AND the normal val split, and their AUC is reported separately.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_gate_module import CachedGateFrames, auc, VIEW_KEYS


class SmallGateLang(nn.Module):
    def __init__(self, ch=32, temb_in=384, proj_dim=32):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.net = nn.Sequential(blk(9, ch), blk(ch, ch * 2), blk(ch * 2, ch * 4),
                                 blk(ch * 4, ch * 4), nn.AdaptiveAvgPool2d(1))
        self.proj = nn.Linear(temb_in, proj_dim)
        self.head = nn.Sequential(nn.Linear(ch * 4 + proj_dim, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, t):
        f = self.net(x).flatten(1)
        return self.head(torch.cat([f, self.proj(t)], dim=1))


def evaluate(model, loader, dev):
    model.eval()
    scores, ys = [], []
    with torch.no_grad():
        for x, t, y in loader:
            s = torch.sigmoid(model(x.to(dev), t.to(dev))).cpu().numpy().ravel()
            scores += s.tolist()
            ys += y.numpy().ravel().tolist()
    return scores, ys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.25)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--wandb", default="")
    p.add_argument("--task-emb", required=True, help="precomputed frozen text-encoder npz (tasks, emb)")
    p.add_argument("--proj-dim", type=int, default=32)
    p.add_argument("--holdout-tasks", default="", help="comma-separated env names (episodes.jsonl tasks[1]) to hold out")
    p.add_argument("--tau", type=float, default=0.5)
    args = p.parse_args()

    wb = None
    if args.wandb:
        import wandb as wb
        wb.init(project="gate-distill", name=args.wandb, config=vars(args))

    labels = pd.read_parquet(args.labels)
    hold = pd.DataFrame()
    if args.holdout_tasks:
        names = set(args.holdout_tasks.split(","))
        ep2env = {}
        with open(os.path.join(args.dataset_path, "meta", "episodes.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                ep2env[r["episode_index"]] = r["tasks"][1]
        bad = names - set(ep2env.values())
        assert not bad, f"unknown env names: {bad}"
        mask = labels["episode_index"].map(ep2env).isin(names)
        hold = labels[mask]
        labels = labels[~mask]
        print(f"[lang] holdout {sorted(names)}: {len(hold)} frames / "
              f"{hold['episode_index'].nunique()} eps excluded; {len(labels)} remain")

    eps = sorted(labels["episode_index"].unique())
    n_val = max(1, int(len(eps) * args.val_frac))
    val_eps = set(eps[-n_val:])
    tr = labels[~labels["episode_index"].isin(val_eps)]
    va = labels[labels["episode_index"].isin(val_eps)]
    print(f"[lang] episodes train/val = {len(eps)-n_val}/{n_val}, frames {len(tr)}/{len(va)}, "
          f"label mean {labels['p_yes'].mean():.3f}")

    z = np.load(args.task_emb, allow_pickle=True)
    temb = {t: e for t, e in zip(z["tasks"], z["emb"])}
    for df, nm in [(tr, "train"), (va, "val"), (hold, "holdout")]:
        if len(df):
            missing = set(df["task"].unique()) - set(temb)
            assert not missing, f"{nm}: tasks missing from emb file: {list(missing)[:3]}"

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtr = CachedGateFrames(args.cache_dir, tr, temb)
    dva = CachedGateFrames(args.cache_dir, va, temb)
    ltr = DataLoader(dtr, batch_size=args.bs, shuffle=True, num_workers=args.num_workers)
    lva = DataLoader(dva, batch_size=args.bs, num_workers=args.num_workers)
    lho = None
    if len(hold):
        lho = DataLoader(CachedGateFrames(args.cache_dir, hold, temb),
                         batch_size=args.bs, num_workers=args.num_workers)

    temb_in = next(iter(temb.values())).shape[0]
    model = SmallGateLang(temb_in=temb_in, proj_dim=args.proj_dim).to(dev)
    print(f"[lang] params: {sum(x.numel() for x in model.parameters())/1e6:.2f}M, "
          f"temb_in={temb_in}, proj_dim={args.proj_dim}, device={dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()

    os.makedirs(args.out_dir, exist_ok=True)
    for ep in range(args.epochs):
        model.train(); tl = 0.0; n = 0
        for x, t, y in ltr:
            x, t, y = x.to(dev), t.to(dev), y.to(dev)
            opt.zero_grad()
            loss = lossf(model(x, t), y)
            loss.backward(); opt.step()
            tl += loss.item() * len(x); n += len(x)
        scores, ys = evaluate(model, lva, dev)
        a = auc(scores, ys)
        agree = np.mean([(s >= args.tau) == (y >= args.tau) for s, y in zip(scores, ys)])
        msg = (f"[lang] epoch {ep+1}/{args.epochs} loss={tl/max(n,1):.4f} "
               f"val_AUC={a:.3f} val_agree@tau={agree:.3f}")
        log = {"epoch": ep + 1, "train_loss": tl / max(n, 1), "val_auc": a, "val_agree_tau": agree}
        if lho is not None:
            hs, hy = evaluate(model, lho, dev)
            ha = auc(hs, hy)
            msg += f" holdout_AUC={ha:.3f}"
            log["holdout_auc"] = ha
        print(msg, flush=True)
        if wb:
            wb.log(log)
    torch.save({"model": model.state_dict(), "res": 128, "views": VIEW_KEYS,
                "arch": "lang", "temb_in": temb_in, "proj_dim": args.proj_dim,
                "task_emb_file": args.task_emb,
                "holdout_tasks": args.holdout_tasks},
               os.path.join(args.out_dir, "gate_module.pt"))
    print(f"[lang] saved -> {os.path.join(args.out_dir, 'gate_module.pt')}")


if __name__ == "__main__":
    main()
