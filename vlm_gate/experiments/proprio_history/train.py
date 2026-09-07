"""Train the isolated proprio/history student. One GPU, no production mutations.

Smoke limits are explicit CLI options; zero means all rows/tasks/episodes.
Images are gathered from the existing uint8 cache, with strict join checks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from model import SmallGateBaseline, SmallGateProprioHistory, pack_history

VIEW_KEYS = ["observation.images.left_view", "observation.images.right_view",
             "observation.images.wrist_view"]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def select_split(labels, args):
    keys = ["episode_index", "frame_index"]
    if labels.duplicated(keys).any():
        raise ValueError("duplicate episode/frame labels")
    if not np.isfinite(labels.p_yes).all() or not labels.p_yes.between(0, 1).all():
        raise ValueError("p_yes must be finite and in [0,1]")
    # A complete episode belongs to exactly one split, even if it has subtasks.
    ep_task = labels.groupby("episode_index").task.first()
    rng = np.random.default_rng(args.seed)
    tasks = sorted(ep_task.unique())
    if args.max_tasks:
        tasks = [tasks[i] for i in sorted(rng.choice(len(tasks), min(len(tasks), args.max_tasks), replace=False))]
    if args.val_frac <= 0:
        # 전량 학습. VLA 학습이 그렇듯 끝까지 돌리고 마지막 것을 쓴다.
        # 판정은 폐루프 eval 이 한다 -- 게이트는 수단이고, 안 본 데이터의 BCE 가
        # 낮은 쪽이 실제로 성공률을 올리는 쪽이라는 보장이 없다.
        #
        # 그러면 지시문마다 에피소드 2개를 요구할 이유도 없어진다. 나눌 것이
        # 없으니 한 에피소드짜리 지시문도 그냥 학습에 들어간다.
        lab = labels.copy()
        if args.max_tasks:
            keep = set(tasks[:args.max_tasks])
            lab = lab[lab.task.isin(keep)]
        if args.episodes_per_task or args.frames_per_episode:
            sel = []
            for _, g in lab.groupby("task"):
                eps = rng.permutation(g.episode_index.unique())
                if args.episodes_per_task:
                    eps = eps[:args.episodes_per_task]
                gg = g[g.episode_index.isin(eps)]
                if args.frames_per_episode:
                    gg = (gg.sort_values("frame_index").groupby("episode_index")
                          .head(args.frames_per_episode))
                sel.append(gg)
            lab = pd.concat(sel) if sel else lab
        lab = lab.sort_values(keys).reset_index(drop=True)
        lab["split"] = "train"
        print(json.dumps({"val_frac": 0, "train_rows": len(lab),
                          "train_episodes": int(lab.episode_index.nunique()),
                          "tasks": int(lab.task.nunique())}), flush=True)
        return lab

    split, dropped = {}, []
    for task in tasks:
        eps = rng.permutation(ep_task[ep_task == task].index.to_numpy())
        if args.episodes_per_task:
            eps = eps[:args.episodes_per_task]
        if len(eps) < 2:
            # 한 에피소드짜리 지시문은 층화 분할이 성립하지 않는다 -- 학습과 검증
            # 어느 한쪽에만 들어가고, 그러면 그 지시문에 대해 두 집합이 비교되지
            # 않는다. 통째로 실패시키는 대신 그 지시문만 빼고 몇 개를 뺐는지
            # 남긴다. phase9 라벨에서는 334종 중 9종 · 408행(0.2%) 이다.
            dropped.append(task)
            continue
        nv = min(len(eps)-1, max(1, round(len(eps)*args.val_frac)))
        split.update({int(ep): ("val" if i < nv else "train") for i, ep in enumerate(eps)})
    if dropped:
        print(json.dumps({"dropped_single_episode_tasks": len(dropped),
                          "examples": dropped[:3]}, ensure_ascii=False), flush=True)
    if not split:
        raise ValueError("에피소드가 2개 이상인 지시문이 하나도 없다")
    lab = labels[labels.episode_index.isin(split)].copy()
    if args.frames_per_episode:
        groups = []
        for _, g in lab.groupby("episode_index"):
            g = g.sort_values("frame_index")
            idx = np.unique(np.linspace(0, len(g)-1, min(len(g), args.frames_per_episode)).astype(int))
            groups.append(g.iloc[idx])
        lab = pd.concat(groups)
    lab = lab.sort_values(keys).reset_index(drop=True)
    lab["split"] = lab.episode_index.map(split)
    if not len(lab) or set(lab.split) != {"train", "val"}:
        raise ValueError("empty training or validation split")
    return lab


class MotionFrames(Dataset):
    def __init__(self, labels, ds_root, cache_dir, embeddings, history):
        self.labels = labels.reset_index(drop=True)
        self.embeddings = embeddings
        self.maps, lookup = [], {}
        wanted = set(zip(labels.episode_index, labels.frame_index))
        for ix in sorted(Path(cache_dir).glob("index_shard*.parquet")):
            sid = ix.stem.replace("index_shard", "")
            rows = pd.read_parquet(ix)
            chosen = [r for r in rows.itertuples() if (r.episode_index, r.frame_index) in wanted]
            if not chosen:
                continue
            mm = np.load(Path(cache_dir)/f"frames_shard{sid}.u8", mmap_mode="r")
            if mm.dtype != np.uint8 or mm.shape[1:] != (9,128,128):
                raise ValueError("requires uint8 9x128x128 image cache")
            k = len(self.maps)
            self.maps.append(mm)
            for r in chosen:
                key = (r.episode_index, r.frame_index)
                if key in lookup or not 0 <= r.row < len(mm):
                    raise ValueError("duplicate cache key or out-of-range row")
                lookup[key] = (k, int(r.row))
        missing = wanted-set(lookup)
        if missing:
            raise ValueError(f"{len(missing)} labels missing from image cache; e.g. {sorted(missing)[:5]}")
        self.refs = [lookup[(r.episode_index, r.frame_index)] for r in labels.itertuples()]
        info = json.loads((Path(ds_root)/"meta/info.json").read_text())
        sd = info["features"]["observation.state"]["shape"][0]
        ad = info["features"]["action"]["shape"][0]
        self.states = np.empty((len(labels), sd), dtype=np.float32)
        self.actions = np.empty((len(labels), history, ad), dtype=np.float32)
        self.mask = np.empty((len(labels), history), dtype=np.float32)
        self.episode_files = []
        for ep, group in self.labels.groupby("episode_index"):
            rel = info["data_path"].format(episode_chunk=int(ep)//info["chunks_size"], episode_index=int(ep))
            p = Path(ds_root)/rel
            df = pd.read_parquet(p)
            if not (df.episode_index.to_numpy() == ep).all():
                raise ValueError("episode mismatch in source parquet")
            if "timestamp" in df and not np.array_equal(np.rint(df.timestamp.to_numpy()*info["fps"]).astype(int), np.arange(len(df))):
                raise ValueError("source rows are not in consecutive frame order")
            a = np.stack(df.action).astype(np.float32)
            s = np.stack(df["observation.state"]).astype(np.float32)
            if not np.isfinite(a).all() or not np.isfinite(s).all():
                raise ValueError("nonfinite state/action source")
            for i, r in group.iterrows():
                f = int(r.frame_index)
                if not 0 <= f < len(df):
                    raise ValueError("label frame outside episode")
                self.states[i] = s[f]
                self.actions[i], self.mask[i] = pack_history(a, f, history)
            self.episode_files.append({"path": str(p), "sha256": digest(p)})
        missing_text = set(labels.task)-set(embeddings)
        if missing_text:
            raise ValueError(f"missing instruction embeddings: {missing_text}")
        self.text = np.stack([embeddings[t] for t in labels.task]).astype(np.float32)
        if not np.isfinite(self.text).all():
            raise ValueError("invalid embeddings")
        self.targets = labels.p_yes.to_numpy(dtype=np.float32)
        self.images = None

    def preload(self):
        self.images = np.stack([self.maps[k][row] for k,row in self.refs])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        k, row = self.refs[i]
        im = self.images[i] if self.images is not None else self.maps[k][row]
        return (torch.from_numpy(im.astype(np.float32)/255), torch.from_numpy(self.text[i]),
                torch.from_numpy(self.states[i]), torch.from_numpy(self.actions[i]),
                torch.from_numpy(self.mask[i]), torch.tensor([self.targets[i]]))


def evaluate(model, loader, device):
    model.eval()
    scores, targets = [], []
    with torch.no_grad():
        for batch in loader:
            *inputs, y = [t.to(device) for t in batch]
            scores.extend(model(*inputs).sigmoid().cpu().numpy().ravel().tolist())
            targets.extend(y.cpu().numpy().ravel().tolist())
    s, y = np.asarray(scores), np.asarray(targets)
    binary = y >= .5
    return {"auc": float(roc_auc_score(binary, s)) if len(np.unique(binary)) == 2 else None,
            "bce": float(-(y*np.log(s.clip(1e-7,1-1e-7))+(1-y)*np.log((1-s).clip(1e-7,1-1e-7))).mean()),
            "agreement_at_0.5": float(((s >= .5) == binary).mean()),
            "prediction_std": float(s.std()), "prediction_min": float(s.min()),
            "prediction_max": float(s.max()), "count": len(s)}


def main():
    p = argparse.ArgumentParser()
    for name in ("dataset-path", "labels", "cache-dir", "task-emb", "out-dir"):
        p.add_argument("--"+name, required=True)
    # 두 팔을 같은 split/seed/에폭/선택기준으로 돌리기 위한 것. 입력만 다르다.
    p.add_argument("--arch", choices=("proprio", "baseline"), default="proprio",
                   help="proprio: 이미지+지시문+현재 proprio+직전 실행 액션, "
                        "baseline: 이미지+지시문")
    p.add_argument("--history", type=int, default=16)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    # 0 이면 나누지 않고 전량 학습한다. VLA 학습이 그렇듯 끝까지 돌리고 마지막을
    # 쓰며, 어느 구조가 나은지는 폐루프 eval 이 판정한다.
    p.add_argument("--val-frac", type=float, default=0.0)
    p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--episodes-per-task", type=int, default=0)
    p.add_argument("--frames-per-episode", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--preload", action="store_true", help="RAM cache for bounded smoke runs")
    args = p.parse_args()
    # val_frac 0 은 홀드아웃 없이 전량 학습이다. 음수나 1 이상은 여전히 오류다.
    if not 0 <= args.val_frac < 1 or args.history < 1 or args.epochs < 1 or args.bs < 1:
        raise ValueError("invalid training configuration")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)  # Never overwrite or silently resume.
    if not torch.cuda.is_available():
        raise RuntimeError("This training test must run inside a GPU allocation")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.set_num_threads(4)
    start = time.perf_counter()
    z = np.load(args.task_emb, allow_pickle=True)
    emb = {t: e for t,e in zip(z["tasks"], z["emb"])}
    labels = select_split(pd.read_parquet(args.labels), args)
    labels[["episode_index","frame_index","task","p_yes","split"]].to_parquet(out/"split.parquet", index=False)
    data = MotionFrames(labels, args.dataset_path, args.cache_dir, emb, args.history)
    if args.preload:
        data.preload()
    train_idx = np.flatnonzero(labels.split.to_numpy() == "train")
    val_idx = np.flatnonzero(labels.split.to_numpy() == "val")
    HOLDOUT = len(val_idx) > 0
    if args.arch == "baseline":
        # 이미지와 지시문만 본다. state/action 은 로더에서 그대로 오지만 무시된다 --
        # 데이터 경로를 공유해야 두 팔이 정확히 같은 행을 같은 순서로 본다.
        model = SmallGateBaseline(text_dim=data.text.shape[1])
    else:
        model = SmallGateProprioHistory(state_dim=data.states.shape[1],
                                        action_dim=data.actions.shape[-1],
                                        history=args.history, text_dim=data.text.shape[1])
        model.set_normalization(data.states[train_idx], data.actions[train_idx],
                                data.mask[train_idx])
    model.cuda()
    print(json.dumps({"arch": args.arch,
                      "parameters": sum(p_.numel() for p_ in model.parameters())}), flush=True)
    tr = DataLoader(torch.utils.data.Subset(data, train_idx), batch_size=args.bs, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True)
    va = (DataLoader(torch.utils.data.Subset(data, val_idx), batch_size=args.bs,
                     num_workers=args.num_workers, pin_memory=True)
          if HOLDOUT else None)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    setup_s = time.perf_counter()-start
    initial = evaluate(model, va, "cuda") if HOLDOUT else None
    print(json.dumps({"setup_seconds":setup_s,"train_rows":len(train_idx),"val_rows":len(val_idx),"initial":initial}), flush=True)
    metrics, gradients = [], {}
    best = float("inf")
    provenance = {"args":vars(args), "label_sha256":digest(args.labels),
                  "text_sha256":digest(args.task_emb), "episode_files":data.episode_files,
                  "model_source_sha256":digest(Path(__file__).with_name("model.py")),
                  "train_source_sha256":digest(__file__),
                  "dataset_info":json.loads((Path(args.dataset_path)/"meta/info.json").read_text()),
                  "dataset_modality":json.loads((Path(args.dataset_path)/"meta/modality.json").read_text())}
    write_json(out/"provenance.json", provenance)
    for epoch in range(args.epochs):
        model.train(); total = 0.; n = 0
        torch.cuda.synchronize(); tick=time.perf_counter()
        for batch in tr:
            *inputs,y = [t.cuda(non_blocking=True) for t in batch]
            opt.zero_grad(set_to_none=True)
            loss=loss_fn(model(*inputs),y)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            loss.backward()
            if not gradients:
                # 검사할 층은 모델에 있는 것만 고른다. baseline 에는 motion 가지가
                # 없으므로 이름을 박아 두면 그 팔에서만 터진다.
                named = dict(model.named_parameters())
                probe = [nm for nm in ("net.0.0.weight", "motion.0.weight",
                                       "head.0.weight") if nm in named]
                if len(probe) < 2:
                    raise RuntimeError(f"검사할 층을 못 찾았다: {sorted(named)[:5]}")
                for name in probe:
                    grad = named[name].grad
                    if grad is None or not torch.isfinite(grad).all() or not grad.abs().sum() > 0:
                        raise RuntimeError(f"missing/nonfinite/zero gradient: {name}")
                    gradients[name] = float(grad.norm())
            opt.step(); total += float(loss.detach())*len(y); n += len(y)
        torch.cuda.synchronize(); train_s=time.perf_counter()-tick
        tick=time.perf_counter(); v=evaluate(model,va,"cuda") if HOLDOUT else None
        item={"epoch":epoch+1,"train_bce":total/n,"train_seconds":train_s,
              "val_seconds":time.perf_counter()-tick,"val":v}
        metrics.append(item); print(json.dumps(item), flush=True)
        # 홀드아웃이 있으면 가장 좋은 에폭을, 없으면 마지막 에폭을 저장한다.
        # 같은 집합으로 고르고 채점하면 점수가 낙관적으로 나오므로, 비교가
        # 목적일 때는 홀드아웃 없이 마지막을 쓰는 쪽이 깨끗하다.
        if (v["bce"] < best) if HOLDOUT else (epoch + 1 == args.epochs):
            best = v["bce"] if HOLDOUT else float(total / n)
            torch.save({"model":model.state_dict(),"config":model.config,"epoch":epoch+1,
                        "res":128,"views":VIEW_KEYS,"task_emb_file":str(Path(args.task_emb).resolve()),
                        "args":vars(args),"val":v,"format":"proprio_history_v1",
                        "input_semantics":"state[f], action[max(0,f-H):f], left padding + mask"},out/"checkpoint.pt")
    summary={"job_id":os.getenv("SLURM_JOB_ID"),"gpu":torch.cuda.get_device_name(),
             "torch":torch.__version__,"params":sum(p.numel() for p in model.parameters()),
             "train_rows":len(train_idx),"val_rows":len(val_idx),
             "train_episodes":int(labels.iloc[train_idx].episode_index.nunique()),
             "val_episodes":int(labels.iloc[val_idx].episode_index.nunique()),
             "matched_rows":len(data),"requested_rows":len(labels),
             "setup_seconds":setup_s,"total_seconds":time.perf_counter()-start,
             "gradient_norms":gradients,"initial_val":initial,"epochs":metrics,
             "best_val_bce":best,"checkpoint":str(out/"checkpoint.pt"),
             "scope":"bounded wiring/training smoke, not a quality or closed-loop comparison"}
    write_json(out/"summary.json", summary)
    print("TRAINING_COMPLETE",out,flush=True)


if __name__ == "__main__":
    main()
