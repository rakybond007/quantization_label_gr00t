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
from torch.utils.data import DataLoader, Dataset

from model import GRID, RatioHead, pack_planned

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
    bad = sorted(set(np.round(labels.ratio.unique(), 6)) - set(GRID))
    if bad:
        raise ValueError(f"눈금 밖의 배속: {bad} (허용 {GRID})")
    if "fixed" not in labels.columns:
        raise ValueError("fixed 열이 없다 -- 접촉 가드 판이 아니다. "
                         "가드 없는 판으로 학습하면 1.0 이 왜 나오는지 못 가른다")
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
                self.actions[i], self.mask[i] = pack_planned(a, f, history)
            self.episode_files.append({"path": str(p), "sha256": digest(p)})
        # **지시문으로 찾는다, 태스크 이름이 아니라.** 이 머리는 정책 위에 얹히고
        # 정책이 받는 것은 `TurnOnStove` 라는 분류명이 아니라 "turn on the front
        # right burner of the stove" 라는 문장이다. 태스크로 뭉치면 배치할 때
        # 학습에서 못 본 입력을 받게 되고, 한 태스크 안의 좌우 버너가 한 벡터가
        # 된다. 임베딩 파일도 지시문 단위(334종)로 되어 있다.
        #
        # 옛 라벨 파일에는 `instruction` 열이 없다. 그때는 태스크로 떨어지되
        # 무엇으로 찾았는지 남긴다 -- 두 방식의 점수를 나중에 비교하려면 어느
        # 쪽이었는지 알아야 한다.
        self.text_col = "instruction" if "instruction" in labels.columns else "task"
        keys = labels[self.text_col].tolist()
        missing_text = set(keys)-set(embeddings)
        if missing_text:
            raise ValueError(
                f"임베딩에 없는 {self.text_col} {len(missing_text)}개: "
                f"{sorted(str(m) for m in missing_text)[:5]}")
        self.text = np.stack([embeddings[t] for t in keys]).astype(np.float32)
        if not np.isfinite(self.text).all():
            raise ValueError("invalid embeddings")
        # 배속을 눈금 인덱스로. 가드는 따로 낸다 -- 라벨의 16.6% 가 가드로
        # 1.0 에 박힌 것이라, 한 머리로 뭉뚱그리면 "왜 1.0 인가" 가 안 갈린다.
        g = np.asarray(GRID, dtype=np.float32)
        r = labels.ratio.to_numpy(dtype=np.float32)
        self.targets = np.abs(r[:, None] - g[None, :]).argmin(1).astype(np.int64)
        self.gate = labels.fixed.to_numpy(dtype=np.float32)
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
                torch.from_numpy(self.mask[i]),
                torch.tensor(self.targets[i]), torch.tensor(self.gate[i]))


def text_floor(labels, train_idx, val_idx):
    """**지시문만 보고** 배속을 맞히면 몇 점인가 -- 학습 없이 세는 바닥선.

    배속 라벨은 태스크별 상한 띠 안에서 신뢰도로 자리를 잡는다. 그래서 지시문
    하나가 이미 그 태스크의 흔한 칸을 거의 정해 놓는다. 지시문 벡터를 384차원
    으로 통째로 넣어 주면, 모델은 그림도 액션도 안 보고 **"이 지시문은 보통
    1.5x"** 만 외워도 점수가 나온다. 그러면 시점을 가르는 머리가 아니라 태스크
    이름을 읽는 머리가 된다.

    그래서 학습 쪽 다수결을 그대로 홀드아웃에 적용해 본다. 학습한 머리가 이
    선을 못 넘으면 **그림과 액션에서 아무것도 못 읽은 것이다.** 넘은 만큼이
    시점에서 온 몫이다. 못 본 지시문은 전체 다수결로 답한다.
    """
    for col in ("instruction", "text_key", "task"):
        if col in labels.columns:
            break
    key = labels[col].to_numpy()
    y = labels["_y"].to_numpy()
    if len(val_idx) == 0:
        return {"key": col, "note": "홀드아웃이 없어 못 잰다"}
    tr, va = np.asarray(train_idx), np.asarray(val_idx)
    glob = int(np.bincount(y[tr], minlength=len(GRID)).argmax())
    vote = {}
    for k in np.unique(key[tr]):
        vote[k] = int(np.bincount(y[tr][key[tr] == k], minlength=len(GRID)).argmax())
    pred = np.array([vote.get(k, glob) for k in key[va]])
    unseen = int(sum(k not in vote for k in key[va]))
    return {"key": col, "n_keys": len(vote), "unseen_val_rows": unseen,
            "acc": float((pred == y[va]).mean()),
            "acc_majority_only": float((y[va] == glob).mean())}


def evaluate(model, loader, device):
    """배속 분류와 가드를 따로 잰다.

    이진 AUC 는 안 쓴다 -- 네 칸 분류라 뜻이 없고, sklearn 을 끌어오면 이 환경에
    없다. **칸별 정확도를 따로 본다.** 전체 정확도만 보면 흔한 칸(1.0 이 34.8%)
    으로 쏠린 모델이 좋아 보인다.
    """
    model.eval()
    P, Y, G, GP = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            *inputs, y, gate = [t.to(device) for t in batch]
            rl, gl = model(*inputs)
            P.extend(rl.argmax(1).cpu().numpy().tolist())
            Y.extend(y.cpu().numpy().tolist())
            GP.extend(torch.sigmoid(gl).cpu().numpy().tolist())
            G.extend(gate.cpu().numpy().tolist())
    p, y = np.asarray(P), np.asarray(Y)
    gp, g = np.asarray(GP), np.asarray(G)
    out = {"count": int(len(y)), "ratio_acc": float((p == y).mean())}
    for i, v in enumerate(GRID):
        m = y == i
        out[f"acc_{v}x"] = float((p[m] == y[m]).mean()) if m.any() else None
        out[f"n_{v}x"] = int(m.sum())
        out[f"pred_share_{v}x"] = float((p == i).mean())
    if len(np.unique(g > .5)) == 2:
        out["gate_agreement"] = float(((gp >= .5) == (g > .5)).mean())
        tp = float(((gp >= .5) & (g > .5)).sum())
        out["gate_recall"] = tp / max(1.0, float((g > .5).sum()))
        out["gate_precision"] = tp / max(1.0, float((gp >= .5).sum()))
    return out


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
    p.add_argument("--gate-weight", type=float, default=0.5,
                   help="접촉 가드 손실의 무게. 배속 분류가 본체이므로 1 보다 작게 둔다")
    p.add_argument("--max-steps", type=int, default=0,
                   help=">0 이면 에폭 대신 스텝 수로 끝낸다")
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
    # 바닥선이 학습과 같은 칸 번호를 쓰도록 여기서 한 번만 만든다.
    labels["_y"] = np.abs(labels.ratio.to_numpy(dtype=np.float32)[:, None]
                          - np.asarray(GRID, dtype=np.float32)[None, :]).argmin(1)
    # `p_yes` 는 이전 게이트 실험의 열이라 배속 라벨에는 없다. 스키마에 있는
    # 것만 적는다. `ratio` 와 `fixed` 를 같이 남기는 것은 뜻도 맞다 -- 어느 행이
    # 가드로 1.0 에 박힌 것인지 split 파일만 보고 알 수 있어야 나중에 칸별
    # 정확도를 다시 볼 때 편하다.
    keep = ["episode_index", "frame_index", "task", "split"]
    keep += [c for c in ("instruction", "ratio", "fixed", "conf")
             if c in labels.columns]
    labels[keep].to_parquet(out / "split.parquet", index=False)
    data = MotionFrames(labels, args.dataset_path, args.cache_dir, emb, args.history)
    if args.preload:
        data.preload()
    train_idx = np.flatnonzero(labels.split.to_numpy() == "train")
    val_idx = np.flatnonzero(labels.split.to_numpy() == "val")
    HOLDOUT = len(val_idx) > 0
    if False:
        # 이미지와 지시문만 본다. state/action 은 로더에서 그대로 오지만 무시된다 --
        # 데이터 경로를 공유해야 두 팔이 정확히 같은 행을 같은 순서로 본다.
        model = SmallGateBaseline(text_dim=data.text.shape[1])
    else:
        model = RatioHead(state_dim=data.states.shape[1],
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
    # 배속은 네 칸 분류, 가드는 이진. 둘을 따로 낸다.
    #
    # **칸마다 수가 다르다** -- 1.0 이 34.8% 인데 2.5 는 6.5% 다. 그냥 두면
    # 흔한 칸으로 쏠린다. 빈도의 역수로 가중해 드문 칸이 묻히지 않게 한다.
    cnt = np.bincount(data.targets, minlength=len(GRID)).astype(np.float64)
    w = (cnt.sum() / np.maximum(cnt, 1)) ** 0.5      # 완전 역수는 과교정이라 제곱근
    w = w / w.mean()
    print(json.dumps({"ratio_counts": cnt.astype(int).tolist(),
                      "class_weight": [round(float(v), 3) for v in w]}), flush=True)
    ratio_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(w, dtype=torch.float32, device=device))
    gate_fn = torch.nn.BCEWithLogitsLoss()
    GATE_W = float(args.gate_weight)
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
    # 스텝으로 끝낼 수 있게 한다. 데이터 크기가 판마다 달라 에폭 수로는
    # 같은 학습량을 못 맞춘다.
    steps_done = 0
    per_epoch = max(1, len(tr))
    n_epoch = (args.epochs if not args.max_steps
               else int(np.ceil(args.max_steps / per_epoch)))
    print(json.dumps({"rows": len(data), "bs": args.bs, "steps_per_epoch": per_epoch,
                      "max_steps": args.max_steps or per_epoch * args.epochs,
                      "epochs_planned": n_epoch,
                      "epochs_arg": args.epochs}), flush=True)
    for epoch in range(n_epoch):
        model.train(); total = 0.; n = 0
        torch.cuda.synchronize(); tick=time.perf_counter()
        for batch in tr:
            *inputs, y, gate = [t.cuda(non_blocking=True) for t in batch]
            opt.zero_grad(set_to_none=True)
            rl, gl = model(*inputs)
            loss = ratio_fn(rl, y) + GATE_W * gate_fn(gl, gate)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite loss")
            loss.backward()
            if not gradients:
                # 검사할 층은 모델에 있는 것만 고른다. baseline 에는 motion 가지가
                # 없으므로 이름을 박아 두면 그 팔에서만 터진다.
                named = dict(model.named_parameters())
                probe = [nm for nm in ("net.0.0.weight", "motion.0.weight",
                                       "shared.0.weight", "ratio_head.weight",
                                       "gate_head.weight") if nm in named]
                if len(probe) < 2:
                    raise RuntimeError(f"검사할 층을 못 찾았다: {sorted(named)[:5]}")
                for name in probe:
                    grad = named[name].grad
                    if grad is None or not torch.isfinite(grad).all() or not grad.abs().sum() > 0:
                        raise RuntimeError(f"missing/nonfinite/zero gradient: {name}")
                    gradients[name] = float(grad.norm())
            opt.step(); total += float(loss.detach())*len(y); n += len(y)
            steps_done += 1
            if args.max_steps and steps_done >= args.max_steps:
                break
        torch.cuda.synchronize(); train_s=time.perf_counter()-tick
        tick=time.perf_counter(); v=evaluate(model,va,"cuda") if HOLDOUT else None
        item={"epoch":epoch+1,"train_bce":total/n,"train_seconds":train_s,
              "val_seconds":time.perf_counter()-tick,"val":v}
        metrics.append(item); print(json.dumps(item), flush=True)
        # 홀드아웃이 있으면 가장 좋은 에폭을, 없으면 마지막 에폭을 저장한다.
        # 같은 집합으로 고르고 채점하면 점수가 낙관적으로 나오므로, 비교가
        # 목적일 때는 홀드아웃 없이 마지막을 쓰는 쪽이 깨끗하다.
        #
        # **`n_epoch` 로 센다, `args.epochs` 가 아니라.** `--max-steps` 를 주면
        # 실제로 도는 에폭은 `ceil(max_steps/per_epoch)` 인데 여기서 기본값 10 과
        # 견주면 두 가지로 조용히 틀린다. 도는 에폭이 10보다 적으면 한 번도
        # 저장되지 않고, 많으면 10에폭짜리를 마지막이라고 저장한다. 뒤엣것은
        # 터지지도 않아서 30k스텝 결과라고 들고 나오게 된다.
        if (v["bce"] < best) if HOLDOUT else (epoch + 1 == n_epoch):
            best = v["bce"] if HOLDOUT else float(total / n)
            torch.save({"model":model.state_dict(),"config":model.config,"epoch":epoch+1,
                        "res":128,"views":VIEW_KEYS,"task_emb_file":str(Path(args.task_emb).resolve()),
                        "args":vars(args),"val":v,"format":"proprio_history_v1",
                        "input_semantics":"state[f], action[max(0,f-H):f], left padding + mask"},out/"checkpoint.pt")
    # 저장이 한 번도 안 됐으면 여기서 크게 죽는다. 조건이 어긋나면 `best` 가
    # `inf` 인 채로 JSON 을 쓰다 엉뚱한 자리에서 터지는데, 그러면 원인이 안
    # 보인다. 산출물이 없다는 것을 산출물 이름으로 말한다.
    if not (out / "checkpoint.pt").exists():
        raise RuntimeError(
            f"{n_epoch}에폭을 다 돌았는데 checkpoint.pt 가 없다 -- 저장 조건이 "
            f"한 번도 안 맞았다 (holdout={HOLDOUT}, best={best})")

    floor = text_floor(labels, train_idx, val_idx)
    print(json.dumps({"text_floor": floor}, ensure_ascii=False), flush=True)
    summary={"job_id":os.getenv("SLURM_JOB_ID"),"gpu":torch.cuda.get_device_name(),
             "text_floor":floor,"text_key":data.text_col,
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
