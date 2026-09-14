"""배달본 parquet 의 `hojin` 열(배속 K)을 서브태스크별 상한에 맞춰 비례로 내린다.

대상은 `meta/hojin_quantization_confidence.json` 이 아니라 **데이터셋 안의 열**이다:

    /rlwrld2/home/david/action_quantization/merged_v5tempo_hojin/
        data/chunk-*/episode_*.parquet   열 `hojin` = 배속 K (float32)
        meta/subtasks.jsonl              episode_index + start_frame/end_frame + label

`allex_rescale_ceiling.py` 는 못 쓴다. 그쪽은 청크마다 `p`(1단 신뢰도)와
`K_max` 가 저장된 JSON 을 다시 굽는 도구인데, 여기에는 **최종 K 만** 있어서
비례 이동의 기준을 p 에서 얻을 수 없다. 기준을 그 서브태스크의 현재 최대 K 로
잡는다.

두 가지 기준점이 있고 결과가 꽤 다르다.

    anchor=floor (기본)   K' = F + (K - F) * (T - F) / (C - F)
    anchor=one            K' = 1 + (K - 1) * (T - 1) / (C - 1)

배달본은 **하한 1.5 가 이미 지켜지고 있다.** 1 을 기준으로 줄이면 1.5 가
1.25 로 내려가 하한을 깨고, 그걸 다시 1.5 로 자르면 바닥이 뭉개진다. 그래서
기본은 하한을 고정점으로 두고 꼭대기만 내린다. 서브태스크 안의 상대 순서는
두 방식 모두 보존된다.

    python allex_rescale_hojin_column.py <데이터셋> --ceiling "Rotate Box=2.0,..." [--apply <출력디렉터리>]

`--apply` 없이 돌리면 읽기만 하고 바뀔 분포를 낸다. **입력을 제자리에서
덮어쓰지 않는다.**
"""
import argparse, collections, glob, json, os, shutil, sys
import numpy as np
import pandas as pd

FLOOR_DEFAULT = 1.5


def load_segments(root):
    seg = collections.defaultdict(list)
    with open(f"{root}/meta/subtasks.jsonl") as f:
        for line in f:
            r = json.loads(line)
            seg[r["episode_index"]].append((r["start_frame"], r["end_frame"], r["label"]))
    return seg


def episode_index(path):
    return int(path.split("episode_")[1][:6])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--ceiling", required=True,
                    help='"Bring Object=2.0,Pass Object=2.5,Rotate Box=2.0,Rotate PolyBag=2.5"')
    ap.add_argument("--floor", type=float, default=FLOOR_DEFAULT)
    ap.add_argument("--anchor", choices=("floor", "one"), default="floor")
    ap.add_argument("--column", default="hojin")
    ap.add_argument("--apply", default="",
                    help="쓸 디렉터리. 없으면 읽기만 하고 분포만 낸다.")
    ap.add_argument("--limit", type=int, default=0, help="앞 N 개 에피만 (점검용)")
    a = ap.parse_args()

    tgt = {}
    for part in filter(None, (x.strip() for x in a.ceiling.split(","))):
        lb, _, v = part.rpartition("=")
        tgt[lb.strip()] = float(v)

    seg = load_segments(a.root)
    files = sorted(glob.glob(f"{a.root}/data/*/*.parquet"))
    if a.limit:
        files = files[:a.limit]
    print(f"[i] 에피 {len(files)} · 구간 {sum(len(v) for v in seg.values())} · "
          f"기준점 {a.anchor} · 하한 {a.floor}", flush=True)

    # --- 1차: 서브태스크별 현재 최대 ---
    cur = collections.defaultdict(float)
    before = collections.defaultdict(list)
    for p in files:
        ei = episode_index(p)
        if ei not in seg:
            continue
        h = pd.read_parquet(p, columns=[a.column])[a.column].values
        for s, e, lb in seg[ei]:
            v = h[s:min(e, len(h))]
            if not len(v):
                continue
            cur[lb] = max(cur[lb], float(v.max()))
            before[lb].append(v)
    miss = [lb for lb in cur if lb not in tgt]
    if miss:
        sys.exit(f"상한이 안 주어진 서브태스크: {miss}")

    fac = {}
    for lb, c in cur.items():
        base = a.floor if a.anchor == "floor" else 1.0
        if c - base <= 1e-9:
            fac[lb] = 1.0
        else:
            fac[lb] = (tgt[lb] - base) / (c - base)

    # --- 2차: 적용 ---
    after = collections.defaultdict(list)
    base = a.floor if a.anchor == "floor" else 1.0
    if a.apply:
        os.makedirs(a.apply, exist_ok=True)
        for sub in ("meta", "videos"):
            src, dst = f"{a.root}/{sub}", f"{a.apply}/{sub}"
            if os.path.isdir(src) and not os.path.exists(dst):
                os.symlink(src, dst)
    for p in files:
        ei = episode_index(p)
        if ei not in seg:
            continue
        d = pd.read_parquet(p) if a.apply else None
        h = (d[a.column].values.copy() if a.apply
             else pd.read_parquet(p, columns=[a.column])[a.column].values.copy())
        for s, e, lb in seg[ei]:
            j = min(e, len(h))
            if j <= s:
                continue
            v = h[s:j]
            nv = base + (v - base) * fac[lb]
            nv = np.clip(nv, a.floor, tgt[lb])
            h[s:j] = nv
            after[lb].append(nv)
        if a.apply:
            d[a.column] = h.astype(np.float32)
            out = f"{a.apply}/data/{os.path.basename(os.path.dirname(p))}"
            os.makedirs(out, exist_ok=True)
            d.to_parquet(f"{out}/{os.path.basename(p)}", index=False)

    print(f"\n{'서브태스크':18s} {'현재최대':>8s} {'새상한':>7s} {'배율':>6s} "
          f"{'평균 전':>7s} {'후':>6s} {'최대 후':>7s}")
    for lb in sorted(before, key=lambda x: -sum(len(v) for v in before[x])):
        b = np.concatenate(before[lb])
        n = np.concatenate(after[lb]) if after[lb] else b
        print(f"  {lb:16s} {cur[lb]:8.2f} {tgt[lb]:7.2f} {fac[lb]:6.3f} "
              f"{b.mean():7.3f} {n.mean():6.3f} {n.max():7.2f}")
    B = np.concatenate([np.concatenate(v) for v in before.values()])
    A = np.concatenate([np.concatenate(v) for v in after.values()])
    print(f"\n전체 평균 배속 {B.mean():.4f} -> {A.mean():.4f}")
    if a.apply:
        print(f"기록 -> {a.apply}  (meta·videos 는 원본 심볼릭 링크)")
    else:
        print("읽기만 함. 쓰려면 --apply <출력디렉터리>")


if __name__ == "__main__":
    main()
