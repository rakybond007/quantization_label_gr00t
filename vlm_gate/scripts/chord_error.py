"""압축이 **경로를 얼마나 질러가는지** 액션만으로 잰다. VLM 도 GPU 도 안 쓴다.

    python vlm_gate/scripts/chord_error.py robocasa --max-episodes 400 --stride 4

## 무엇을 재나

`compress_chunk` 는 블록 안의 델타를 **더해서 한 스텝에 실행한다**
(`block.sum(axis=0)`, 그리퍼 같은 이산 키만 `block[-1]` 로 래치). 그러니
블록의 시작과 끝은 그대로인데 **그 사이를 안 지나고 직선으로 질러간다.**

    p[0] = 0,  p[j] = 델타 누적            원래 지나야 했던 EE 경로
    블록 [s,e) -> 압축 후엔 p[s] 에서 p[e] 로 한 번에 간다
    벗어남 = max_{s<j<e}  점 p[j] 와 선분 p[s]p[e] 사이 거리
    현오차(f,K) = 그 청크의 블록들 중 최대

블록 나누기는 `fractional_blocks.block_sizes` 를 그대로 쓴다 -- 라벨·평가와
같은 규칙이어야 한다.

세 가지를 따로 낸다. 하나로 뭉치면 무엇이 큰지 안 보인다.

    pos    위치(5:7) 코드 오차      -- 물건을 놓치거나 부딪히는 쪽
    rot    회전(8:10) 코드 오차     -- 자세가 빗나가는 쪽
    grip   그리퍼 래치 밀림(스텝)   -- 쥐고 놓는 시점이 최대 K-1 늦거나 이르다

## 무엇을 못 재나

**명령 경로지 실제 경로가 아니다.** 컨트롤러(OSC)와 ±1 클립이 사이에 있어서
실제 벗어남은 이보다 작다. 그래도 순서를 매기는 데는 쓸 수 있고, 상한과
맞춰 볼 때 상수배는 흡수된다. robocasa 는 클립을 풀어도 성공률이 안 움직인다.

**단위가 cm 가 아니다.** 액션은 정규화된 델타(CLIP=1.0)다. 그래서 절대값이
아니라 **태스크 사이 비교**로 읽는다.

## 무엇을 보려는 것인가

이 값이 배속을 정하는 틀의 절반이다. 나머지 절반(이 순간 얼마나 벗어나도
되는가)은 화면을 봐야 알고, 그것만 VLM 에 남긴다.

그래서 여기서 확인할 것은 둘이다.

    1. 태스크 안에서 현오차가 갈리는가   -- 안 갈리면 시점별 배속을 못 만든다.
                                          지금 배속이 딱 그래서 24태스크 중
                                          14개가 한 종류다
    2. 태스크 사이 현오차가 실측 상한과 관계가 있는가

**2번이 0 이면 이 틀은 틀린 것이다.** 1번이 낮으면 틀은 맞아도 이 신호로는
시점을 못 가른다. 둘 다 서면 다음으로 간다.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fractional_blocks import block_sizes          # noqa: E402

DS = {"robocasa": "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/"
                  "kimtaey/robocasa_mg_gr00t_300"}
POS, ROT, GRIP = slice(5, 8), slice(8, 11), 11
KS = (1.5, 2.0, 2.5)


def seg_dev(p):
    """점들이 시작-끝 선분에서 얼마나 벗어나는가. p:(n,3) -> 최대 거리.

    블록 안의 모든 중간 점을 본다. 평균이 아니라 **최대**인 이유는, 한 점만
    크게 벗어나도 그 점에서 부딪히거나 놓치기 때문이다.
    """
    if len(p) < 3:
        return 0.0
    v = p[-1] - p[0]
    L = float(np.linalg.norm(v))
    q = p[1:-1] - p[0]
    if L < 1e-9:                    # 제자리 -- 선분이 없으니 시작점에서의 거리
        return float(np.linalg.norm(q, axis=1).max())
    t = np.clip(q @ v / (L * L), 0.0, 1.0)
    return float(np.linalg.norm(q - t[:, None] * v[None, :], axis=1).max())


def chunk_error(a, f, K, horizon=16):
    """프레임 f 청크를 K 로 묶었을 때의 (위치오차, 회전오차, 그리퍼 밀림)."""
    w = a[f:f + horizon]
    if len(w) < horizon:
        return None
    sizes = block_sizes(horizon, K)
    pos = np.vstack([np.zeros(3), np.cumsum(w[:, POS], axis=0)])
    rot = np.vstack([np.zeros(3), np.cumsum(w[:, ROT], axis=0)])
    g = w[:, GRIP]
    ep_, er_, eg_ = 0.0, 0.0, 0.0
    s = 0
    for n in sizes:
        e = s + n
        ep_ = max(ep_, seg_dev(pos[s:e + 1]))
        er_ = max(er_, seg_dev(rot[s:e + 1]))
        if n > 1:
            # 래치는 블록의 마지막 값을 쓴다. 블록 안에서 그리퍼가 바뀌면
            # 그 전환이 블록 끝까지 밀린다 -- 밀린 스텝 수를 센다.
            blk = g[s:e]
            ch = np.flatnonzero(np.abs(np.diff(blk)) > 0.5)
            if len(ch):
                eg_ = max(eg_, float(n - 1 - ch[0]))
        s = e
    return ep_, er_, eg_


def selftest():
    """손으로 아는 답 넷. 데이터셋 없이 돈다.

    코드를 믿기 전에 이걸 본다 -- 선분 거리와 블록 나누기는 조용히 틀리기 쉽고,
    틀려도 그럴듯한 숫자가 나온다.
    """
    ok = True

    def chk(name, got, want, tol=1e-6):
        nonlocal ok
        good = abs(got - want) <= tol
        ok = ok and good
        print(f"  {'ok ' if good else '틀림'} {name:28s} {got:.6f}  기대 {want:.6f}")

    chk("직선은 0", seg_dev(np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]], float)), 0.0)
    # (0,0)->(1,0)->(1,1) 을 (0,0)->(1,1) 로 질러가면 중간점 거리는 1/sqrt2
    chk("직각 모서리 1/√2",
        seg_dev(np.array([[0,0,0],[1,0,0],[1,1,0]], float)), 2 ** -0.5)
    # 제자리에서 나갔다 오면 선분이 없다 -- 시작점에서의 거리
    chk("왕복은 나간 거리",
        seg_dev(np.array([[0,0,0],[0,0,2],[0,0,0]], float)), 2.0)

    # 16스텝: 앞 8 +x, 뒤 8 +y. 모서리를 가로지르는 블록만 오차를 낸다.
    a = np.zeros((40, 12)); a[:8, 5] = 1.0; a[8:16, 6] = 1.0
    prev = -1.0
    for K in (1.5, 2.0, 2.5):
        e = chunk_error(a, 0, K)[0]
        print(f"  ..  ㄱ자 청크 K={K:<4} 위치오차 {e:.4f}")
        if e < prev - 1e-9:
            print("  틀림 K 가 커지는데 오차가 줄었다"); ok = False
        prev = e
    if prev <= 0:
        print("  틀림 꺾인 경로인데 오차가 0 이다"); ok = False

    # 그리퍼: 5스텝째 전환. K=2 면 블록 [4,6) 안이라 1스텝 밀린다.
    a2 = np.zeros((40, 12)); a2[:, 5] = 1.0; a2[5:, 11] = 1.0
    chk("그리퍼 래치 밀림 K=2", chunk_error(a2, 0, 2.0)[2], 1.0)

    print("\n자체검사 " + ("통과" if ok else "실패 -- 여기서 멈춘다"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench", choices=list(DS))
    ap.add_argument("--max-episodes", type=int, default=400)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true",
                    help="손으로 아는 답 넷만 확인하고 끝낸다. 데이터셋 안 읽는다")
    a = ap.parse_args()
    print("== 자체검사 ==")
    if selftest():
        return 1                      # 계산이 틀렸으면 실측을 낼 이유가 없다
    if a.selftest:
        return 0
    print()
    root = DS[a.bench]

    ep2t = {}
    for line in open(f"{root}/meta/episodes.jsonl"):
        d = json.loads(line)
        for v in (d.get("tasks") or [])[1:]:
            if isinstance(v, str) and v and " " not in v and v != "Valid":
                ep2t[d["episode_index"]] = v
                break

    # 태스크마다 고르게 뽑는다. 앞에서 자르면 한두 태스크만 들어온다.
    per = max(1, a.max_episodes // max(1, len(set(ep2t.values()))))
    picked, seen = [], {}
    for ep in sorted(ep2t):
        t = ep2t[ep]
        if seen.get(t, 0) < per:
            seen[t] = seen.get(t, 0) + 1
            picked.append(ep)

    rows = []
    for i, ep in enumerate(picked):
        try:
            act = np.stack(pd.read_parquet(
                f"{root}/data/chunk-{ep // 1000:03d}/"
                f"episode_{ep:06d}.parquet")["action"].values).astype(np.float64)
        except Exception:
            continue
        for f in range(0, max(0, len(act) - 16), a.stride):
            for K in KS:
                r = chunk_error(act, f, K)
                if r:
                    rows.append((ep2t[ep], ep, f, K, *r))
        if (i + 1) % 50 == 0:
            print(f"  에피 {i+1}/{len(picked)} · 행 {len(rows):,}", flush=True)

    df = pd.DataFrame(rows, columns=["task", "ep", "f", "K", "pos", "rot", "grip"])
    print(f"\n행 {len(df):,} · 에피 {df.ep.nunique()} · 태스크 {df.task.nunique()}\n")

    ceil = json.load(open(os.path.join(HERE, "..", "analysis",
                                       f"{a.bench}_task_ceilings.json")))
    hi = {t: (v[1] if isinstance(v, list) else v) for t, v in ceil.items()}

    for K in KS:
        d = df[df.K == K]
        print(f"===== K = {K} =====")
        # 1) 태스크 안에서 갈리는가
        for col in ("pos", "rot", "grip"):
            g = d[col].to_numpy()
            tm = d.groupby("task")[col].transform("mean").to_numpy()
            wf = np.var(g - tm) / (np.var(g) or 1e-12)
            print(f"  {col:5s} 태스크안 분산몫 {wf:5.3f}   "
                  f"중앙 {np.median(g):7.4f}  p90 {np.percentile(g,90):7.4f}  "
                  f"최대 {g.max():7.4f}")
        # 2) 태스크 사이 -- 상한과 관계가 있는가
        agg = d.groupby("task")["pos"].quantile(0.90)
        common = [t for t in agg.index if t in hi]
        x = np.array([agg[t] for t in common])
        y = np.array([hi[t] for t in common])
        c = dd = 0
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                s = (x[i]-x[j]) * (y[i]-y[j])
                c += s > 0
                dd += s < 0
        tau = (c - dd) / max(1, c + dd)
        print(f"  태스크별 pos p90 대 실측 상한 · 켄달 타우 {tau:+.3f}  "
              f"(음수여야 한다: 많이 질러가는 태스크일수록 상한이 낮다)")
        for t in sorted(common, key=lambda t: -agg[t])[:5]:
            print(f"      {t:24s} p90 {agg[t]:7.4f}  상한 {hi[t]}")
        for t in sorted(common, key=lambda t: agg[t])[:3]:
            print(f"      {t:24s} p90 {agg[t]:7.4f}  상한 {hi[t]}")
        print()

    if a.out:
        df.to_parquet(a.out, index=False)
        print(f"-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
