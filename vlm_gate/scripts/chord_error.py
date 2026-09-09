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

블록 나누기는 `fractional_blocks.blocks_for` 다 -- 라벨 눈금 1.5·2.5 를 실제로
실행하는 경로가 그쪽이다. `block_sizes` 는 옛 고정 K 경로라 2.5 를 요청하면
2.286 이 나오므로 그것으로 재면 실행되지 않는 경계를 재게 된다.

**오차는 K 에 단조가 아니다.** 경로가 꺾이는 자리가 블록 경계에 떨어지면 모든
블록이 직선이라 0 이고, 경계 안쪽에 들면 크다. 그래서 위상(carry)을 훑어
평균과 최대를 같이 낸다.

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
# **`blocks_for` 를 쓴다, `block_sizes` 가 아니라.** 둘은 다른 실행 경로를
# 모사한다 -- `block_sizes` 는 옛 고정 K 경로(`compress_chunk`, 꼬리를 낱개로
# 붙임)라 2.5 를 요청하면 실제로 2.286 이 나오고, `blocks_for` 는 분수 배속
# 경로라 2.5 를 낸다. 라벨 눈금 1.5·2.5 를 실제로 실행하는 것은 뒤엣것이므로
# 앞엣것으로 재면 **실행되지 않는 경계**를 재게 된다. 정수 배속에서는 둘이
# 같으므로 `blocks_for` 하나로 전부 덮인다. 동료가 자체검사에서 잡았다.
from fractional_blocks import blocks_for          # noqa: E402

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


# 블록 경계는 청크 사이로 넘어오는 잔여(carry)에 따라 움직인다. 그래서 같은
# 순간이라도 언제 그 순간에 닿았느냐에 따라 경계가 다르게 떨어진다.
#
# **오차는 K 에 단조가 아니다.** 경로가 꺾이는 자리가 블록 경계에 떨어지면 모든
# 블록이 직선이라 질러갈 것이 없어 0 이고, 경계 안쪽에 들면 크다. 그러니 한
# 위상만 재면 그 순간이 안전한지 위험한지가 우연에 달린다. 여러 위상을 훑어
# **평균과 최대를 같이** 낸다 -- 평균은 그 순간의 기대 손상이고, 최대는 위상을
# 고를 수 없을 때 각오해야 하는 값이다.
CARRIES = (-0.4, -0.2, 0.0, 0.2, 0.4)


def chunk_error_at(a, f, K, carry, horizon=16):
    """한 위상에서의 (위치오차, 회전오차, 그리퍼 밀림)."""
    w = a[f:f + horizon]
    blocks, _ = blocks_for(horizon, K, carry)
    pos = np.vstack([np.zeros(3), np.cumsum(w[:, POS], axis=0)])
    rot = np.vstack([np.zeros(3), np.cumsum(w[:, ROT], axis=0)])
    g = w[:, GRIP]
    ep_, er_, eg_ = 0.0, 0.0, 0.0
    for s, e in blocks:
        ep_ = max(ep_, seg_dev(pos[s:e + 1]))
        er_ = max(er_, seg_dev(rot[s:e + 1]))
        if e - s > 1:
            # 래치는 블록의 마지막 값을 쓴다. 블록 안에서 그리퍼가 바뀌면
            # 그 전환이 블록 끝까지 밀린다 -- 밀린 스텝 수를 센다.
            ch = np.flatnonzero(np.abs(np.diff(g[s:e])) > 0.5)
            if len(ch):
                eg_ = max(eg_, float(e - s - 1 - ch[0]))
    return ep_, er_, eg_


def chunk_error(a, f, K, horizon=16, carries=CARRIES):
    """위상을 훑은 (위치 평균, 위치 최대, 회전 평균, 그리퍼 최대)."""
    if len(a[f:f + horizon]) < horizon:
        return None
    v = np.array([chunk_error_at(a, f, K, c, horizon) for c in carries])
    return (float(v[:, 0].mean()), float(v[:, 0].max()),
            float(v[:, 1].mean()), float(v[:, 2].max()))


def selftest():
    """손으로 아는 답. 데이터셋 없이 돈다.

    코드를 믿기 전에 이걸 본다 -- 선분 거리와 블록 경계는 조용히 틀리기 쉽고,
    틀려도 그럴듯한 숫자가 나온다.

    **단조성은 검사하지 않는다.** 오차가 K 에 단조라고 가정했다가 첫 판에서
    걸렸는데, 안 단조인 것이 맞았다 -- 꺾이는 자리가 블록 경계에 떨어지면 0 이다.
    그 성질을 버그로 잡는 대신 **성질 그대로 검사한다.**
    """
    ok = True

    def chk(name, got, want, tol=1e-6):
        nonlocal ok
        good = abs(got - want) <= tol
        ok = ok and good
        print(f"  {'ok ' if good else '틀림'} {name:30s} {got:.6f}  기대 {want:.6f}")

    chk("직선은 0", seg_dev(np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]], float)), 0.0)
    chk("직각 모서리 1/√2",
        seg_dev(np.array([[0,0,0],[1,0,0],[1,1,0]], float)), 2 ** -0.5)
    chk("왕복은 나간 거리",
        seg_dev(np.array([[0,0,0],[0,0,2],[0,0,0]], float)), 2.0)

    # 요청 배속이 실현되는가 -- 이것이 첫 판에서 틀렸던 자리다
    # **한 청크로 재면 안 된다.** `blocks_for` 는 잔여를 다음 청크로 넘겨
    # 수렴하는 함수라 16스텝 하나로는 2.5 를 아예 못 만든다(6블록 2.667,
    # 7블록 2.286). 첫 판이 한 청크로 재서 2.5 만 걸렸는데, K=1.5 는 1.4545 로
    # 허용오차에 우연히 들어가 통과했다. 하나만 걸리면 재는 방식이 틀렸다고
    # 안 보이고 그 값 계산이 틀렸다고 보인다. 동료가 잡았다.
    for K in (1.5, 2.0, 2.5):
        carry, n = 0.0, 0
        for _ in range(20):
            b, carry = blocks_for(16, K, carry)
            n += len(b)
        chk(f"blocks_for 실현 배속 K={K} (20청크)", 20 * 16 / n, K, tol=0.02)

    def corner(at):
        """at 스텝째에 +x 에서 +y 로 꺾이는 16스텝 청크."""
        a = np.zeros((40, 12))
        a[:at, 5] = 1.0
        a[at:16, 6] = 1.0
        return a

    # 모서리가 경계에 떨어지면 0 -- 버그가 아니라 성질이다
    chk("모서리가 경계에 있으면 0",
        chunk_error_at(corner(8), 0, 2.0, 0.0)[0], 0.0)
    # 경계 안쪽에 들면 1/√2
    chk("모서리가 블록 안이면 1/√2",
        chunk_error_at(corner(7), 0, 2.0, 0.0)[0], 2 ** -0.5)

    # 위상을 훑으면 어느 K 에서도 꺾임이 잡힌다 -- 한 위상만 보면 놓친다
    for K in (1.5, 2.0, 2.5):
        mean, mx, _, _ = chunk_error(corner(7), 0, K)
        print(f"  ..  ㄱ자(모서리 7) K={K:<4} 평균 {mean:.4f}  최대 {mx:.4f}")
        if mx <= 0:
            print("  틀림 꺾인 경로인데 어느 위상에서도 0 이다")
            ok = False

    # 그리퍼: 5스텝째 전환. K=2 면 블록 [4,6) 안이라 1스텝 밀린다.
    a2 = np.zeros((40, 12)); a2[:, 5] = 1.0; a2[5:, 11] = 1.0
    chk("그리퍼 래치 밀림 K=2", chunk_error_at(a2, 0, 2.0, 0.0)[2], 1.0)

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

    # **청크 크기를 데이터셋에서 읽는다.** 1000 으로 박아 뒀는데 이 데이터셋은
    # 300 이라 ep>=300 이 전부 없는 경로가 됐다. 첫 실행에서 384개 중 368개가
    # 조용히 빠지고 태스크 하나만 남았다. 동료가 잡았다.
    info = json.load(open(f"{root}/meta/info.json"))
    CH = int(info.get("chunks_size", 1000))
    print(f"청크 크기 {CH} (meta/info.json)")

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

    rows, failed = [], []
    for i, ep in enumerate(picked):
        try:
            act = np.stack(pd.read_parquet(
                f"{root}/data/chunk-{ep // CH:03d}/"
                f"episode_{ep:06d}.parquet")["action"].values).astype(np.float64)
        except Exception as exc:
            # **조용히 넘기지 않는다.** 경고 한 줄 없이 표본이 작아지면, 그
            # 작은 표본에서 나온 상관을 판정으로 읽게 된다. 첫 실행이 그랬다.
            failed.append((ep, str(exc)[:80]))
            continue
        for f in range(0, max(0, len(act) - 16), a.stride):
            for K in KS:
                r = chunk_error(act, f, K)
                if r:
                    rows.append((ep2t[ep], ep, f, K, *r))
        if (i + 1) % 50 == 0:
            print(f"  에피 {i+1}/{len(picked)} · 행 {len(rows):,}", flush=True)

    if failed:
        print(f"\n[!] 못 읽은 에피소드 {len(failed)}/{len(picked)}개")
        for ep, why in failed[:3]:
            print(f"    ep{ep} · {why}")

    df = pd.DataFrame(rows, columns=["task", "ep", "f", "K",
                                     "pos", "pos_max", "rot", "grip"])
    print(f"\n행 {len(df):,} · 에피 {df.ep.nunique()} · 태스크 {df.task.nunique()}\n")

    ceil = json.load(open(os.path.join(HERE, "..", "analysis",
                                       f"{a.bench}_task_ceilings.json")))
    hi = {t: (v[1] if isinstance(v, list) else v) for t, v in ceil.items()}

    for K in KS:
        d = df[df.K == K]
        print(f"===== K = {K} =====")
        # 1) 태스크 안에서 갈리는가
        for col in ("pos", "pos_max", "rot", "grip"):
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
        # **태스크가 적으면 상관을 찍지 않는다.** 첫 실행이 태스크 하나였는데
        # 타우가 +0.000 으로 나왔고, 그것을 "관계가 없다" 로 읽으면 멀쩡한
        # 방향을 잘못된 근거로 버린다. 못 재는 것과 재서 0 인 것은 다르다.
        if len(x) < 8:
            print(f"  태스크 {len(x)}개 -- 상관을 낼 표본이 아니다. "
                  f"안 찍는다(0 으로 보이면 판정으로 읽힌다)")
            continue
        tau = (c - dd) / max(1, c + dd)
        se = (2 * (2 * len(x) + 5) / (9 * len(x) * (len(x) - 1))) ** 0.5
        print(f"  태스크별 pos p90 대 실측 상한 · 켄달 타우 {tau:+.3f}  "
              f"({abs(tau)/se:.1f}SE, n={len(x)})  "
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
