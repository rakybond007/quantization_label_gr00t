"""libero v2 문항 점검. `PROMPT_METHOD.md` 의 판정 중 지금 볼 수 있는 것만 본다.

    [1] 형식        5칸 완답률 ≥99%
    [3] 부호        감점 문항이 위험 풀에서 높고 가점 문항이 안정 풀에서 높은가
    [등급 사용]     2·4 등급이 죽어 있으면 등급표 결함이다
    [문항 상관]     서로 같은 답만 하면 한 문항이 남는 것과 같다 (참고용)

**[3] 이 이 점검의 본체다.** 나머지는 진단이다 -- 값이 나쁘다고 문항을 고치지
않고, 왜 그런지를 먼저 본다 (`memory/gates-need-evidence.md`).

    python vlm_gate/scripts/libero_v2_verify.py [labels.jsonl]
"""
import json
import os
import sys
from collections import Counter

import numpy as np

SLOTS = "ABCDE"
# 부호·가중치는 한 군데에만 둔다. 두 벌을 두면 한쪽만 고쳐지고, 그러면 검증이
# 통과한 값과 라벨이 쓴 값이 달라진다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from libero_v2_checks import NAME, SIGN, WEIGHT  # noqa: E402

# 도출에서 나온 두 풀 (derive_libero_questions.py 와 같은 값)
RISK = {"long/1", "spatial/3", "spatial/0", "goal/1", "spatial/5", "spatial/8",
        "goal/3", "goal/9", "long/8"}
STABLE = {"object/2", "object/4", "goal/8", "goal/7", "long/4", "long/7",
          "goal/2", "long/5", "goal/4", "object/3", "object/5", "object/9",
          "object/0", "object/7", "goal/5", "long/3"}

BASE = os.path.expanduser("~/quantization_agent_workspace/vlm_gate")
SRC = sys.argv[1] if len(sys.argv) > 1 else \
    f"{BASE}/output/_gate_distill/libero_v2_smoke_s1_0.jsonl"


DS = "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"


def _ep_to_task():
    """에피소드 -> 'suite/idx'. **번호로 추측하지 않고 지시문으로 맞춘다.**
    데이터셋의 에피소드 순서를 가정하면 조용히 틀리고, 그러면 부호 판정이
    통째로 뒤집힌다."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from derive_libero_questions import INSTR
    by_text = {v.strip().lower(): k for k, v in INSTR.items()}
    out, miss = {}, set()
    for l in open(f"{DS}/meta/episodes.jsonl"):
        d = json.loads(l)
        for t in d.get("tasks", []):
            if isinstance(t, str) and t.strip().lower() in by_text:
                out[d["episode_index"]] = by_text[t.strip().lower()]
                break
        else:
            miss.add(d["episode_index"])
    if miss:
        print(f"[!] 지시문을 못 맞춘 에피소드 {len(miss)}개 -- 그 행은 부호 판정에서 뺀다")
    return out


EP2T = None


def task_of(ep):
    return EP2T.get(ep)


def main():
    global EP2T
    EP2T = _ep_to_task()
    rows = [json.loads(l) for l in open(SRC) if l.strip()]
    print(f"라벨 {len(rows)}행  <- {SRC}\n")
    if not rows:
        return

    # [1] 형식 -- 라벨러가 5칸을 못 받으면 아예 안 쓰므로, 쓰인 행의 완답률과
    # 답변 원문이 형식대로인지를 같이 본다.
    ok = sum(1 for r in rows if all(k in r for k in SLOTS))
    bad = [r["ans"] for r in rows
           if len([l for l in str(r.get("ans", "")).splitlines() if ")" in l]) != 5]
    print(f"[1] 형식      5칸 완답 {ok}/{len(rows)} = {ok / len(rows):.1%}"
          f"   답변 줄 수 어긋남 {len(bad)}")

    # [3] 부호
    print("\n[3] 부호      문항별 평균 등급")
    print(f"     {'문항':<16} {'위험풀':>7} {'안정풀':>7} {'차이':>7}  기대")
    allgood = True
    for k in SLOTS:
        rr = [r[k] for r in rows if task_of(r["ep"]) in RISK and k in r]
        ss = [r[k] for r in rows if task_of(r["ep"]) in STABLE and k in r]
        if not rr or not ss:
            print(f"     {k} {NAME[k]:<14} 표본 없음 (위험 {len(rr)} 안정 {len(ss)})")
            continue
        d = float(np.mean(rr) - np.mean(ss))
        want = "위험>안정" if SIGN[k] < 0 else "안정>위험"
        good = (d > 0) if SIGN[k] < 0 else (d < 0)
        allgood &= good
        print(f"     {k} {NAME[k]:<14} {np.mean(rr):7.2f} {np.mean(ss):7.2f} "
              f"{d:+7.2f}  {want} {'OK' if good else '어긋남'}")
    print(f"     -> {'통과' if allgood else '부호 어긋난 문항 있음'}")

    # 등급 사용
    print("\n[진단] 등급 분포")
    for k in SLOTS:
        c = Counter(int(r[k]) for r in rows if k in r)
        n = sum(c.values()) or 1
        bars = " ".join(f"{g}:{c.get(g, 0) / n:4.0%}" for g in (1, 2, 3, 4, 5))
        dead = [g for g in (2, 4) if c.get(g, 0) / n < 0.02]
        print(f"     {k} {NAME[k]:<14} {bars}"
              + (f"   <- {dead} 등급이 죽어 있다" if dead else ""))

    # 문항 상관
    print("\n[진단] 문항 상관 (같은 답만 하면 한 문항과 같다)")
    M = np.array([[r[k] for k in SLOTS] for r in rows if all(k in r for k in SLOTS)],
                 dtype=float)
    if len(M) > 5:
        C = np.corrcoef(M.T)
        for i in range(5):
            for j in range(i + 1, 5):
                if abs(C[i, j]) > 0.6:
                    print(f"     {SLOTS[i]}-{SLOTS[j]}  {C[i, j]:+.2f}")
        print(f"     최대 |상관| {np.abs(C - np.eye(5)).max():.2f}")

    # 신뢰도 분포
    conf = []
    for r in rows:
        if not all(k in r for k in SLOTS):
            continue
        risk = sum(WEIGHT[k] * (r[k] - 1) / 4 for k in SLOTS if SIGN[k] < 0)
        safe = sum(WEIGHT[k] * (r[k] - 1) / 4 for k in SLOTS if SIGN[k] > 0)
        conf.append(min(1.0, max(0.0, (1.0 + safe - risk) / 2)))
    if conf:
        q = np.percentile(conf, [0, 25, 50, 75, 100])
        print(f"\n[진단] 신뢰도  min {q[0]:.3f}  p25 {q[1]:.3f}  중앙 {q[2]:.3f}  "
              f"p75 {q[3]:.3f}  max {q[4]:.3f}   서로 다른 값 {len(set(np.round(conf, 4)))}개")


if __name__ == "__main__":
    main()
