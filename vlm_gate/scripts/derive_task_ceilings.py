"""태스크마다 배속 상한을 측정에서 끌어낸다.

allex 는 운용자가 서브태스크별로 배속을 올려가며 성공 횟수를 세어 상한을 줬다.
시뮬은 eval 이 있으므로 같은 것을 계산으로 만든다.

상한은 두 조건이 다 성립하는 가장 높은 배속이다.

    1. 성공률이 허용 폭(TOL) 안에서 유지된다
    2. 스텝이 앞 배속보다 더 줄어든다

두 번째가 없으면 놓치는 것이 있다. 성공률은 지켰는데 시간을 더 쓰는 배속이
있을 수 있고, 압축의 목적이 시간을 아끼는 것이므로 시간을 못 아끼면 그 배속을
쓸 이유가 없다.

**스텝은 반드시 짝지어 본다.** `steps_on_success` 는 성공한 에피소드만의
평균이라 조건마다 부분집합이 다르다. 배속을 올려 성공률이 떨어지면 살아남은
에피소드가 다른(대개 더 쉬운) 집합이고, 그 두 평균을 빼면 배속 효과가 아니라
집합이 바뀐 효과가 나온다. 초판이 그걸 안 하고 세 태스크를 상한 2.0 으로
묶었는데, 짝지어 다시 재니 방향이 사라지거나 뒤집혔다.

    태스크              성공만 평균        짝지은 차이
    CloseSingleDoor     165.1 -> 179.6    +9.2  (n=44, t +0.56)   잡음
    TurnOnStove         285.0 -> 324.8   -21.2  (n=15, t -0.44)   부호가 뒤집힘
    TurnOffStove        156.1 -> 213.6    -2.8  (n= 4, t -0.51)   짝이 4개뿐
    CloseDrawer(대조)   116.6 ->  99.4   -17.3  (n=50, t -13.08)  일치

대조군이 규칙 자체는 멀쩡함을 보여 준다. CloseDrawer 는 양쪽 100% 성공이라
부분집합이 안 갈리고, 두 값이 일치한다. **부분집합이 안 갈릴 때만 성립하는
비교였다.**

짝이 적으면(MIN_PAIRS 미만) "판정 불가" 로 두고 성공률 규칙만 쓴다.
TurnOffStove 처럼 짝이 4개인 것은 "3x 가 나쁘다" 가 아니라 "이 데이터로는
못 정한다" 다.

**한계.** 지금 robocasa 는 2·3·4 배속만 있어 상한이 네 칸뿐이다. 상한 1.0 으로
떨어진 태스크는 "2x 도 안 된다" 까지만 아는 것이고 1.5 가 되는지는 모른다.
1.5·2.5 를 채우면 그 칸이 갈린다. 그리고 태스크당 50에피라 성공률의 표준오차가
약 0.07 이고, TOL 0.10 은 1.4 표준오차다.

    python derive_task_ceilings.py [벤치마크]
"""
import sys
sys.path.insert(0, "/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate")
from qgate import evalscan as E

BENCH = sys.argv[1] if len(sys.argv) > 1 else "robocasa"
rs = {r.name: r for r in E.list_runs(BENCH)}
BASE = {"robocasa": "baseline_full_v2_with_action_steps",
        "libero": "baseline_bs32_allfine_K1",
        "dexjoco": "baseline_60k_K1"}[BENCH]
RUNS = {"robocasa": [(2.0, "baseline_compress_K2"), (3.0, "baseline_compress_K3"),
                     (4.0, "baseline_compress_K4")],
        "libero": [(3.0, "baseline_K3"), (4.0, "baseline_K4")],
        "dexjoco": [(2.0, "baseline_60k_K2"), (3.0, "baseline_60k_K3")]}[BENCH]
TOL = 0.10          # 성공률 허용 폭. 50에피 기준 표준오차 약 0.07
MIN_PAIRS = 10      # 짝이 이보다 적으면 스텝 규칙은 판정 불가


def ep_steps(run):
    """(태스크, 에피소드) -> 스텝. 성공한 것만."""
    out = {}
    for t, task in run.tasks.items():
        for e in task.episodes:
            idx, ok = e[0], e[1]
            st = e[2] if len(e) > 2 else None
            if ok and st is not None:
                out[(t, idx)] = st
    return out


STEPS = {nm: ep_steps(rs[nm]) for _, nm in RUNS if nm in rs}
if BASE in rs:
    STEPS[BASE] = ep_steps(rs[BASE])


def paired(task, a, b):
    """두 실행에서 다 성공한 에피소드만. (짝수, 평균차, t)."""
    A, B = STEPS.get(a, {}), STEPS.get(b, {})
    both = [k for k in A if k in B and k[0] == task]
    if not both:
        return 0, None, None
    d = [B[k] - A[k] for k in both]
    n = len(d)
    m = sum(d) / n
    if n < 2:
        return n, m, None
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    se = (var / n) ** 0.5
    return n, m, (m / se if se > 0 else None)


b = rs[BASE]
print(f"기준선 {BASE}  성공 {b.success:.3f}  태스크 {len(b.tasks)}")
print(f"규칙: 성공률이 {TOL} 안에서 유지되고, 짝지은 스텝이 줄어드는 가장 높은 배속")
print(f"      짝이 {MIN_PAIRS} 미만이면 스텝은 판정 불가 (성공률만 본다)\n")
print(f"{'태스크':<24}{'상한':>5}  근거")
rows = []
for t in sorted(b.tasks):
    s0 = b.tasks[t].success
    cap, why = 1.0, f"{RUNS[0][0]:g}x 에서 이미 손상"
    prev = BASE
    for r, nm in RUNS:
        tk = rs[nm].tasks.get(t) if nm in rs else None
        if tk is None:
            why = f"{r:g}x 실행이 없다"
            break
        if tk.success - s0 < -TOL:
            why = f"{r:g}x 에서 성공률 {tk.success - s0:+.2f}"
            break
        n, m, tt = paired(t, prev, nm)
        if n < MIN_PAIRS:
            why = (f"{r:g}x 까지 성공률 유지 (스텝은 짝 {n}개라 판정 불가)")
            cap, prev = r, nm
            continue
        if m is None or m >= 0:
            why = (f"{r:g}x 에서 스텝이 더 안 줄어듦 "
                   f"({m:+.1f}, n={n}" + (f", t {tt:+.2f})" if tt is not None else ")"))
            break
        cap, prev = r, nm
        why = (f"{r:g}x 까지 유지 (스텝 {m:+.1f}, n={n}"
               + (f", t {tt:+.2f})" if tt is not None else ")"))
    rows.append((t, cap))
    print(f"{t[:24]:<24}{cap:5.1f}  {why}")

from collections import Counter
print("\n상한 분포:", dict(sorted(Counter(c for _, c in rows).items())))
import json
print(json.dumps({t: c for t, c in rows}, ensure_ascii=False, indent=1))
