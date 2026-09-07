"""태스크마다 배속 상한을 측정에서 끌어낸다.

allex 는 운용자가 서브태스크별로 배속을 올려가며 성공 횟수를 세어 상한을 줬다.
시뮬은 eval 이 있으므로 같은 것을 계산으로 만든다.

상한은 두 조건이 다 성립하는 가장 높은 배속이다.

    1. 성공률이 허용 폭(TOL) 안에서 유지된다
    2. 스텝이 앞 배속보다 더 줄어든다

두 번째가 없으면 놓치는 것이 있다. 성공률은 지켰는데 시간을 더 쓰는 배속이
실제로 있다 -- robocasa 에서 CloseSingleDoor(2x 165 -> 3x 180), TurnOffStove
(156 -> 214), TurnOnStove(285 -> 325) 셋이 그렇다. 성공률만 봤으면 3x 를
허용했을 자리다. 압축의 목적이 시간을 아끼는 것이므로 시간을 못 아끼면
그 배속을 쓸 이유가 없다.

**한계.** 지금 robocasa 는 2·3·4 배속만 있어 상한이 네 칸뿐이다. 상한 1.0 으로
떨어진 태스크는 "2x 도 안 된다" 까지만 아는 것이고 1.5 가 되는지는 모른다.
1.5·2.5 를 채우면(fractional_blocks) 그 칸이 갈린다. 그리고 태스크당 50에피라
성공률의 표준오차가 약 0.07 이고, TOL 0.10 은 1.4 표준오차다.

    python derive_task_ceilings.py [벤치마크]
"""
import sys
sys.path.insert(0, "/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate")
from qgate import evalscan as E

BENCH = sys.argv[1] if len(sys.argv) > 1 else "robocasa"
rs = {r.name: r for r in E.list_runs(BENCH)}
BASE = "baseline_full_v2_with_action_steps"
RUNS = [(2.0, "baseline_compress_K2"), (3.0, "baseline_compress_K3"),
        (4.0, "baseline_compress_K4")]
b = rs[BASE]
TOL = 0.10          # 성공률을 이만큼까지는 깎여도 봐준다 (50에피 기준 SE 약 0.07)

print(f"{'태스크':<24}{'기준':>6}{'2x':>7}{'3x':>7}{'4x':>7}   "
      f"{'스텝1':>6}{'2x':>6}{'3x':>6}{'4x':>6}   상한  근거")
rows = []
for t in sorted(b.tasks):
    s0 = b.tasks[t].success
    p0 = b.tasks[t].steps_on_success
    succ, step = {}, {}
    for r, nm in RUNS:
        tk = rs[nm].tasks.get(t)
        succ[r] = tk.success if tk else None
        step[r] = tk.steps_on_success if tk else None
    cap, why = 1.0, "2x 에서 이미 손상"
    prev_step = p0
    for r, _ in RUNS:
        if succ[r] is None or step[r] is None:
            break
        if succ[r] - s0 < -TOL:
            why = f"{r:g}x 에서 성공률 {succ[r]-s0:+.2f}"
            break
        if step[r] >= prev_step:
            why = f"{r:g}x 에서 스텝이 더 안 줄어듦"
            break
        cap, why, prev_step = r, f"{r:g}x 까지 성공률·스텝 모두 유지", step[r]
    rows.append((t, cap))
    f = lambda d, k, w=6, p=2: (f"{d[k]:{w}.{p}f}" if d[k] is not None else " " * w)
    print(f"{t[:24]:<24}{s0:6.2f}{f(succ,2.0,7)}{f(succ,3.0,7)}{f(succ,4.0,7)}   "
          f"{p0:6.0f}{f(step,2.0,6,0)}{f(step,3.0,6,0)}{f(step,4.0,6,0)}   "
          f"{cap:4.1f}  {why}")

from collections import Counter
print("\n상한 분포:", dict(sorted(Counter(c for _, c in rows).items())))
