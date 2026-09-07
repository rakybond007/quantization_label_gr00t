"""세 벤치마크의 압축 배속별 성공률을 태스크 단위까지 파일로 뽑는다.

결과가 서버의 `output/<bench>/<run>/server-*.log` 안에만 있어서, 다른 서버나
다른 세션에서 참고하려면 그 로그를 다시 훑어야 한다. 이 스크립트가 한 번
훑어서 저장소에 올릴 수 있는 파일로 만든다.

    eval_results.json   기계가 읽는 것. 실행 → 태스크 → (성공률, n, 성공수)
    eval_results.md     사람이 읽는 것. 배속별 표와 태스크별 손상

손상은 압축 없는 기준선 대비 차이다. 기준선은 벤치마크마다 이름이 다르다:
robocasa 는 baseline_full_v2_with_action_steps, libero 는 baseline_raw,
dexjoco 는 baseline_60k_K1.

**무효 실행은 이름에 _INVALID_ 를 달고 있고 여기서도 그대로 실어 나른다.**
지우면 왜 그 배속이 비어 있는지 다음 사람이 모른다.

    python export_eval_results.py [출력 디렉터리]
"""
import json
import os
import sys

sys.path.insert(0, os.path.expanduser(
    "~/quantization_agent_workspace/vlm_gate"))
from qgate import evalscan as E  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "."
BASE = {"robocasa": "baseline_full_v2_with_action_steps",
        "libero": "baseline_raw",
        "dexjoco": "baseline_60k_K1"}
# 균일 고정 배속 실행만 손상표에 쓴다. 게이트가 붙은 실행이나 적응형 압축은
# 배속이 청크마다 달라서 "이 태스크는 몇 배속까지 견디나" 를 말해 주지 않는다.
UNIFORM = ("baseline_compress_K", "baseline_K", "baseline_60k_K")

db, notes = {}, {}
for bench in ("robocasa", "libero", "dexjoco"):
    runs = {}
    for r in E.list_runs(bench):
        try:
            # 성공 스텝 수도 같이 뽑는다. 배속을 올렸는데 스텝이 그만큼 안 주는
            # 태스크가 있다 -- 과한 명령이 유발한 재시도로 사라지기 때문이다.
            # 성공률만 보면 그게 안 보인다.
            tasks = {}
            for t, v in r.tasks.items():
                st = v.steps_on_success
                tasks[t] = {"success": round(v.success, 4), "n": v.n,
                            "n_success": v.n_success,
                            "steps_on_success": (round(st, 1) if st else None)}
        except Exception:
            continue
        if not tasks:
            continue
        rs_ = r.steps_on_success
        runs[r.name] = {"success": round(r.success, 4), "n": r.n,
                        "n_tasks": len(tasks),
                        "steps_on_success": (round(rs_, 1) if rs_ else None),
                        "tasks": tasks}
    db[bench] = runs
    notes[bench] = {"baseline": BASE.get(bench), "n_runs": len(runs)}
    print(f"{bench}: 실행 {len(runs)}개")

json.dump({"runs": db, "meta": notes}, open(f"{OUT}/eval_results.json", "w"),
          ensure_ascii=False, indent=1, sort_keys=True)


def damage(bench, run):
    """기준선 대비 태스크별 차이. 기준선이나 실행이 없으면 None."""
    b = db[bench].get(BASE.get(bench))
    k = db[bench].get(run)
    if not b or not k:
        return None
    out = {}
    for t in set(b["tasks"]) & set(k["tasks"]):
        out[t] = round(k["tasks"][t]["success"] - b["tasks"][t]["success"], 4)
    return out


L = []
L.append("# 압축 배속별 평가 결과\n")
L.append("`export_eval_results.py` 가 서버의 eval 로그에서 뽑았다. "
         "다른 서버·세션에서 이 파일만 보고 손상표를 만들 수 있게 하려는 것이다.\n")
L.append("손상은 압축 없는 기준선 대비 성공률 차이다. 음수가 압축이 깎은 만큼이다.\n")

for bench in ("robocasa", "libero", "dexjoco"):
    base = BASE.get(bench)
    L.append(f"\n## {bench}\n")
    b = db[bench].get(base)
    if b:
        L.append(f"기준선 `{base}` — 성공률 **{b['success']:.3f}**, "
                 f"태스크 {b['n_tasks']}개, 에피소드 {b['n']}개\n")
    L.append("\n### 균일 고정 배속\n")
    L.append("| 실행 | 성공률 | 에피소드 | 태스크 |")
    L.append("|---|---:|---:|---:|")
    uni = sorted(n for n in db[bench]
                 if any(n.startswith(p) for p in UNIFORM) or n == base
                 or "_INVALID_" in n)
    for n in uni:
        r = db[bench][n]
        L.append(f"| `{n}` | {r['success']:.3f} | {r['n']} | {r['n_tasks']} |")
    L.append("")

    # 손상표: 기준선 대비 K2 (없으면 K3)
    for cand in (f"baseline_compress_K2", "baseline_K2", "baseline_60k_K2",
                 "baseline_compress_K3", "baseline_K3", "baseline_60k_K3"):
        d = damage(bench, cand)
        if d:
            L.append(f"\n### 태스크별 손상 — `{base}` → `{cand}`\n")
            L.append("| 태스크 | 기준선 | 압축 | 차이 |")
            L.append("|---|---:|---:|---:|")
            for t, v in sorted(d.items(), key=lambda kv: kv[1]):
                bs = db[bench][base]["tasks"][t]["success"]
                ks = db[bench][cand]["tasks"][t]["success"]
                L.append(f"| {t} | {bs:.2f} | {ks:.2f} | {v:+.3f} |")
            L.append("")
            break

open(f"{OUT}/eval_results.md", "w").write("\n".join(L))
print(f"-> {OUT}/eval_results.json, {OUT}/eval_results.md")
