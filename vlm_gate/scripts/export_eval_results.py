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

# 결과가 이 저장소의 output/ 안에만 있는 것이 아니다. 동료가 낸 것도 있고,
# 그것을 못 찾아 같은 실행을 다시 뜨는 것이 실제로 있었다. 밖에 있는 정본은
# 여기에 적어 두고 같이 읽는다.
EXTERNAL = {
    "libero": {
        "baseline_bs32_allfine_K1": {
            "path": "/sjw_alinlab2/home/taekwan/Data/libero_all_Fine",
            "note": "taekwan 이 2026-08-05 에 정리한 40태스크 K=1 정본. "
                    "ckpt prehj/GR00T-N1.5-libero-baseline-bs32-60k, seed 7, "
                    "gate_mode=none, 태스크당 50에피. clip1/dyn1(당시 플래그 없음). "
                    "output/libero/baseline_raw(0.932)는 b64 라 b32 사다리의 "
                    "기준선으로 쓸 수 없고 이쪽을 쓴다. 실측 0.9565.",
        },
        "taekwan_r1p67_clipon": {
            "path": "/sjw_alinlab2/home/taekwan/Data/libnaive_clipon_s7_K2_allk2",
            "note": "이름은 K2 지만 **실제 배속은 1.67 배다.** libero 는 replan 이 "
                    "5스텝마다 걸려서 정수 블록으로는 5를 (2,2,1) 로밖에 못 쪼갠다 "
                    "-- 5스텝이 3스텝이 되므로 5/3=1.67 이다. 2.0 이 아니다. "
                    "b32 체크포인트·seed 7·40태스크 50에피, 클리핑 켠 채. "
                    "실측 0.9060, 기준선 0.9565 대비 -5.05%p. "
                    "**libero 문항을 뽑을 때 깔았던 'K2 는 공짜(93.0 vs 92.9)' 는 "
                    "b64 숫자였고, b32 에서는 1.67 배만으로도 5%p 가 깎인다.**",
        },
        "taekwan_r1p67_clipoff": {
            "path": "/sjw_alinlab2/home/taekwan/Data/libnaive_clipoff_s7_K2_allk2",
            "note": "위와 같은 1.67 배에 명령 클리핑만 푼 것. 실측 0.9460, "
                    "-1.05%p. 클리핑을 푸는 것만으로 손상의 4%p 가 돌아온다. "
                    "구동부 한계(dyn)는 켠 채라 CLIP_SCALE=3,DYN_SCALE=3 으로 "
                    "돌린 released_* 와 같은 조건이 아니다. "
                    "진짜 2.0 배는 fractional_blocks 로 청크 사이에 잔여를 넘겨야 "
                    "나오고, 그게 released_K2(job 162314)다.",
        },
    }
}


def read_external(root):
    """suite/<idx>_results.txt 의 에피소드 줄에서 직접 센다. 요약줄을 안 믿는다."""
    from pathlib import Path
    tasks, tot, ok = {}, 0, 0
    for f in sorted(Path(root).glob("*/*_results.txt")):
        name = f"{f.parent.name}_{f.stem.split('_')[0]}"
        n = s = 0
        st = []
        for line in open(f):
            q = line.split("\t")
            if len(q) >= 2 and q[1].strip() in ("True", "False"):
                n += 1
                if q[1].strip() == "True":
                    s += 1
                    if len(q) >= 3:
                        try:
                            st.append(float(q[2]))
                        except ValueError:
                            pass
        if n:
            tasks[name] = {"success": round(s / n, 4), "n": n, "n_success": s,
                           "steps_on_success": (round(sum(st) / len(st), 1) if st else None)}
            tot += n
            ok += s
    if not tot:
        return None
    allst = [t["steps_on_success"] for t in tasks.values() if t["steps_on_success"]]
    return {"success": round(ok / tot, 4), "n": tot, "n_tasks": len(tasks),
            "steps_on_success": (round(sum(allst) / len(allst), 1) if allst else None),
            "tasks": tasks}


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
    for nm, meta_ in EXTERNAL.get(bench, {}).items():
        try:
            r = read_external(meta_["path"])
        except Exception as e:
            print(f"  바깥 결과 {nm} 못 읽음: {e}")
            r = None
        if r:
            r["external_path"] = meta_["path"]
            r["note"] = meta_["note"]
            runs[nm] = r
            print(f"  바깥 결과 {nm}: {r['success']:.4f} · {r['n_tasks']}태스크")
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
