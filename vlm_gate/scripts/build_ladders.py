"""문항·라벨 작업에 쓸 **배속 사다리**만 골라 한 파일로 만든다.

    python vlm_gate/scripts/build_ladders.py
    -> vlm_gate/analysis/eval_results/LADDERS.json
       vlm_gate/analysis/eval_results/LADDERS.md

## 왜 따로 만드나

`eval_results.json` 은 서버에 있는 실행을 다 담는다 -- 무효 실행, 가변 압축,
적응 게이트, 체크포인트가 다른 것까지. 그것을 그대로 주면 받는 쪽이 **무엇이
사다리이고 무엇이 아닌지** 를 다시 가려야 하고, 실제로 섞어 쓰다 결론이 뒤집힌
적이 있다.

이 파일에는 **균일 고정 배속 사다리만** 들어간다. 한 벤치마크의 한 사다리는
같은 정책·같은 조건·같은 블록 모양에서 배속만 다른 실행들이다. 그래야

  * 문항을 다시 뽑을 수 있다 (태스크를 손상으로 갈라야 하므로)
  * 상한표를 다시 만들 수 있다 (태스크마다 어디서 깨지는지 봐야 하므로)
  * 라벨 방식을 다듬을 수 있다 (띠와 규칙을 바꿔 보려면 태스크별 값이 필요)

그래서 **태스크별로 다 넣는다.** 종합 성공률만으로는 위 셋 중 아무것도 못 한다.
"""
import json
import os
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "analysis", "eval_results", "eval_results.json")
DST = os.path.join(HERE, "..", "analysis", "eval_results", "LADDERS")

# 사다리 = (배속, 실행 이름). 첫 줄이 기준선(1.0)이다.
LADDERS = OrderedDict([
 ("libero/flevel", {
   "what": "F_level 배속별 디코더. **libero 상한표의 정본이다.**",
   "how": "잡 157729 checkpoint-60000, level_ks 1·1.5·2·2.5 로 학습. level 을 "
          "하나씩 고정(--force-level). seed 7 · replan 5 · --args.no-action-clip · "
          "kp 150 · torque stock. 40태스크 × 50에피.",
   "blocks": "16스텝을 [1]*16 / [1,2] 교대 / [2]*8 / [2,3] 교대 로 묶는다. 고르다.",
   "rungs": [(1.0, "flevel157729_K1"), (1.5, "flevel157729_K1p5"),
             (2.0, "flevel157729_K2"), (2.5, "flevel157729_K2p5")],
   "compare": {"flevel157729_adaptive_rev":
               "readout 이 chunk 마다 level 을 고른 것. 사다리가 아니라 **대조군**이다 -- "
               "우리 confidence 라벨이 넘어야 할 선."}}),
 ("libero/naive", {
   "what": "압축 없이 학습된 베이스라인(b32)에 균일 압축. **상한표에 쓰지 않는다.**",
   "how": "compress_scope=replan -- 재예측 창을 K 로 쪼개고 꼬리를 낱개로 남긴다. "
          "클리핑만 풂(--args.no-action-clip). 40태스크 × 50에피.",
   "blocks": "창 5 를 [2,2,1](1.667) 또는 [3,1,1](1.667), 창 6 을 [2,2,2](2.0). "
             "**모양이 배속만큼 중요하다** -- 같은 1.667배가 0.938 대 0.890 이다.",
   "rungs": [(1.0, "baseline_bs32_allfine_K1"), (1.667, "naive_r1p667_even"),
             (2.0, "naive_r2p0")],
   "compare": {"naive_r1p667_front":
               "같은 1.667배인데 블록이 [3,1,1] 로 앞에 몰린 것. 모양의 효과를 재는 짝."}}),
 ("robocasa/uniform", {
   "what": "균일 고정 배속. **robocasa 상한표의 정본이다.**",
   "how": "압축 없이 학습된 GR00T 베이스라인. 24태스크 × 50에피. 네 칸을 **클립 푼 "
          "조건 하나**로 본다 -- robocasa 는 클립을 풀어도 성공률이 안 움직인다"
          "(압축을 건 채 K=3 +0.002 t +0.16, K=4 −0.022 t −1.52).",
   "blocks": "16스텝 청크. K=3 은 [3]*5 + 낱개 1 이라 **실제 2.67배**다(3.0 이 아니다).",
   "rungs": [(1.0, "baseline_full_v2_with_action_steps"),
             (1.5, "baseline_compress_K1p5_clip3"),
             (2.0, "baseline_compress_K2"),
             (2.5, "baseline_compress_K2p5_clip3"),
             (2.67, "baseline_compress_K3"),
             (4.0, "baseline_compress_K4")]}),
 ("dexjoco/uniform", {
   "what": "균일 고정 배속.",
   "how": "6태스크 × 50에피. 클립을 풀 것이 없다 -- 실행 경로에 ±1 컨트롤러 클립이 "
          "없고 손 관절한계 클램프 발화율이 0 이다.",
   "blocks": "절대 관절 목표라 블록의 **마지막**을 쓴다(더하면 각도를 두 배로 명령한다).",
   "rungs": [(1.0, "baseline_60k_K1_alin"), (1.5, "baseline_60k_K1p5"),
             (2.0, "baseline_60k_K2_alin"), (2.5, "baseline_60k_K2p5"),
             (3.0, "baseline_60k_K3")]}),
])

# 사다리에 **넣지 않은** 실행과 그 이유. 다음 사람이 "왜 없지" 하지 않도록 남긴다.
EXCLUDED = {
 "_INVALID_*": "무효 실행. 이름에 이유가 붙어 있다(off-by-one, replan 버그).",
 "libero baseline_K3 · baseline_K4": "b64 체크포인트가 섞여 b32 사다리와 못 견준다.",
 "libero baseline_K4_fixed": "28태스크뿐이라 40태스크 사다리와 나란히 못 놓는다.",
 "libero baseline_raw": "b64 체크포인트(0.9325). b32 기준선은 0.9565 다.",
 "robocasa interp_*": "가변 압축이라 배속이 청크마다 다르다. '이 태스크는 몇 배속까지 "
                      "견디나' 를 말해 주지 않는다.",
 "flevel adaptive_rev*": "적응 선택이라 사다리가 아니다. compare 에 대조군으로 넣었다.",
 "dexjoco baseline_60k_K1 · K2": "ALIN 재현본(_alin)이 같은 조건에서 더 최근이라 그쪽을 쓴다.",
}


def main():
    db = json.load(open(SRC))["runs"]
    out, md = {}, ["# 배속 사다리 — 문항·라벨 작업용\n",
                   "`build_ladders.py` 가 `eval_results.json` 에서 **균일 고정 배속만** "
                   "골라 낸 것이다. 잡다한 실행이 섞이면 무엇이 사다리인지 다시 가려야 "
                   "하고, 실제로 섞어 쓰다 결론이 뒤집힌 적이 있다.\n",
                   "**태스크별 값이 다 들어 있다.** 종합 성공률만으로는 문항을 다시 뽑는 것도, "
                   "상한표를 다시 만드는 것도, 라벨 방식을 다듬는 것도 못 한다.\n",
                   "기계는 `LADDERS.json`, 사람은 이 문서를 본다.\n"]
    for key, spec in LADDERS.items():
        bench = key.split("/")[0]
        runs = db.get(bench, {})
        rungs = []
        for ratio, name in spec["rungs"]:
            r = runs.get(name)
            if not r:
                print(f"[ ] {key}  {ratio} -> {name} 없음")
                continue
            rungs.append({"ratio": ratio, "run": name, "success": r["success"],
                          "n": r["n"], "n_tasks": r["n_tasks"],
                          "steps_on_success": r.get("steps_on_success"),
                          "tasks": r.get("tasks", {})})
        cmp_ = {}
        for name, why in (spec.get("compare") or {}).items():
            r = runs.get(name)
            if r:
                cmp_[name] = {"why": why, "success": r["success"],
                              "steps_on_success": r.get("steps_on_success"),
                              "tasks": r.get("tasks", {})}
        out[key] = {k: spec[k] for k in ("what", "how", "blocks")}
        out[key]["rungs"] = rungs
        if cmp_:
            out[key]["compare"] = cmp_
        print(f"[o] {key}  칸 {len(rungs)}  대조군 {len(cmp_)}")

        md.append(f"\n## {key}\n")
        md.append(spec["what"] + "\n")
        md.append(f"\n- **어떻게** {spec['how']}")
        md.append(f"- **블록** {spec['blocks']}\n")
        md.append("\n| 배속 | 실행 | 성공률 | 성공스텝 | 에피 | 태스크 |")
        md.append("|---:|---|---:|---:|---:|---:|")
        for g in rungs:
            st = f"{g['steps_on_success']:.1f}" if g["steps_on_success"] else "-"
            md.append(f"| {g['ratio']} | `{g['run']}` | {g['success']:.4f} | {st} "
                      f"| {g['n']} | {g['n_tasks']} |")
        for name, c in cmp_.items():
            st = f"{c['steps_on_success']:.1f}" if c["steps_on_success"] else "-"
            md.append(f"| — | `{name}` (대조군) | {c['success']:.4f} | {st} | | |")
        md.append("")
        # 태스크별 표
        tks = sorted({t for g in rungs for t in g["tasks"]})
        if tks:
            md.append("\n### 태스크별\n")
            md.append("| 태스크 | " + " | ".join(str(g["ratio"]) for g in rungs) + " |")
            md.append("|---|" + "---:|" * len(rungs))
            for t in tks:
                row = [f"{g['tasks'][t]['success']:.2f}" if t in g["tasks"] else "-"
                       for g in rungs]
                md.append(f"| {t} | " + " | ".join(row) + " |")
            md.append("")

    md.append("\n## 사다리에 넣지 않은 것\n")
    md.append("| 실행 | 왜 |")
    md.append("|---|---|")
    for k, v in EXCLUDED.items():
        md.append(f"| `{k}` | {v} |")
    md.append("")

    json.dump({"ladders": out, "excluded": EXCLUDED}, open(DST + ".json", "w"),
              ensure_ascii=False, indent=1, sort_keys=True)
    open(DST + ".md", "w").write("\n".join(md))
    print(f"-> {os.path.normpath(DST)}.json / .md")


if __name__ == "__main__":
    main()
