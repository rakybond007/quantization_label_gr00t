"""바깥 클러스터에서 돈 실행을 `eval_results.json` 에 **태스크별로** 넣는다.

종합 성공률만 적으면 받는 쪽이 상한표를 다시 못 만든다. 상한은 태스크마다
따로이므로 태스크별 성공률·에피소드 수·성공 스텝이 다 있어야 한다.

    python vlm_gate/scripts/add_external_runs.py

`EXTRA` 에 적힌 것을 읽어 `runs.<bench>.<name>` 에 넣는다. 이미 있으면 덮는다.
에피소드 줄에서 직접 세고 헤더는 안 믿는다(`qgate/audit.py` 머리말 참고).
"""
import json
import os
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, "..", "analysis", "eval_results", "eval_results.json")

_T = Path("/sjw_alinlab2/home/taekwan/Data")
_R = Path("/rlwrld-unified-checkpoints/taekwan")
_F = _T / "flevel_libero_157729_ckpt60000"

EXTRA = {
 "libero": {
  # --- F_level 사다리. **libero 상한표의 정본이다.**
  "flevel157729_K1": (_F / "K1",
    "잡 157729 checkpoint-60000, level_ks 1·1.5·2·2.5 로 학습된 체크포인트에서 "
    "level 1 을 모든 chunk 에 고정(--force-level 1). seed 7 · replan 5 · "
    "--args.no-action-clip · kp 150 · torque stock · level_hold 1.0."),
  "flevel157729_K1p5": (_F / "K1p5", "같은 체크포인트, level 1.5 고정(force2). 블록 [1,2] 교대."),
  "flevel157729_K2": (_F / "K2", "같은 체크포인트, level 2 고정(force3)."),
  "flevel157729_K2p5": (_F / "K2p5", "같은 체크포인트, level 2.5 고정(force4). 블록 [2,3] 교대."),
  "flevel157729_adaptive_rev": (_F / "adaptive_rev",
    "같은 체크포인트, readout 이 chunk 마다 level 을 고른다(level-map reverse). "
    "**우리 confidence 라벨의 대조군이다** -- 넘어야 할 선."),
  "flevel157729_adaptive_rev_gs1p5": (_F / "adaptive_rev_gs1p5",
    "위와 같으나 gripper speed 1.5배."),
  # --- naive 사다리. 상한표에 **쓰지 않는다** (모양이 어긋난다).
  "naive_r1p667_even": (_R / "eqs_libero_naive_k2_noclip_repeat",
    "이름은 K2 지만 compress_scope=replan 이라 창 5 를 [2,2,1] 로 쪼갠다 -> "
    "실제 배속 1.667. 압축 없이 학습된 베이스라인(b32)에 균일 압축. 클리핑만 풂."),
  "naive_r1p667_front": (_R / "eqs_libero_naive_k3_noclip",
    "이름은 K3. 창 5 를 [3,1,1] 로 -> 실제 1.667. **위와 배속이 같은데 모양만 다르다** -- "
    "0.938 대 0.890 으로 갈리는 것이 '모양이 배속만큼 중요하다' 의 근거다."),
  "naive_r2p0": (_R / "eqs_libero_naive_k2_replan6", "창 6 을 [2,2,2] 로 -> 실제 2.000."),
  # **원본이 사라졌다.** 2026-09-08 오전에 읽어 0.597 을 뽑았는데 오후에 보니
  # 디렉터리가 없다(`eqs_libero_noclip_k4_sweep` 에는 결과 파일이 0개다).
  # 상한표에 쓰지 않기로 한 것이라 다시 뜨지 않는다 -- 창 5 를 [4,1] 로 쪼갠
  # 것이라 한 명령이 4스텝을 몰아 먹고, 0.597 은 2.5배의 손상이 아니라 그 모양의
  # 손상이었다. 값이 필요하면 F_level K2p5(0.785)를 쓴다.
  # "naive_r2p5_lumpy": (_R / "eqs_libero_naive_k4_noclip", ...),
 },
}


def load(root):
    """태스크 -> (성공, 에피소드, 성공스텝). 에피소드 줄에서 직접 센다."""
    o = {}
    for pat in ("*/*_results.txt", "*/*/*_results.txt"):
        for f in Path(root).glob(pat):
            k = f.parent.name.replace("vid_", "") + "/" + f.stem.split("_")[0]
            if k in o:
                continue
            n = s = 0
            st = []
            for line in open(f, errors="replace"):
                q = line.strip().split("\t")
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
                o[k] = {"success": round(s / n, 4), "n": n, "n_success": s,
                        "steps_on_success": (round(sum(st) / len(st), 1) if st else None)}
    return o


def main():
    db = json.load(open(JSON))
    for bench, runs in EXTRA.items():
        for name, (path, note) in runs.items():
            if not Path(path).exists():
                print(f"[ ] {bench}/{name} 없음 -- {path}")
                continue
            tasks = load(path)
            if not tasks:
                print(f"[ ] {bench}/{name} 결과가 비었다 -- {path}")
                continue
            tot = sum(v["n"] for v in tasks.values())
            ok = sum(v["n_success"] for v in tasks.values())
            sts = [v["steps_on_success"] for v in tasks.values() if v["steps_on_success"]]
            db["runs"].setdefault(bench, {})[name] = {
                "success": round(ok / tot, 4), "n": tot, "n_tasks": len(tasks),
                "steps_on_success": (round(sum(sts) / len(sts), 1) if sts else None),
                "external_path": str(path), "note": note, "tasks": tasks}
            print(f"[o] {bench}/{name}  {ok/tot:.4f}  태스크 {len(tasks)}  에피 {tot}")
    json.dump(db, open(JSON, "w"), ensure_ascii=False, indent=1, sort_keys=True)
    print(f"-> {os.path.normpath(JSON)}")


if __name__ == "__main__":
    main()
