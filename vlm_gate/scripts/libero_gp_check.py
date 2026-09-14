"""LIBERO 장면 집합을 문항 판으로 라벨한다. 등급 분포를 반드시 남긴다.

robocasa 의 `gp_conf_check.py` 와 같은 자리. 다른 것 셋.
  * 뷰가 둘이다(front + left_wrist). 타일을 2등분한다.
  * 액션이 7차원 -- `libero_descriptors` 가 그 규격으로 계산한다.
  * 그리퍼 사실 첫 줄을 `libero_phase.gripper_fact` 로 덮어쓴다. 기존
    `libero_descriptors.facts_text` 는 개폐를 방향 없이 말하는데, robocasa 에서
    방향을 주자 파지-해제 구분이 -0.003 -> -0.121 (p=0.0009) 로 갈렸다.

    CHECKS=libero_v2_checks PROMPT_VER=v2 \
      python libero_gp_check.py <port> <scenes.json> <out.jsonl>

이어서 돌 수 있다 -- 출력에 이미 있는 (ep,f) 는 건너뛰고 append 한다.
"""
import importlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.path.dirname(HERE)
DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "libero_gr00t_delta")
TILES = os.environ.get("TILES", f"{BASE}/output/_gate_distill/libero_full/tiles")
PV = os.environ.get("PROMPT_VER", "v2")
CK = os.environ.get("CHECKS", "libero_v2_checks")
N = int(os.environ.get("GP_N", 100000))
NVIEW = 2


def main():
    import pandas as pd
    from PIL import Image
    from vlm_gate import VLMGate
    from libero_descriptors import descriptors, facts_text
    from libero_phase import gripper_fact
    C = importlib.import_module(CK)
    CH = int(json.load(open(f"{DS}/meta/info.json")).get("chunks_size") or 1000)
    port, scenes_f, out_f = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    ASK = open(f"{BASE}/analysis/_evolver/_libero/"
               f"libero_questions_{PV}.txt").read().strip()
    G = open(f"{BASE}/analysis/_evolver/_libero/"
             f"libero_guidance_{PV}.txt").read().strip()

    scenes = json.load(open(scenes_f))[:N]
    done = set()
    if os.path.exists(out_f):
        for line in open(out_f):
            try:
                r = json.loads(line)
                done.add((r["ep"], r["f"]))
            except Exception:
                pass
    scenes = [s for s in scenes if (s["ep"], s["f"]) not in done]
    print(f"장면 {len(scenes)} (건너뜀 {len(done)}) · CHECKS={CK} · 문항 {PV}",
          flush=True)

    # **문항 수는 판이 정한다.** 5 로 박아 두면 문항이 셋인 판에서 picks 가 전부
    # None 이 되어 모든 장면이 버려지고 0행으로 끝난다(배속 문항 sa1 이 그랬다).
    NQ = len(getattr(C, "NAME", None) or getattr(C, "SIGN", {})) or 5
    QCOLS = sorted(getattr(C, "NAME", None) or getattr(C, "SIGN", {})) or list("ABCDE")
    # 지시문은 태스크마다 상수라, 넣으면 모델이 그것만 읽고 그림을 안 본다
    # (실측: 태스크 안/간 답 변동 비가 0.08 로 죽었다). 배속 문항에는 안 넣는다.
    USE_INSTR = os.environ.get("USE_INSTR", "1") == "1"
    gate = VLMGate(f"http://127.0.0.1:{port}", timeout=900)
    acts, n, ngp, bad = {}, 0, 0, 0
    with open(out_f, "a") as fh:
        for s in scenes:
            ep = s["ep"]
            if ep not in acts:
                if len(acts) > 24:
                    acts.clear()
                acts[ep] = np.stack(pd.read_parquet(
                    f"{DS}/data/chunk-{ep // CH:03d}/"
                    f"episode_{ep:06d}.parquet")["action"].values)
            a = acts[ep]
            x = descriptors(a, s["f"])
            # 그리퍼 줄만 방향·시점 있는 것으로 갈아끼운다. 나머지 문장은 그대로.
            ft = facts_text(x)
            ft = re.sub(r"(?<=estimates\): )[^;]+", gripper_fact(a, s["f"]), ft, count=1)
            im = np.array(Image.open(f"{TILES}/{s['tile']}.png").convert("RGB"))
            w = im.shape[1] // NVIEW
            views = [im[:, k * w:(k + 1) * w] for k in range(NVIEW)]
            _ctx = (f"{s.get('instruction','')}\n{ft}") if USE_INSTR else ft
            r = gate.judge(views, _ctx, G,
                           question=ASK, n_ask=NQ, n_grade=C.NGRADE, mode="text")
            picks = r.get("picks") or [None] * NQ
            if any(p is None for p in picks):
                bad += 1
                continue
            gp = r.get("grade_probs")
            eg = ([sum((i + 1) * float(q) for i, q in enumerate(row)) for row in gp]
                  if gp and len(gp) == NQ else None)
            ngp += int(bool(eg))
            g = {q: ((eg[i] if eg else float(picks[i])) - 1.0) / (C.NGRADE - 1)
                 for i, q in enumerate(QCOLS)}
            conf = min(1.0, max(0.0, (1.0
                                      + sum(C.WEIGHT[q] * g[q] for q in QCOLS if C.SIGN.get(q,0) > 0)
                                      - sum(C.WEIGHT[q] * g[q] for q in QCOLS if C.SIGN.get(q,0) < 0)) / 2))
            n += 1
            fh.write(json.dumps({**{k: s[k] for k in ("cell", "ep", "f", "tile", "phase")},
                                 "picks": picks, "gp": gp, "eg": eg,
                                 "conf_exp": conf}, ensure_ascii=False) + "\n")
            if n % 40 == 0:
                fh.flush()
                print(f"  {n}/{len(scenes)} gp있음 {ngp} 형식실패 {bad}", flush=True)
    print(f"[gp] {n} 장면 · gp {ngp} ({ngp/max(n,1):.1%}) · 형식실패 {bad}"
          f" · CHECKS={CK} 문항={PV} -> {out_f}", flush=True)


if __name__ == "__main__":
    main()
