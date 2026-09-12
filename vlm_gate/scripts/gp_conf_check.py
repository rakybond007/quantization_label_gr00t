"""등급 분포(gp)를 실제로 받아, 정수 등급 conf 와 기댓값 conf 를 견준다.

단일 경로가 grade_probs 를 내지 않아 robocasa 는 라벨·게이트 양쪽에서 늘 정수
등급으로 떨어져 있었다(라벨 파일 전수 확인, gp 있는 행 0%). 그것을 고친 뒤,
같은 장면·같은 답에서 두 conf 가 얼마나 다른지가 이 스크립트가 내는 숫자다.

    python gp_conf_check.sh.py <port> <scenes.json> <out.jsonl>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.path.expanduser("~/quantization_agent_workspace/vlm_gate")
TILES = f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
N = int(os.environ.get("GP_N", 240))


def main():
    import pandas as pd
    from PIL import Image
    from vlm_gate import VLMGate
    from robocasa_descriptors import descriptors, facts_text
    import phase9_checks as P
    from ratio_label import confidence, expected_grades
    DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
          "robocasa_mg_gr00t_300")
    CH = int(json.load(open(f"{DS}/meta/info.json")).get("chunks_size") or 1000)
    port, scenes_f, out_f = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    instr = {}
    for line in open(f"{DS}/meta/episodes.jsonl"):
        d = json.loads(line)
        c = [t for t in d.get("tasks", []) if isinstance(t, str) and len(t.split()) > 1]
        instr[d["episode_index"]] = c[0] if c else ""
    scenes = json.load(open(scenes_f))[:N]
    gate = VLMGate(f"http://127.0.0.1:{port}", timeout=900)
    acts, n_gp, n = {}, 0, 0
    with open(out_f, "w") as fh:
        for s in scenes:
            ep = s["ep"]
            if ep not in acts:
                if len(acts) > 24:
                    acts.clear()
                acts[ep] = np.stack(pd.read_parquet(
                    f"{DS}/data/chunk-{ep // CH:03d}/"
                    f"episode_{ep:06d}.parquet")["action"].values)
            x = descriptors(acts[ep], s["f"])
            im = np.array(Image.open(f"{TILES}/{s['tile']}.png").convert("RGB"))
            w = im.shape[1] // 3
            views = [im[:, k * w:(k + 1) * w] for k in range(3)]
            r = gate.judge(views, f"{instr.get(ep, '')}\n{facts_text(x)}",
                           P.GUIDANCE, question=P.ASK, n_ask=5,
                           n_grade=P.NGRADE, mode="text")
            picks = r.get("picks") or [None] * 5
            if any(p is None for p in picks):
                continue
            gp = r.get("grade_probs")
            rec = {q: int(v) for q, v in zip("ABCDE", picks)}
            c_int = confidence(dict(rec), P.SIGN, P.WEIGHT, P.NGRADE)
            c_exp = c_int
            if gp and len(gp) == 5:
                n_gp += 1
                c_exp = confidence({**rec, "gp": gp}, P.SIGN, P.WEIGHT, P.NGRADE)
            n += 1
            fh.write(json.dumps({**{k: s[k] for k in ("task", "ep", "f")},
                                 "picks": picks, "gp": gp,
                                 "eg": expected_grades(gp, P.NGRADE) if gp else None,
                                 "conf_int": c_int, "conf_exp": c_exp}) + "\n")
            if n % 40 == 0:
                print(f"  {n}/{len(scenes)} gp있음 {n_gp}", flush=True)
    print(f"[gp] {n} 장면 · gp 있는 행 {n_gp} ({n_gp/max(n,1):.1%}) -> {out_f}",
          flush=True)


if __name__ == "__main__":
    main()
