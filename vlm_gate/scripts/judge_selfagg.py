"""집계를 모델에게 맡긴 판. 등급 다섯을 받아 우리가 더하는 대신, 어느 문항이
감점이고 어느 문항이 가산점인지 알려 주고 **최종 판단 하나**를 받는다.

`judge_ab.py` 와 같은 장면·같은 그림·같은 계산 사실을 쓴다. 바뀌는 것은 마지막
물음뿐이다 -- 그래야 "집계를 누가 하느냐" 만 비교된다.
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.path.expanduser("~/quantization_agent_workspace/vlm_gate")
TILES = f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
CHUNKS = int(json.load(open(f"{DS}/meta/info.json")).get("chunks_size") or 1000)


def main():
    import pandas as pd
    from PIL import Image
    from vlm_gate import VLMGate
    from robocasa_descriptors import descriptors, facts_text
    port, scenes_f, out_f = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    ASK = open(f"{BASE}/prompts/robocasa_phase9_selfagg.txt").read().strip()
    src = open(f"{BASE}/prompts/robocasa_phase9.txt").read()
    G = src.split("### GUIDANCE\n", 1)[1].split("\n### QUESTION", 1)[0].strip()
    instr = {}
    for line in open(f"{DS}/meta/episodes.jsonl"):
        d = json.loads(line)
        c = [t for t in d.get("tasks", []) if isinstance(t, str) and len(t.split()) > 1]
        instr[d["episode_index"]] = c[0] if c else ""
    gate = VLMGate(f"http://127.0.0.1:{port}", timeout=600)
    acts, bad, n = {}, 0, 0
    with open(out_f, "w") as fh:
        for s in json.load(open(scenes_f)):
            ep, f = s["ep"], s["f"]
            if ep not in acts:
                acts.clear()
                acts[ep] = np.stack(pd.read_parquet(
                    f"{DS}/data/chunk-{ep // CHUNKS:03d}/"
                    f"episode_{ep:06d}.parquet")["action"].values)
            x = descriptors(acts[ep], f)
            im = np.array(Image.open(f"{TILES}/{s['tile']}.png").convert("RGB"))
            w = im.shape[1] // 3
            views = [im[:, k * w:(k + 1) * w] for k in range(3)]
            # **한 칸짜리 등급 하나.** n_ask=1 이라 서버가 "Z) " 슬롯 하나를 둔다.
            r = gate.judge(views, f"{instr.get(ep,'')}\n{facts_text(x)}", G,
                           question=ASK, n_ask=1, n_grade=5, mode="text")
            pick = (r.get("picks") or [None])[0]
            bad += int(pick is None)
            n += 1
            fh.write(json.dumps({**{k: s[k] for k in ("task", "phase", "ep", "f", "tile")},
                                 "pick": pick, "text": r.get("text", "")},
                                ensure_ascii=False) + "\n")
            if n % 8 == 0:
                print(f"  {n}/39  형식실패 {bad}", flush=True)
    print(f"[selfagg] {n}장면, 형식실패 {bad} -> {out_f}", flush=True)


if __name__ == "__main__":
    main()
