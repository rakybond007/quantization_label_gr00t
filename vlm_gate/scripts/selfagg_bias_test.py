"""집계를 모델에 맡긴 판이 **한쪽으로 쏠리는가.**

`judge_selfagg.py` 와 같은 물음·같은 그림·같은 계산 사실을 쓰되, 39 장면이 아니라
태스크당 60 장면(24 태스크, 1440 장면)을 돌린다. 장면마다 기존 5문항 라벨의
A..E 와 conf 를 같이 실어 두었으므로, 같은 프레임에서 "우리가 더한 값" 과
"모델이 낸 값" 을 짝지어 볼 수 있다.

    python selfagg_bias_test.py <port> <scenes.json> <out.jsonl>

SPEED_PCT=0 이면 속도 줄이 예전 절대 경계 문구로 돌아간다 -- 프롬프트의 그 한
줄만 바꿔 같은 장면을 두 번 돌리면 속도 표현이 판정을 움직이는지 분리된다.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.path.expanduser("~/quantization_agent_workspace/vlm_gate")
TILES = f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
BATCH = int(os.environ.get("SA_BATCH", 8))


def main():
    import pandas as pd
    from PIL import Image
    from vlm_gate import VLMGate
    from robocasa_descriptors import descriptors, facts_text
    DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
          "robocasa_mg_gr00t_300")
    CH = int(json.load(open(f"{DS}/meta/info.json")).get("chunks_size") or 1000)
    port, scenes_f, out_f = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    # PROMPT 로 물음만 갈아끼운다. 자체집계 판(감점/가산점 문항 다섯 + 최종 등급)과
    # 직접 질문 판("두 액션을 하나로 합쳐도 성공하겠는가")을 같은 장면에서 견주려고
    # 둔 자리다 -- 자체집계 판의 최대 가중 문항(A, 0.667)이 "버튼·스위치는 압축 불가"
    # 인데, 폐루프 실측에서 그 태스크들이 오히려 압축에 가장 강했다.
    ASK = open(f"{BASE}/prompts/"
               f"{os.environ.get('PROMPT', 'robocasa_phase9_selfagg')}.txt").read().strip()
    src = open(f"{BASE}/prompts/robocasa_phase9.txt").read()
    G = src.split("### GUIDANCE\n", 1)[1].split("\n### QUESTION", 1)[0].strip()
    instr = {}
    for line in open(f"{DS}/meta/episodes.jsonl"):
        d = json.loads(line)
        c = [t for t in d.get("tasks", []) if isinstance(t, str) and len(t.split()) > 1]
        instr[d["episode_index"]] = c[0] if c else ""

    scenes = json.load(open(scenes_f))
    # 이미 한 것은 건너뛴다 -- 디버그 할당이 3시간에 끊기므로 이어서 돌 수 있어야 한다
    done = set()
    if os.path.exists(out_f):
        for line in open(out_f):
            try:
                r = json.loads(line)
                done.add((r["ep"], r["f"]))
            except Exception:
                pass
    scenes = [s for s in scenes if (s["ep"], s["f"]) not in done]
    print(f"장면 {len(scenes)}개 (건너뜀 {len(done)}) · batch {BATCH} · "
          f"speed={'pct' if os.environ.get('SPEED_PCT','1') != '0' else 'abs'}",
          flush=True)

    gate = VLMGate(f"http://127.0.0.1:{port}", timeout=900)
    acts, bad, n = {}, 0, 0
    with open(out_f, "a") as fh:
        for b0 in range(0, len(scenes), BATCH):
            grp = scenes[b0:b0 + BATCH]
            payload = []
            for s in grp:
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
                payload.append(([im[:, k * w:(k + 1) * w] for k in range(3)],
                                f"{instr.get(ep, '')}\n{facts_text(x)}"))
            try:
                rs = gate.judge_batch(payload, G, question=ASK, n_ask=1, n_grade=5)
            except Exception as e:                              # noqa: BLE001
                print(f"  batch {b0} 실패: {e}", flush=True)
                continue
            for s, r in zip(grp, rs):
                pick = (r.get("picks") or [None])[0]
                bad += int(pick is None)
                n += 1
                fh.write(json.dumps({**s, "Z": pick,
                                     "text": (r.get("text") or "")[:200]},
                                    ensure_ascii=False) + "\n")
            fh.flush()
            if (b0 // BATCH) % 10 == 0:
                print(f"  {n}/{len(scenes)} 형식실패 {bad}", flush=True)
    print(f"[bias] {n} 장면, 형식실패 {bad} -> {out_f}", flush=True)


if __name__ == "__main__":
    main()
