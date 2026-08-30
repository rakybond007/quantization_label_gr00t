"""전 프레임 라벨링 — 타일 파일 없이 영상에서 바로 디코딩한다.

왜 타일을 안 쓰는가
-------------------
기존 경로(cosmos_1call_v6.py)는 미리 구운 타일 PNG 를 읽는다. 간격 8 에서 26만 개면
버틸 만하지만, 간격 1 이면 207만 개가 되고 그건 이 공유 파일시스템에서 감당이 안 된다
(디렉터리 하나에 200만 파일, 약 200GB). 여기서는 에피소드마다 비디오 리더를 한 번 열고
프레임을 순서대로 꺼낸다. 순차 접근이라 디코딩이 싸고, 중간 산출물이 없어 굽는 단계도
사라진다.

왜 전 프레임인가
----------------
VLA 데이터로더는 모든 시점을 돈다. t 시점 샘플의 정답은 "t 에서 시작하는 16스텝 창을
압축해도 되는가" 인데, 간격 8 로 라벨링하면 t, t+8 만 답이 있고 t+4 는 답이 없다.
그런 샘플은 gate_valid=0 으로 손실에서 빠지므로 게이트는 8분의 1의 시점에서만 배운다
(실측 12.8%). 게다가 추론 때 재계획 시점은 8의 배수가 아니므로, 학습한 적 없는
위상에서 판단하게 된다.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

BASE = os.path.expanduser("~/quantization_agent_workspace/vlm_gate")
sys.path.insert(0, f"{BASE}/scripts")
from vlm_gate import VLMGate                                  # noqa: E402
from robocasa_descriptors import descriptors, facts_text, computed_risk  # noqa: E402

DS = os.environ.get(
    "DENSE_DS",
    "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300")
PORT, SHARD, NSH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
STRIDE = int(os.environ.get("LABEL_STRIDE", "1"))
TAG = os.environ.get("TAG", "dense_phase6")
OUT = f"{BASE}/output/_gate_distill/{TAG}_s{NSH}_{SHARD}.jsonl"

GUIDANCE_FILE = os.environ.get("GUIDANCE_FILE", "robocasa_guidance_phase_v5.txt")
G = open(f"{BASE}/analysis/_evolver/_varkA/{GUIDANCE_FILE}").read().strip()
sys.path.insert(0, f"{BASE}/scripts")
import cosmos_1call_v6 as _ref                                # noqa: E402
ASK, NQ, SLOTS = _ref.ASK6, 5, "ABCDE"

info = json.load(open(f"{DS}/meta/info.json"))
CS = info["chunks_size"]
vks = [k for k in info["features"] if info["features"][k].get("dtype") == "video"]
VK = ([k for k in vks if "left" in k]
      + [k for k in vks if "right" in k and "wrist" not in k]
      + [k for k in vks if "wrist" in k])

instr, lengths = {}, {}
for line in open(f"{DS}/meta/episodes.jsonl"):
    d = json.loads(line)
    c = [t for t in d.get("tasks", []) if isinstance(t, str) and len(t.split()) > 1 and t != "Valid"]
    instr[d["episode_index"]] = c[0] if c else ""
    lengths[d["episode_index"]] = int(d.get("length", 0))

# 이미 쓴 (ep, f) 는 건너뛴다. background 파티션은 선점되므로 재개가 전제다.
done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            r = json.loads(line)
            done.add((r["ep"], r["f"]))
        except Exception:
            pass
print(f"[dense] shard{SHARD}/{NSH} stride={STRIDE} 재개 {len(done)}개", flush=True)

gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=180)
out = open(OUT, "a")
import pandas as pd                                            # noqa: E402
from decord import VideoReader                                 # noqa: E402

eps = [e for e in sorted(lengths) if e % NSH == SHARD]
n = 0
for ei, ep in enumerate(eps):
    frames_wanted = [f for f in range(0, max(lengths[ep] - 4, 0), STRIDE)
                     if (ep, f) not in done]
    if not frames_wanted:
        continue
    ch = ep // CS
    try:
        a = np.stack(pd.read_parquet(
            f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        vrs = [VideoReader(f"{DS}/" + info["video_path"].format(
            episode_chunk=ch, episode_index=ep, video_key=k)) for k in VK]
    except Exception as e:
        print(f"[dense] ep{ep} 열기 실패: {e}", flush=True)
        continue
    nfr = min(len(a), min(len(v) for v in vrs))
    for f in frames_wanted:
        if f >= nfr - 4:
            break
        x = descriptors(a, f)
        views = [Image.fromarray(v[f].asnumpy()) for v in vrs]
        r = gate.judge(views, f"{instr.get(ep, '')}\n{facts_text(x)}", G,
                       question=ASK, n_ask=NQ)
        c = r.get("confidences") or [0.0] * NQ
        out.write(json.dumps({
            "ep": ep, "f": f,
            **{k: float(v) for k, v in zip(SLOTS, c)},
            **computed_risk(x), "speed_mean": x["speed_mean"],
            "ans": r.get("answer", "")}) + "\n")
        n += 1
        if n % 200 == 0:
            print(f"[dense] shard{SHARD}: {n} (ep {ei+1}/{len(eps)})", flush=True)
            out.flush()
    del vrs
out.close()
print(f"[dense] shard{SHARD} 완료 {n} -> {OUT}", flush=True)
