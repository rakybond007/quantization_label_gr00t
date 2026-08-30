"""LIBERO 청크 라벨러 — cosmos_1call_v6.py 의 LIBERO 판.

로보카사와 같은 1콜 설계: 2층(계산)은 libero_descriptors 가 사실로 진술하고,
3층(VLM)은 정지화면으로만 알 수 있는 다섯 문항을 한 번의 이미지 prefill 로 답한다.

로보카사와 다른 점.
  * 뷰가 둘(front + left_wrist)이다. 타일을 3등분하지 않고 2등분한다.
  * 액션이 7차원(0:3 위치 델타 · 3:6 회전 델타 · 6 그리퍼 ±1)이다 —
    libero_descriptors 가 이미 그 규격으로 계산한다.
  * 프레임 스트라이드 4 (10fps · 평균 162 스텝). 로보카사 stride 8/20fps 와
    같은 0.4 초 간격이다.

재개(resume) 규율. 백그라운드 파티션은 선점당한다. 시작할 때
  1) 기존 출력에서 파싱되는 줄만 남기고 다시 쓴다 (선점으로 끊긴 마지막 줄 제거),
  2) (ep,f) 를 done 으로 올려 다시 내보내지 않는다.
그래서 재큐된 샤드가 행을 중복시키지 않는다 — qgate labels 가 잡는 실패 모드다.
판정 실패(judge 오류)는 행을 쓰지 않고 건너뛴다. 0 으로 채워 쓰면 다음 실행이
그 청크를 완료로 보고 영영 다시 묻지 않는다.

사용법:  python libero_label_chunks.py <port> <shard> <nshards>
"""
import json, os, sys, numpy as np, pandas as pd
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_gate import VLMGate
from libero_descriptors import descriptors, facts_text, computed_risk

BASE = "/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
DS = "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"
TIL = f"{BASE}/output/_gate_distill/libero_full/tiles"
MAN = os.environ.get("MANIFEST", f"{BASE}/output/_gate_distill/libero_tiles_manifest.txt")
TAG = os.environ.get("TAG", "libero_v1")
NVIEW = 2
NQ, SLOTS = 5, "ABCDE"

PORT, SHARD, NSH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
LIMIT = int(os.environ.get("LIMIT", "0"))          # >0 이면 스모크용으로 이만큼만
OUT = f"{BASE}/output/_gate_distill/{TAG}_s{NSH}_{SHARD}.jsonl"

G = open(f"{BASE}/analysis/_evolver/_libero/libero_guidance_v1.txt").read().strip()
ASK = open(f"{BASE}/analysis/_evolver/_libero/libero_questions_v1.txt").read().strip()

info = json.load(open(f"{DS}/meta/info.json"))
instr = {}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d = json.loads(l)
    c = [t for t in d.get("tasks", []) if isinstance(t, str) and len(t.split()) > 1]
    instr[d["episode_index"]] = c[0] if c else ""

# --- 재개: 온전한 줄만 남기고 다시 쓴 뒤 done 집합을 만든다 ---
done = set()
if os.path.exists(OUT):
    keep = []
    for l in open(OUT):
        try:
            r = json.loads(l)
        except Exception:
            continue                      # 선점으로 잘린 줄 — 버린다
        if (r.get("ep"), r.get("f")) in done:
            continue                      # 과거 실행이 남긴 중복 — 한 번만 남긴다
        done.add((r["ep"], r["f"])); keep.append(json.dumps(r))
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        f.write("".join(s + "\n" for s in keep))
    os.replace(tmp, OUT)
    print(f"shard{SHARD}: resume, {len(done)} chunks already done", flush=True)

gate = VLMGate(f"http://127.0.0.1:{PORT}", timeout=180)

acts = {}
def A(ep):
    if ep not in acts:
        ch = ep // info["chunks_size"]
        try:
            acts[ep] = np.stack(pd.read_parquet(
                f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception:
            acts[ep] = None
        if len(acts) > 40:
            for k in list(acts)[:20]:
                acts.pop(k, None)
    return acts[ep]

out = open(OUT, "a")
n = skipped = 0
for nm in sorted(open(MAN).read().split()):
    ep = int(nm[2:6]); f = int(nm.split("_f")[1][:3])
    if ep % NSH != SHARD: continue
    if (ep, f) in done: continue
    a = A(ep)
    if a is None or f >= len(a) - 4: continue
    x = descriptors(a, f)
    try:
        im = np.array(Image.open(f"{TIL}/{nm}").convert("RGB"))
    except Exception:
        skipped += 1; continue
    h, w, _ = im.shape
    views = [Image.fromarray(im[:, k * w // NVIEW:(k + 1) * w // NVIEW]) for k in range(NVIEW)]
    ins = f"{instr.get(ep, '')}\n{facts_text(x)}"
    r = gate.judge(views, ins, G, question=ASK, n_ask=NQ)
    c = r.get("confidences")
    if not c or len(c) != NQ:               # 판정 실패 — 쓰지 않고 다음 실행에 넘긴다
        skipped += 1
        if skipped % 20 == 1:
            print(f"shard{SHARD}: judge miss ep{ep} f{f}: {r.get('error','')}", flush=True)
        continue
    rec = {"ep": ep, "f": f, **{k: float(v) for k, v in zip(SLOTS, c)},
           **computed_risk(x), "speed_mean": x["speed_mean"], "ans": r.get("answer", "")}
    out.write(json.dumps(rec) + "\n"); out.flush()   # 선점에 대비해 매 행 flush
    n += 1
    if n % 200 == 0: print(f"shard{SHARD}: {n}", flush=True)
    if LIMIT and n >= LIMIT: break
out.close()
print(f"shard{SHARD} 완료 {n} (skipped {skipped}) -> {OUT}")
