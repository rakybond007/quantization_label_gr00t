"""human_data 두 데이터셋 전량 라벨링 (stride 16, 청크 길이와 같아 겹침 없음).

문항은 libero 정본에서 셋을 고친 판이다 (analysis/human_data/RESULT_QUESTIONS.md):
  C  컷을 "받아주냐" -> "입구가 넉넉하냐"
  D  `or about to` 제거
  F  B 의 거울 추가 (부호 −1)

계산 사실은 humandata_descriptors -- 액션이 절대 관절이라 병합이 블록-라스트이고
문턱은 데이터셋 자기 분포에서 나온다.

국면 정답을 같이 적는다. **문턱은 에피소드별로 정규화한다** -- 전역 상수를 쓰면
물체마다 쥐는 폭이 달라(0.62 vs 0.95) pnp 101에피 중 33에피가 문턱을 못 넘었다.
그리고 **모든 잡기-놓기 주기를 쓴다** -- 첫 주기만 쓰면 long 은 154주기 중 103개를
버린다(물건이 셋이다).

**나간 전문을 출력 폴더에 남긴다.** 없으면 나중에 어떤 프롬프트로 만든 라벨인지
복구할 수 없다(robocasa 에서 실제로 그 일이 있었다).

  python humandata_label_full.py <출력디렉터리>
"""
import base64
import importlib
import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import av
import numpy as np
import pandas as pd
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{BASE}/scripts")

ROOT = "/sjw_alinlab2/home/taekwan/Data/human_data"

# **판본이 GUIDANCE/VIEW/ASK 를 들고 있으면 그것을 쓴다.** v1 은 libero v3c 파일을
# 읽어 문자열로 세 군데를 갈아끼웠다 -- 어떤 문항으로 만든 라벨인지 이 파일을 읽어야만
# 알 수 있었고, 부호·가중은 아예 어디에도 없어 humandata_tau.py 가 런타임에 만들었다.
# v2 부터는 humandata_v2_checks.py 하나에 다 있다.
#
#   CHECKS=humandata_v2_checks DESCRIPTORS=humandata_v2_descriptors \
#     python humandata_label_full.py <출력디렉터리>
_CK_NAME = os.environ.get("CHECKS", "")
CK = importlib.import_module(_CK_NAME) if _CK_NAME else None
D = importlib.import_module(os.environ.get("DESCRIPTORS", "humandata_descriptors"))

if CK is not None:
    GUID, VIEW, ASK = CK.GUIDANCE, CK.VIEW, CK.ASK
    Q = tuple(sorted(CK.SIGN))
    NGRADE = CK.NGRADE
else:
    # v1 경로 -- libero v3c 에 문자열 패치. 재현용으로만 남긴다.
    EV = f"{BASE}/analysis/_evolver/_libero"
    GUID = open(f"{EV}/libero_guidance_v3c.txt").read().strip()
    _ASK = open(f"{EV}/libero_questions_v3c.txt").read().strip()
    _C_OLD = ("C) Is an object being put into something that would catch it -- a basket, a bin,\n"
              "   a bowl -- so that landing off-centre changes nothing?")
    _C_NEW = ("C) Is the place this object has to end up ROOMY FOR IT -- a wide bowl, an open\n"
              "   basket, a broad plate with slack all round -- rather than a cup, a jar or a\n"
              "   slot barely wider than the thing itself?")
    _D_OLD = ("D) Is the hand letting go, or about to -- the object already resting where it is\n"
              "   meant to end up, so that what remains is only to open and withdraw?")
    _D_NEW = ("D) Is the object ALREADY RESTING where it is meant to end up, taking its own\n"
              "   weight, with nothing left but to open the hand and draw back?")
    _F = ("F) Is the robot LOWERING what it still holds onto its target right now -- the object\n"
          "   still in the hand, coming down onto the spot it must end up, not yet let go?")
    assert _C_OLD in _ASK and _D_OLD in _ASK, "문항 원문을 못 찾았다 -- 파일이 바뀌었나"
    ASK = _ASK.replace(_C_OLD, _C_NEW, 1).replace(_D_OLD, _D_NEW, 1)
    ASK = ASK.replace("\nAnswer:", "\n" + _F + "\nAnswer:", 1)
    VIEW = ("You are shown 2 camera views of this one moment: a scene view and a wrist "
            "(eye-in-hand) close-up. The wrist camera is mounted on the gripper, so objects "
            "normally look close in it -- general closeness is normal. Use the wrist view "
            "only to spot the actual grasp-closure or fine-insertion instant.")
    Q = tuple("ABCDEF")
    NGRADE = 5

CHUNK = 16
# stride 16 이면 청크 길이와 같아 전 프레임을 겹침도 빈틈도 없이 한 번씩 덮는다.
# allex 배달본도 stride 16 이다. 8 로 주면 절반씩 겹치는 2배 밀도가 된다.
STRIDE = int(os.environ.get("HD_STRIDE", "16"))
# 데이터셋당 상한. 0 이면 전량. 문구 A/B 팔을 같은 청크에서 돌릴 때 쓴다.
_LIMIT = int(os.environ.get("HD_LIMIT", "0"))
MODEL = os.environ.get("HD_MODEL", "gemini/gemini-3.8-flash")
CONC = int(os.environ.get("HD_CONC", "16"))
KEY = open(os.path.expanduser("~/.config/litellm/key")).read().strip()
URL = open(os.path.expanduser("~/.config/litellm/base")).read().strip() + "/v1/chat/completions"
_lock = threading.Lock()
STAT = {"n": 0, "bad": 0, "cost": 0.0, "t0": time.time()}


def ask_one(imgs, fx, instr):
    content = [{"type": "image_url",
                "image_url": {"url": "data:image/png;base64," + base64.b64encode(i).decode()}}
               for i in imgs]
    content.append({"type": "text",
                    "text": f"{GUID}\n\n{VIEW}\n\nThe robot was told: {instr}\n\n{fx}\n\n{ASK}"})
    body = {"model": MODEL, "max_tokens": 1024,
            "messages": [{"role": "user", "content": content}]}
    # `temperature` 는 최신 Anthropic 모델(opus-4-7 이상 · opus-5 · fable)에서 400 이다.
    if not any(k in MODEL for k in ("opus-5", "opus-4-7", "opus-4-8", "fable")):
        body["temperature"] = 0
    # `reasoning_effort` 는 Anthropic 경로 전체에서 400 이다. Gemini 계열에만 붙인다.
    if "gemini" in MODEL or "gpt" in MODEL:
        body["reasoning_effort"] = os.environ.get("HD_EFFORT", "low")
    data = json.dumps(body).encode()
    for a in range(5):
        try:
            req = urllib.request.Request(URL, data=data, headers={
                "Authorization": f"Bearer {KEY}", "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as resp:
                cost = float(resp.headers.get("x-litellm-response-cost") or 0.0)
                j = json.load(resp)
            return j["choices"][0]["message"]["content"], cost
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and a < 4:
                time.sleep(4 * (a + 1))
                continue
            return None, 0.0
        except Exception:
            if a < 4:
                time.sleep(4 * (a + 1))
                continue
            return None, 0.0


def parse(t):
    g = {}
    for line in (t or "").splitlines():
        s = line.strip().lstrip("*# ")
        if len(s) >= 3 and s[0] in Q and s[1] == ")":
            d = "".join(c for c in s[2:] if c.isdigit())
            if d:
                g[s[0]] = max(1, min(NGRADE, int(d[0])))
    return g if len(g) == len(Q) else None


def png(arr, px=448):
    im = Image.fromarray(arr).convert("RGB")
    im.thumbnail((px, px))
    b = io.BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def phase_map(a):
    """국면 정답. 문턱은 이 에피의 최대값으로 정규화한다."""
    g = a[:, 7]
    t = max(0.15, 0.5 * float(g.max()))
    gz = g > t
    tr = np.flatnonzero(np.diff(gz.astype(int)))
    cl = [x for x in tr if gz[x + 1]]
    op = [x for x in tr if not gz[x + 1]]
    cyc = []
    for c in cl:
        nx = [o for o in op if o > c]
        # 24스텝보다 짧은 주기는 접근·운반·내려놓기가 겹친다
        if nx and (not cyc or c > cyc[-1][1]) and nx[0] - c >= 24:
            cyc.append((c, nx[0]))
    m = {}
    for c0, o0 in cyc:
        m[max(0, c0 - 12)] = "1집으러"
        m[c0] = "2파지"
        m[(c0 + o0) // 2] = "3운반"
        m[max(c0 + 1, o0 - 12)] = "4내려놓으러"
        m[o0] = "5물러남"
    return m


def nearest_phase(f, pm):
    """stride 16 격자는 국면 지점과 정확히 안 맞으므로 8스텝 안이면 그 국면으로 본다."""
    if not pm:
        return None
    k = min(pm, key=lambda x: abs(x - f))
    return pm[k] if abs(k - f) <= 8 else None


def collect(ds, done, batch=int(os.environ.get("HD_BATCH", "6"))):
    R = f"{ROOT}/{ds}/lerobot"
    info = json.load(open(f"{R}/meta/info.json"))
    instr = {}
    for l in open(f"{R}/meta/episodes.jsonl"):
        d = json.loads(l)
        t = [x for x in d.get("tasks", []) if isinstance(x, str) and len(x.split()) > 1]
        if t:
            instr[d["episode_index"]] = t[0]
    # **에피 batch 개씩 모아 내보내는 제너레이터다.** batch 기본값이 6 인 이유:
    # 이 계정의 cgroup 메모리 한도가 4 GiB 다
    # (/sys/fs/cgroup/user.slice/user-<uid>.slice/memory.max). 20에피를 모으면
    # 디코딩 버퍼와 PNG 가 겹쳐 RSS 가 튀고, 다른 파이썬이 같이 떠 있으면 OOM 킬러가
    # 조용히 죽인다(트레이스백 없이 SIGKILL -- 2026-09-16~17 에 네 번). 전에는 데이터셋 전체(101 +
    # 51 에피)를 다 디코딩한 뒤에야 첫 요청을 보냈고, 그동안 로그인 노드에서 33분
    # 넘게 아무것도 안 나오다가 죽었다(2026-09-16~17 에 세 번). 지금은 20에피마다
    # 바로 보내고 프레임을 버린다 -- 메모리도 그만큼만 든다.
    jobs = []
    nsent = 0
    for ep in sorted(instr):
        if _LIMIT and nsent + len(jobs) >= _LIMIT:
            break
        ch = ep // info["chunks_size"]
        try:
            a = np.stack(pd.read_parquet(
                f"{R}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception:
            continue
        pm = phase_map(a)
        want = [f for f in range(0, max(len(a) - CHUNK, 0), STRIDE)
                if (ds, ep, f) not in done]
        if not want:
            continue
        frames = {}
        for key, slot in (("exterior_image_1_left", "scene"),
                          ("wrist_image_left", "wrist")):
            p = f"{R}/videos/chunk-{ch:03d}/observation.images.{key}/episode_{ep:06d}.mp4"
            need = set(want)
            try:
                c = av.open(p)
                for i, fr in enumerate(c.decode(video=0)):
                    if i in need:
                        frames[(slot, i)] = png(fr.to_ndarray(format="rgb24"))
                        need.discard(i)
                        if not need:
                            break
                c.close()
            except Exception:
                pass
        for f in want:
            if ("scene", f) in frames and ("wrist", f) in frames:
                x = D.descriptors(a, f, dataset=ds)
                jobs.append((ds, ep, f, [frames[("scene", f)], frames[("wrist", f)]],
                             D.facts_text(x), instr[ep], x, nearest_phase(f, pm)))
        if len(jobs) >= batch:
            yield jobs
            nsent += len(jobs)
            jobs = []
    if jobs:
        yield jobs

def run(job):
    ds, ep, f, imgs, fx, instr, x, ph = job
    txt, cost = ask_one(imgs, fx, instr)
    g = parse(txt)
    with _lock:
        STAT["n"] += 1
        STAT["cost"] += cost
        if g is None:
            STAT["bad"] += 1
        if STAT["n"] % 200 == 0:
            el = time.time() - STAT["t0"]
            print(f"  {STAT['n']} · ${STAT['cost']:.2f} · {STAT['n'] / el:.1f} req/s",
                  flush=True)
    return {"ds": ds, "ep": int(ep), "f": int(f), "phase": ph, **(g or {}),
            "merge_k2": float(x["merge_k2"]),
            "grip_closed_frac": float(x["grip_closed_frac"])}


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    open(f"{out}/PROMPT.txt", "w").write(
        "[USER]\n<2 images>\n" + GUID + "\n\n" + VIEW + "\n\n"
        "The robot was told: {instruction}\n\n{computed facts}\n\n" + ASK + "\n")
    # **부호·가중도 같이 남긴다.** v1 은 남기지 않아서 humandata_tau.py 가 libero 것을
    # 가져와 런타임에 만들었고, 어떤 가중으로 계산한 conf 인지 복구할 수 없었다.
    json.dump({"checks": _CK_NAME or "(v1 libero 문자열 패치)",
               "descriptors": os.environ.get("DESCRIPTORS", "humandata_descriptors"),
               "model": MODEL, "stride": STRIDE, "conc": CONC, "ngrade": NGRADE,
               "questions": list(Q),
               "sign": (CK.SIGN if CK is not None else None),
               "weight": (CK.WEIGHT if CK is not None else None),
               "name": (CK.NAME if CK is not None else None)},
              open(f"{out}/meta.json", "w"), indent=1, ensure_ascii=False)
    fp = f"{out}/labels.jsonl"
    done = set()
    if os.path.exists(fp):
        for l in open(fp):
            try:
                r = json.loads(l)
                done.add((r["ds"], r["ep"], r["f"]))
            except Exception:
                pass
    print(f"재개 {len(done)}개", flush=True)
    # **데이터셋을 하나씩 처리한다.** 둘을 모아 두면 long 프레임 추출(51에피 x 2영상)
    # 동안 아무것도 안 쓰이고, 중간에 끊기면 그때까지 뽑은 프레임이 다 날아간다.
    fh = open(fp, "a")
    for ds in ("pnp_task", "long_horizon_task"):
        print(f"[{ds}] 시작", flush=True)
        ex = ThreadPoolExecutor(CONC)
        for batch in collect(ds, done):
            for r in ex.map(run, batch):
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"  [{ds}] 누적 {STAT['n']:,} · ${STAT['cost']:.2f} · "
                  f"{STAT['n']/max(time.time()-STAT['t0'],1e-9):.1f} req/s", flush=True)
        ex.shutdown()
        print(f"[{ds}] 완료 · 누적 {STAT['n']:,} · ${STAT['cost']:.2f}", flush=True)
    fh.close()
    el = time.time() - STAT["t0"]
    print(f"완료 {STAT['n']} · 실패 {STAT['bad']} · ${STAT['cost']:.2f} · "
          f"{el / 60:.1f}분 -> {fp}", flush=True)


if __name__ == "__main__":
    main()
