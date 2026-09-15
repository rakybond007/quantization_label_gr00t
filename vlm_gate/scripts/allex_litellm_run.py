"""allex 전량 라벨링 -- LiteLLM 프록시로 Gemini, 동시 요청, 재개 가능.

에피소드 번호 순서로 돈다. v1_v2 는 v1_v2_v3_v4 의 에피 0~159 와 영상까지
동일하므로, 0~159 를 끝내면 그 자체로 v1_v2 라벨셋이 된다.

    ALLEX_EP_LO=0 ALLEX_EP_HI=159 python allex_litellm_run.py <출력.jsonl>

환경변수
    ALLEX_EP_LO / ALLEX_EP_HI   에피 범위 (양끝 포함)
    ALLEX_STRIDE                기본 16 = 청크 길이. 성글게 하면 사이를 메울 수
                                없다 (docs/LABELING_STRIDE_AND_LITELLM.md 2b)
    ALLEX_CONC                  동시 요청, 기본 64 (실측 7.7 req/s, 실패 0)
    ALLEX_CHECKS_MODULE         기본 allex_v4c_checks
    ALLEX_JUDGE                 가중 선택, 기본 gemini

출력은 청크마다 한 줄이고 이미 있는 줄은 건너뛴다. 중간에 죽어도 다시 돌리면
이어서 한다.
"""
import base64, io, json, os, sys, time, threading, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, av
import importlib

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from allex_v2_common import descriptors
from allex_facts import facts as _facts
CK = importlib.import_module(os.environ.get("ALLEX_CHECKS_MODULE", "allex_v4c_checks"))

DS     = os.environ.get("ALLEX_DS", "/rlwrld2/home/david/frontier_demo_cumul/v1_v2_v3_v4")
OUT    = sys.argv[1]
EP_LO  = int(os.environ.get("ALLEX_EP_LO", "0"))
EP_HI  = int(os.environ.get("ALLEX_EP_HI", "319"))
STRIDE = int(os.environ.get("ALLEX_STRIDE", "16"))
CHUNK  = int(os.environ.get("ALLEX_CHUNK", "16"))
CONC   = int(os.environ.get("ALLEX_CONC", "64"))
MODEL  = os.environ.get("ALLEX_MODEL", "gemini/gemini-3.8-flash")
EFFORT = os.environ.get("ALLEX_EFFORT", "low")
KEY = open(os.path.expanduser("~/.config/litellm/key")).read().strip()
URL = open(os.path.expanduser("~/.config/litellm/base")).read().strip() + "/v1/chat/completions"
Q = tuple(sorted(CK.SIGN))

TASKS = {}
with open(f"{DS}/meta/tasks.jsonl") as f:
    for line in f:
        r = json.loads(line); TASKS[r["task_index"]] = r["task"]

_lock = threading.Lock()
STAT = {"n": 0, "bad": 0, "err": 0, "tin": 0, "tout": 0, "cost": 0.0}


def png_bytes(im):
    """PNG 로 굳혀서 PIL 객체를 즉시 버린다.

    PIL 512x512 RGB 두 장은 청크당 1.6MB 이고 한 에피(175청크)면 275MB 다.
    PNG 바이트로 굳히면 그 자리에서 3분의 1 이하가 되고, 요청을 만들 때만
    base64 로 감싸므로 살아 있는 문자열은 동시 요청 수만큼뿐이다.
    """
    b = io.BytesIO(); im.save(b, format="PNG")
    return b.getvalue()


def rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return float("nan")


def ask(imgs, fx, instr):
    content = [{"type": "image_url",
                "image_url": {"url": "data:image/png;base64," + base64.b64encode(i).decode()}}
               for i in imgs]
    content.append({"type": "text", "text":
        f"{CK.GUIDANCE}\n\nThe two images are the left and right camera views of "
        f"this one moment.\n\nThe robot was told: {instr}\n\n{fx}\n\n{CK.ASK}"})
    body = {"model": MODEL, "max_tokens": 1024, "temperature": 0,
            "messages": [{"role": "user", "content": content}]}
    if EFFORT:
        body["reasoning_effort"] = EFFORT
    data = json.dumps(body).encode()
    for a in range(5):
        try:
            req = urllib.request.Request(URL, data=data, headers={
                "Authorization": f"Bearer {KEY}", "content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as resp:
                cost = float(resp.headers.get("x-litellm-response-cost") or 0.0)
                j = json.load(resp)
            u = j.get("usage", {})
            return j["choices"][0]["message"]["content"], u, cost
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and a < 4:
                time.sleep(4 * (a + 1)); continue
            return None, {}, 0.0
        except Exception:
            if a < 4:
                time.sleep(4 * (a + 1)); continue
            return None, {}, 0.0


def parse(t):
    g = {}
    for line in (t or "").splitlines():
        s = line.strip().lstrip("*# ")
        if len(s) >= 3 and s[0] in Q and s[1] == ")":
            d = "".join(c for c in s[2:] if c.isdigit())
            if d:
                g[s[0]] = max(1, min(CK.NGRADE, int(d[0])))
    return g if len(g) == len(Q) else None


def frames_of(ep, want):
    """에피의 두 영상을 한 번씩 훑어 원하는 프레임만 512px 로 뽑는다."""
    got = {}
    for side in ("left", "right"):
        p = (f"{DS}/videos/chunk-{ep // 1000:03d}/"
             f"observation.images.camera_ego_{side}/episode_{ep:06d}.mp4")
        need = set(want)
        with av.open(p) as c:
            for i, fr in enumerate(c.decode(video=0)):
                if i in need:
                    got.setdefault(i, {})[side] = png_bytes(fr.to_image().resize((512, 512)))
                    need.discard(i)
                    if not need:
                        break
    return {f: [v["left"], v["right"]] for f, v in got.items() if len(v) == 2}


def one(job):
    ep, f, imgs, fx, instr, extra = job
    txt, u, cost = ask(imgs, fx, instr)
    with _lock:
        STAT["tin"] += u.get("prompt_tokens", 0) or 0
        STAT["tout"] += u.get("completion_tokens", 0) or 0
        STAT["cost"] += cost
    if txt is None:
        with _lock: STAT["err"] += 1
        return None
    g = parse(txt)
    if not g:
        with _lock: STAT["bad"] += 1
        return None
    conf = 0.5 * (1 + sum(CK.SIGN[q] * CK.WEIGHT[q] * ((g[q] - 1) / (CK.NGRADE - 1)) for q in Q))
    return {"ep": ep, "f": f, "instr": instr, **g, "conf": round(conf, 4),
            "model": MODEL, "effort": EFFORT, **extra}


done = set()
if os.path.exists(OUT):
    with open(OUT) as fh:
        for l in fh:
            try:
                r = json.loads(l); done.add((r["ep"], r["f"]))
            except Exception:
                pass
    print(f"[i] 이어서 시작 · 이미 {len(done)}청크", flush=True)
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)

EPS = []
with open(f"{DS}/meta/episodes.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if EP_LO <= r["episode_index"] <= EP_HI:
            EPS.append((r["episode_index"], r["length"]))
EPS.sort()
total = sum(max(1, (L - CHUNK) // STRIDE + 1) for _, L in EPS)
print(f"[i] 에피 {EPS[0][0]}~{EPS[-1][0]} ({len(EPS)}개) · 청크 {CHUNK} · stride {STRIDE} "
      f"· 예상 {total:,}청크 · 동시 {CONC} · {MODEL} effort={EFFORT}", flush=True)

out = open(OUT, "a")
t0 = time.time()

# 다음 에피의 영상 디코딩을 현재 에피의 API 호출과 겹친다. 직렬로 두면 디코딩이
# 벽시계의 절반을 먹는다 (실측 3.24 req/s vs 호출만 할 때 7.7).


def prepare(ep, L):
    want = [f for f in range(0, max(1, L - CHUNK + 1), STRIDE) if (ep, f) not in done]
    if not want:
        return None
    d = pd.read_parquet(f"{DS}/data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet")
    A = np.stack(d["action"].values)
    WR = np.stack(d["action.right_wrist_wrt_base"].values)
    WL = np.stack(d["action.left_wrist_wrt_base"].values)
    ti = int(d["task_index"].iloc[0]) if "task_index" in d.columns else 0
    instr = TASKS.get(ti, "")
    imgs = frames_of(ep, want)
    jobs = []
    for f in want:
        if f not in imgs:
            continue
        x = descriptors(A, WR, WL, f, CHUNK)
        jobs.append((ep, f, imgs[f], _facts(x), instr,
                     {"one_handed": bool(x["one_handed"]),
                      "merge_demand_k2": round(float(x["merge_demand_k2"]), 4)}))
    return jobs


# 에피를 하나씩만 들고 간다.  선행 디코딩으로 두 에피를 겹쳐 보유하면 최대
# 메모리가 두 배가 되고, 실측 이득은 3.24 -> 3.49 req/s 뿐이었다 (병목이
# 디코딩이 아니라 프록시 쪽이다).  겹침을 버리고 메모리를 낮게 유지한다.
with ThreadPoolExecutor(CONC) as pool:
    for ep, L in EPS:
        try:
            jobs = prepare(ep, L)
        except Exception as e:
            print(f"  [!] ep{ep} 준비 실패: {type(e).__name__} {e}", flush=True)
            continue
        if not jobs:
            continue
        for rec in pool.map(one, jobs):
            if rec:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                with _lock: STAT["n"] += 1
        out.flush()
        el = time.time() - t0
        jobs = None            # 이 에피의 이미지를 즉시 놓아준다
        print(f"  ep{ep} 끝 · 누적 {STAT['n']:,}/{total:,} · {STAT['n']/max(el,1):.2f} req/s "
              f"· ${STAT['cost']:.2f} · 형식실패 {STAT['bad']} 통신실패 {STAT['err']} "
              f"· RSS {rss_mb():.0f}MB · {el/60:.1f}분", flush=True)
out.close()
el = time.time() - t0
print(f"완료 {STAT['n']:,}청크 · 형식실패 {STAT['bad']} · 통신실패 {STAT['err']} "
      f"· 입력 {STAT['tin']:,} 출력 {STAT['tout']:,} · ${STAT['cost']:.2f} · {el/60:.1f}분 -> {OUT}")
