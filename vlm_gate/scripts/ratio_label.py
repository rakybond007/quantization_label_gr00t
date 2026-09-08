"""최종 라벨 -- 매 시점에 **몇 배속으로 갈지**를 하나의 값으로 적는다.

벤치마크를 인자로 받는다. **문항과 가중치는 벤치마크마다 다르고, 각자의 checks
모듈에서 읽는다** -- robocasa 는 `phase9_checks`, libero 는 `libero_v2_checks`.
여기에 박아 두면 robocasa 라벨을 libero 가중치로 계산하는 일이 조용히 일어난다.

지금까지는 두 갈래였다. F_level 이 접촉 가드로 level 1 을 강제하고, 가드가 서지
않으면 movement head 가 K 를 골랐다. 우리는 따로 confidence 를 만들었다.
그것을 하나로 합친다 (2026-09-08 사용자 지시).

    접촉이 잡히면            -> 1.0배 고정
    접촉이 아니면            -> 신뢰도 순서로 그 태스크의 [1.0, 상한] 안에 앉힌다

**어디에 앉히느냐는 아직 정해진 규칙이 없다.** 셋을 넣어 두고 `RATIO_RULE` 로
고른다. 지금까지 한 검증은 전부 **태스크 단위**(위험 태스크에서 감점 문항이
높은가)이고, 같은 태스크 **안에서** 어느 순간이 위험한지는 확인한 적이 없다.
그래서 규칙은 논증이 아니라 게이트 평가로 정해야 한다 -- 같은 평균 배속의 균일
대조군보다 나은지를 본다.

    global (기본)  **모든 시점을 한 줄로** 세워 눈금에 배분한 뒤 태스크 상한으로
                   자른다. 태스크마다 1.0 과 상한이 다 나오도록 강제하지 않는다 --
                   안전한 태스크는 전부 빠르게, 위험한 태스크는 전부 느리게 갈 수 있다
    levels         태스크 **안에서** 상한 아래 칸 수만큼 등분. 2.5 면 4분위, 2.0 이면 3분위.
                   태스크 사이 신호를 지우는데, 그건 이미 상한표가 담당하므로 두 번 다루는 것이다
    band           [1.0, 상한] 에 균등하게 편 뒤 가까운 눈금으로 스냅.
                   양끝 칸이 절반만 받는다(2.5 상한에서 17/33/33/17)
    mid            conf 0.5 를 경계로 상한 아니면 1.0. 0.5 는 가점 가중합과
                   감점 가중합이 같아지는 자리라 지어낸 선이 아니다

**배속은 연속값으로 적지 않는다.** 디코더가 이산이라 연속값을 적으면 결국
반올림해서 쓰게 되고, 그러면 결정이 반올림에서 일어난다. 처음부터 디코더가 가진
눈금으로 적는다. 그 눈금에서는 배속이 정확히 떨어진다
(`fractional_blocks.block_sizes` + `replan_rows`, `RATIOS_ARE_NOT_K.md`).

    GRID = (1.0, 1.5, 2.0, 2.5)

상한표는 **클립만 푼 사다리**에서 나온다. 구동부를 3배로 연 실행
(`released_*`)은 1.429배에서 이미 0.616 으로 무너져 압축이 아니라 제어기 손상을
재고 있으므로 상한표에 쓰지 않는다.

    python vlm_gate/scripts/ratio_label.py <bench> <labels.jsonl> [out.jsonl]
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SLOTS = "ABCDE"
RULE = "levels"
MISS = set()          # 상한표에서 못 찾은 태스크. 비어 있어야 한다

# 디코더가 가진 눈금. F_level 의 level_ks 와 같아야 한다.
GRID = (1.0, 1.5, 2.0, 2.5)
DEFAULT_BAND = (1.0, 2.0)

_HERE = os.path.dirname(os.path.abspath(__file__))
# 벤치마크 -> (checks 모듈, 에피소드->태스크 함수를 가진 모듈)
CHECKS = {"libero": "libero_v2_checks", "robocasa": "phase9_checks"}


def load_bench(bench):
    """그 벤치마크의 SIGN·WEIGHT·NGRADE 와 상한표를 가져온다."""
    if bench not in CHECKS:
        raise SystemExit(f"모르는 벤치마크 {bench}. 아는 것: {sorted(CHECKS)}")
    mod = __import__(CHECKS[bench])
    sign, weight = dict(mod.SIGN), dict(mod.WEIGHT)
    ngrade = int(getattr(mod, "NGRADE", 5))
    # 각 변의 합이 1 이어야 한다. 아니면 신뢰도가 0~1 을 벗어난다.
    for side, sel in (("감점", -1), ("가점", +1)):
        t = sum(weight[k] for k in weight if sign[k] == sel)
        if abs(t - 1.0) > 1e-6:
            raise SystemExit(f"{bench} {side} 가중치 합이 {t:.4f} 다. 1 이어야 한다")
    ceil_path = os.path.join(_HERE, "..", "analysis",
                             f"{bench}_task_ceilings.json")
    ceil = json.load(open(ceil_path)) if os.path.exists(ceil_path) else {}
    return sign, weight, ngrade, ceil, ceil_path


def confidence(picks, sign, weight, ngrade=5):
    """(1 + Σw·가점 − Σw·감점) / 2. 등급은 (g−1)/(n−1) 로 0~1 에 옮긴다."""
    d = float(ngrade - 1)
    risk = sum(weight[k] * (picks[k] - 1) / d for k in SLOTS if sign[k] < 0)
    safe = sum(weight[k] * (picks[k] - 1) / d for k in SLOTS if sign[k] > 0)
    return float(min(1.0, max(0.0, (1.0 + safe - risk) / 2.0)))


def band_place(confs, lo, hi):
    """신뢰도를 그 태스크의 띠에 앉힌다. 칸 안 **분위수**로 앉힌다.

    신뢰도의 절대 크기는 등급표 눈금이 정하는 임의값이라 그대로 띠에 곱하면
    안 된다 -- 눈금을 늘리기만 해도 배속이 바뀐다. 순서만 뜻이 있다.
    같은 답을 낸 시점은 같은 분위수를 받아 같은 값으로 묶인다.

    `allex_v3_checks.band_place` 와 같은 규칙이다.
    """
    c = np.asarray(confs, dtype=float)
    n = c.size
    if n == 0:
        return c
    if n == 1:
        return np.array([lo + 0.5 * (hi - lo)])
    order = c.argsort(kind="mergesort")
    rank = np.empty(n, dtype=float)
    rank[order] = np.arange(n, dtype=float)
    for v in np.unique(c):
        m = c == v
        rank[m] = rank[m].mean()
    return lo + (rank / (n - 1)) * (hi - lo)


def snap(x):
    """눈금 위의 가장 가까운 값으로. 라벨은 디코더가 실제로 낼 수 있는 값이어야 한다."""
    g = np.asarray(GRID, dtype=float)
    return float(g[int(np.argmin(np.abs(g - float(x))))])


DS = {"libero": "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/"
                "kimtaey/libero_gr00t_delta",
      "robocasa": "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/"
                  "kimtaey/robocasa_mg_gr00t_300"}


def ep_to_task(bench):
    """에피소드 -> 태스크. **번호로 추측하지 않는다** -- 데이터셋 순서를 가정하면
    조용히 틀리고, 그러면 띠가 통째로 어긋난다.

    robocasa 는 `episodes.jsonl` 의 `tasks[1]` 이 곧 태스크 이름이다
    (`["turn on the front right burner of the stove", "TurnOnStove", "Valid"]`).
    libero 는 그런 칸이 없어 지시문으로 맞춘다.
    """
    if bench == "libero":
        from libero_v2_verify import _ep_to_task
        return _ep_to_task()
    out = {}
    for line in open(f"{DS[bench]}/meta/episodes.jsonl"):
        d = json.loads(line)
        t = d.get("tasks") or []
        # 두 번째 칸이 태스크 이름이다. 한 단어이고 공백이 없다.
        for v in t[1:]:
            if isinstance(v, str) and v and " " not in v and v != "Valid":
                out[d["episode_index"]] = v
                break
    return out


def band_of(table, task):
    """그 태스크의 [하한, 상한]. 예전 형식(값 하나)도 읽는다.

    **하한이 1.0 이 아닐 수 있다.** allex 는 운용자가 태스크마다 둘 다 줬고
    (Bring Box 2.0~3.0), 그래서 "태스크마다 1.0 이 강제로 나오는" 문제가 없었다.
    시뮬은 사다리에서 뽑는다 -- 손상이 없는 가장 높은 배속이 하한이다.
    """
    # 키 이름이 두 갈래다 -- 상한표는 결과 디렉터리에서 와서 `libero_spatial/2`
    # 이고, 라벨은 지시문 대조표에서 와서 `spatial/2` 다. 한쪽만 보면 전부
    # 기본 띠로 떨어지는데 **숫자가 그럴듯해서 티가 안 난다.** 둘 다 본다.
    # 이름이 세 갈래다. 상한표는 결과 디렉터리에서 와서 `libero_spatial/2` 이고,
    # 라벨은 지시문 대조표에서 와서 `spatial/2` 다. 게다가 long-horizon 수트는
    # 디렉터리가 `libero_10` 인데 대조표는 `long` 으로 부른다.
    # 한쪽만 보면 전부 기본 띠로 떨어지는데 **숫자가 그럴듯해서 티가 안 난다.**
    v = None
    if task:
        base = task.replace("libero_", "")
        alts = [task, f"libero_{base}", base]
        if base.startswith("long/"):
            alts.append("libero_10/" + base.split("/", 1)[1])
        elif base.startswith("10/"):
            alts.append("long/" + base.split("/", 1)[1])
        for k in alts:
            if k in table:
                v = table[k]
                break
    if v is None:
        MISS.add(task)
        return DEFAULT_BAND
    if isinstance(v, (list, tuple)):
        return float(v[0]), float(v[1])
    return 1.0, float(v)                 # 예전 형식: 상한만 있었다


def levels_between(lo, hi):
    """띠 안의 눈금 칸들. [1.5, 2.0] 이면 (1.5, 2.0), [2.0, 2.0] 이면 (2.0,)."""
    return tuple(g for g in GRID if lo - 1e-9 <= g <= hi + 1e-9) or (float(lo),)


def assign_global(confs, his):
    """**전체를 한 줄로** 세워 눈금에 배분하고, 각자의 상한으로 자른다.

    태스크 안에서 등분하면 위험한 순간이 하나도 없는 태스크에서도 하위 몫이
    1.0 을 받고, 전부 위험한 태스크에서도 상위 몫이 상한을 받는다. 태스크 사이
    차이는 상한표가 이미 담당하므로 안에서 또 등분하는 것은 두 번 다루는 것이다.

    전체 순서로 앉히면 그 강제가 없어진다. 절대값을 그대로 쓰지 않는 이유는
    실측 분포가 좁기 때문이다 -- conf 가 0.205~0.733 에 중앙값 0.441 이라
    선형으로 옮기면 거의 다 한 칸에 뭉친다.
    """
    c = np.asarray(confs, dtype=float)
    h = np.asarray(his, dtype=float)
    n = c.size
    if n == 0:
        return c
    order = c.argsort(kind="mergesort")
    rank = np.empty(n, dtype=float)
    rank[order] = np.arange(n, dtype=float)
    for v in np.unique(c):
        m = c == v
        rank[m] = rank[m].mean()
    q = rank / max(1, n - 1)
    g = np.asarray(GRID, dtype=float)
    k = np.minimum((q * len(g)).astype(int), len(g) - 1)
    return np.minimum(g[k], h)          # 상한으로 자른다


def assign_levels(confs, lo, hi):
    """**칸 수만큼 등분한다.** 상한 2.5 면 4분위, 2.0 이면 3분위, 1.5 면 2분위.

    `band_place` 로 [1.0, hi] 에 균등하게 편 뒤 가까운 눈금으로 스냅하면 양끝 칸이
    절반만 받는다(2.5 상한에서 17/33/33/17). 스냅의 부작용이지 의도가 아니다.
    쓸 수 있는 칸 수만큼 등분하면 25/25/25/25 가 된다.

    같은 confidence 는 같은 칸을 받는다 -- 답이 안 갈렸는데 배속을 갈라 놓지 않는다.
    그래서 실제 비율은 동점 덩어리 크기만큼 등분에서 벗어난다.
    """
    lv = levels_between(lo, hi)
    c = np.asarray(confs, dtype=float)
    n = c.size
    if n == 0 or len(lv) == 1:
        return np.full(n, lv[0], dtype=float)
    # 동점은 평균 순위를 받아 같은 칸으로 묶인다
    order = c.argsort(kind="mergesort")
    rank = np.empty(n, dtype=float)
    rank[order] = np.arange(n, dtype=float)
    for v in np.unique(c):
        m = c == v
        rank[m] = rank[m].mean()
    q = rank / max(1, n - 1)                       # 0~1
    k = np.minimum((q * len(lv)).astype(int), len(lv) - 1)
    return np.asarray(lv, dtype=float)[k]


DESC = {"libero": "libero_descriptors", "robocasa": "robocasa_descriptors"}


def fill_contact(bench, rows):
    """라벨 줄에 접촉 정보가 없으면 **액션에서 다시 계산해 채운다.**

    phase9 는 등급 다섯 칸만 적었다. 그대로 두면 접촉이 하나도 안 걸려서
    (robocasa 204만 행에서 0.0%) 놓기·집기 순간이 안 늦춰진다. 계산으로 끝나는
    것이라 VLM 을 다시 부를 필요가 없다 -- 액션 청크만 읽으면 된다.
    """
    if all(("grip_transition" in r) for r in rows):
        return 0
    import numpy as _np
    import pandas as _pd
    mod = __import__(DESC[bench])
    info = json.load(open(f"{DS[bench]}/meta/info.json"))
    cs = info["chunks_size"]
    cache, n = {}, 0
    for r in sorted(rows, key=lambda r: (r["ep"], r["f"])):
        ep = int(r["ep"])
        if ep not in cache:
            cache.clear()
            try:
                cache[ep] = _np.stack(_pd.read_parquet(
                    f"{DS[bench]}/data/chunk-{ep // cs:03d}/"
                    f"episode_{ep:06d}.parquet")["action"].values)
            except Exception:
                cache[ep] = None
        a = cache[ep]
        if a is None or int(r["f"]) >= len(a) - 4:
            continue
        r.update(mod.computed_risk(mod.descriptors(a, int(r["f"]))))
        n += 1
    return n


# 실제 접촉 라벨. 있으면 이것을 쓴다 -- 액션에서 짐작한 것보다 낫다.
#   robocasa  https://huggingface.co/datasets/TTaekwan/robocasa_contact
#   libero    https://huggingface.co/datasets/TTaekwan/libero_contact
# `flevel_v1/sidecar/episode_XXXXXX.parquet` 의 `flevel` 열이 프레임마다
# [p_start, p_end, p_bnd, level, valid] 다. F_level 의 가드와 같은 식을 쓴다:
# **p_start != p_end 이거나 p_bnd 면 접촉**이고 그 창은 1.0배로 간다.
CONTACT_DIR = os.environ.get("CONTACT_DIR", "")


def load_contact(rows):
    """`CONTACT_DIR/episode_XXXXXX.parquet` 에서 가드를 읽어 라벨 줄에 채운다."""
    if not CONTACT_DIR or not os.path.isdir(CONTACT_DIR):
        return 0
    import numpy as _np
    import pandas as _pd
    n, cache = 0, {}
    for r in sorted(rows, key=lambda r: (r["ep"], r["f"])):
        ep = int(r["ep"])
        if ep not in cache:
            cache.clear()
            f = os.path.join(CONTACT_DIR, f"episode_{ep:06d}.parquet")
            try:
                cache[ep] = _np.stack(_pd.read_parquet(f)["flevel"].values)
            except Exception:
                cache[ep] = None
        v = cache[ep]
        i = int(r["f"])
        if v is None or i >= len(v):
            continue
        p_start, p_end, p_bnd, _lv, valid = (float(x) for x in v[i][:5])
        if valid < 0.5:
            continue
        r["guard"] = float((p_start != p_end) or p_bnd > 0.5)
        n += 1
    return n


def contact(rec):
    """이 시점이 접촉인가.

    실제 접촉 라벨(`guard`)이 있으면 그것만 쓴다. 없을 때만 액션에서 짐작한다.

    `libero_descriptors.computed_risk` 의 `grip_transition` 이 그리퍼가 이 창에서
    열리거나 닫힌다는 뜻이고, 그것이 곧 무게가 옮겨 가는 순간이다. 여기서
    몇 밀리미터가 잡혔는지 떨어졌는지를 가른다.

    `precise_hold`(들고 기어가는 중)도 접촉으로 센다 -- 쥔 채로 느리게 가는 것은
    놓을 자리를 맞추는 중이다.
    """
    if "guard" in rec:
        return bool(rec["guard"] > 0.5)
    return bool(rec.get("grip_transition", 0) > 0
                or rec.get("precise_hold", 0) > 0.5)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    bench, src = sys.argv[1], sys.argv[2]
    global RULE
    RULE = os.environ.get("RATIO_RULE", "levels")
    out = sys.argv[3] if len(sys.argv) > 3 else src.replace(".jsonl", "_ratio.jsonl")
    SIGN, WEIGHT, NGRADE, ceil, ceil_path = load_bench(bench)
    print(f"[{bench}]  감점 " + " ".join(
        f"{k}={WEIGHT[k]:.3f}" for k in SLOTS if SIGN[k] < 0)
        + "  가점 " + " ".join(f"{k}={WEIGHT[k]:.3f}" for k in SLOTS if SIGN[k] > 0))
    rows = [json.loads(l) for l in open(src) if l.strip()]
    rows = [r for r in rows if all(k in r for k in SLOTS)]
    print(f"라벨 {len(rows)}행  <- {src}")

    if ceil:
        print(f"상한표 {len(ceil)}태스크  <- {os.path.normpath(ceil_path)}")
    else:
        print(f"[!] 상한표가 없다 ({os.path.normpath(ceil_path)}). 모든 태스크에 "
              f"기본 띠 {DEFAULT_BAND} 를 쓴다 -- 사다리가 차면 "
              f"derive_libero_ceilings.py 로 만든다.")

    ep2t = ep_to_task(bench)

    # 신뢰도는 **태스크 안에서** 분위수로 앉힌다. 태스크를 섞으면 어려운
    # 태스크의 쉬운 순간이 쉬운 태스크의 어려운 순간보다 높은 배속을 받는다.
    by_task = defaultdict(list)
    for i, r in enumerate(rows):
        by_task[ep2t.get(r["ep"])].append(i)

    ng = load_contact(rows)
    if ng:
        print(f"실제 접촉 라벨을 읽었다: {ng:,}행  <- {CONTACT_DIR}")
    nfill = 0 if ng else fill_contact(bench, rows)
    if nfill:
        print(f"접촉 정보를 액션에서 다시 계산했다: {nfill:,}행 "
              f"(라벨 파일에 없었다 -- VLM 은 안 쓴다)")

    ratio = [None] * len(rows)
    nfix = 0
    if RULE == "global":
        free = [i for i, r in enumerate(rows) if not contact(r)]
        for i, r in enumerate(rows):
            if i not in set(free):
                ratio[i] = 1.0
                nfix += 1
        if free:
            hs = []
            for i in free:
                hs.append(band_of(ceil, ep2t.get(rows[i]["ep"]))[1])
            vals = assign_global(
                [confidence(rows[i], SIGN, WEIGHT, NGRADE) for i in free], hs)
            for i, v in zip(free, vals):
                ratio[i] = float(v)
        by_task = {}
    for tk, idx in by_task.items():
        # 상한표는 값 하나(상한)를 준다. 아래끝은 언제나 1.0 이다.
        lo, hi = band_of(ceil, tk)
        free = [i for i in idx if not contact(rows[i])]
        for i in idx:
            if i not in free:
                ratio[i] = 1.0
                nfix += 1
        if not free:
            continue
        cf = [confidence(rows[i], SIGN, WEIGHT, NGRADE) for i in free]
        if RULE == "levels":
            vals = assign_levels(cf, lo, hi)
        elif RULE == "band":
            vals = [snap(v) for v in band_place(cf, lo, hi)]
        elif RULE == "mid":
            # conf 0.5 는 가점 가중합 = 감점 가중합 인 자리다. 지어낸 선이 아니라
            # 수식이 주는 자리 -- 그 위는 상한, 아래는 1.0.
            vals = [hi if v > 0.5 else lo for v in cf]
        else:
            raise SystemExit(f"모르는 규칙 {RULE}. levels · band · mid 중 하나")
        for i, v in zip(free, vals):
            ratio[i] = float(v)

    with open(out, "w") as f:
        for r, v in zip(rows, ratio):
            f.write(json.dumps({**r, "ratio": v,
                                "task": ep2t.get(r["ep"]),
                                "conf": round(confidence(r, SIGN, WEIGHT, NGRADE), 4)}) + "\n")

    # 배포용 parquet 도 같이 낸다. `apply_ratio_labels.py` 가 읽는 형식이고,
    # 다른 기계에서는 이것 하나만 받으면 된다 -- 등급도 jsonl 도 필요 없다.
    try:
        import pandas as pd
        pq = out.replace(".jsonl", ".parquet")
        # **등급을 같이 담는다.** 비싼 것은 VLM 이 매기는 등급뿐이고 그 뒤는
        # 전부 산수다. 등급이 있으면 띠나 규칙을 바꿔 CPU 만으로 몇 초 만에
        # 다시 라벨링할 수 있다 -- VLM 을 다시 돌릴 필요가 없다.
        cols = {
            "episode_index": [int(r["ep"]) for r in rows],
            "frame_index": [int(r["f"]) for r in rows],
            "task": [ep2t.get(r["ep"]) for r in rows],
            "ratio": np.asarray(ratio, dtype=np.float32),
            "conf": np.array([confidence(r, SIGN, WEIGHT, NGRADE) for r in rows], dtype=np.float32),
            "fixed": np.array([1 if contact(r) else 0 for r in rows], dtype=np.int8),
        }
        for k in SLOTS:                              # 등급 A~E
            cols[k] = np.array([int(r[k]) for r in rows], dtype=np.int8)
        for k in ("grip_transition", "precise_hold"):   # 접촉 판정의 입력
            cols[k] = np.array([float(r.get(k, 0.0)) for r in rows], dtype=np.float32)
        pd.DataFrame(cols).sort_values(
            ["episode_index", "frame_index"]).to_parquet(pq, index=False)
        print(f"-> {pq}  (배포용. docs/RATIO_LABELS.md 참고)")
    except ImportError:
        print("[!] pandas 가 없어 parquet 은 안 만들었다. jsonl 은 나왔다.")

    if MISS:
        print(f"\n[!] 상한표에서 못 찾은 태스크 {len(MISS)}개 -- 기본 띠 "
              f"{DEFAULT_BAND} 로 갔다. 키 이름을 맞춰야 한다:")
        for t in sorted(MISS)[:8]:
            print(f"      {t}")

    import collections
    c = collections.Counter(ratio)
    n = len(rows) or 1
    print(f"\n접촉으로 1.0배 고정  {nfix}/{n} = {nfix / n:.1%}")
    print("배속 분포  " + "  ".join(f"{g}x:{c.get(g, 0) / n:5.1%}" for g in GRID))
    print(f"평균 배속  {np.mean([v for v in ratio if v]):.3f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
