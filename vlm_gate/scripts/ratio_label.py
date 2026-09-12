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


def expected_grades(probs, ngrade=5):
    """등급 분포에서 기댓값. 뽑힌 숫자 하나보다 정보가 많다.

    `P(3)=0.9` 와 `P(2)=.3/P(3)=.35/P(4)=.3` 은 전혀 다른 상태인데 등급을
    정수로만 받으면 둘이 같아진다. 모델은 여전히 **텍스트로 답하고**, 이건 그
    답이 얼마나 확실했는지를 덧붙이는 것이다 -- 강제된 슬롯의 로짓을 답으로
    삼는 옛 방식과 다르다. `allex_v3_checks.expected_grades` 와 같은 규칙이다.
    """
    if not probs:
        return None
    out = []
    for row in probs:
        if not row or len(row) != ngrade:
            return None
        out.append(sum((i + 1) * float(q) for i, q in enumerate(row)))
    return out


def confidence(rec, sign, weight, ngrade=5):
    """(1 + Σw·가점 − Σw·감점) / 2. 등급은 (g−1)/(n−1) 로 0~1 에 옮긴다.

    등급 분포(`gp`)가 있으면 정수 등급 대신 **기댓값**을 쓴다.
    """
    eg = expected_grades(rec.get("gp"), ngrade)
    vals = ({k: v for k, v in zip(SLOTS, eg)} if eg and len(eg) == len(SLOTS)
            else rec)
    d = float(ngrade - 1)
    risk = sum(weight[k] * (float(vals[k]) - 1) / d for k in SLOTS if sign[k] < 0)
    safe = sum(weight[k] * (float(vals[k]) - 1) / d for k in SLOTS if sign[k] > 0)
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


def ep_to_instruction(bench):
    """에피소드 -> **지시문 원문**. VLA 가 실제로 받는 문장이다.

    `episodes.jsonl` 의 `tasks` 에서 공백이 있는 첫 문자열이 지시문이다
    (`["turn on the front right burner of the stove", "TurnOnStove", "Valid"]`).
    태스크 이름은 공백이 없어 저절로 갈린다.

    태스크 이름과 따로 넣는 이유: 이 라벨을 받는 머리는 정책 위에 얹히고,
    정책이 받는 것은 `TurnOnStove` 라는 분류명이 아니라 이 문장이다. 한
    태스크 안에도 "front right burner" 와 "front left burner" 가 따로 있고,
    그것이 배속에 무관하다고 볼 근거가 없다. 라벨 파일이 문장을 안 들고
    있으면 받는 쪽이 매번 원본 데이터셋을 되짚어야 하는데, 배포본만 받은
    사람은 그 길이 없다.
    """
    out = {}
    for line in open(f"{DS[bench]}/meta/episodes.jsonl"):
        d = json.loads(line)
        for v in (d.get("tasks") or []):
            if isinstance(v, str) and " " in v.strip():
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


def conf_of(rec, sign, weight, ngrade):
    """신뢰도. **접촉이 잡히면 0 이다.**

    가드가 배속만 1.0 으로 박고 신뢰도를 안 건드리면, `conf` 로만 자르는 쪽에서
    가드가 **적혀만 있고 작동하지 않는다.** 접촉 구간의 `conf` 가 0.6 으로
    남아 있으면 역치를 넘어 압축된다. 두 판의 평균 신뢰도가 0.4267 로 똑같았던
    것이 그 증상이다 -- 가드를 걸었는데 신뢰도 분포가 하나도 안 움직였다.

    **등급 A~E 는 안 건드린다.** 비싼 것은 등급이고, 받는 쪽이 규칙을 바꿔 다시
    계산할 수 있어야 한다. `fixed` 열이 어느 행인지 표시하므로 되짚을 수 있다.

    이것은 **라벨 파일의 이야기다.** 평가 도중 VLM 이 그 자리에서 내는 신뢰도는
    이 함수를 안 거치므로, 온라인 게이트의 가드는 액션에서 경계를 따로 계산해야
    한다. 두 자리를 헷갈리지 말 것.
    """
    if contact(rec):
        return 0.0
    return confidence(rec, sign, weight, ngrade)


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
    ep2i = ep_to_instruction(bench)

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

    ngp = sum(1 for r in rows if r.get("gp"))
    print(f"등급 분포(gp) 있는 행 {ngp:,}/{len(rows):,} = {ngp/max(1,len(rows)):.1%}")
    # **경고로 두지 않는다.** 이 줄은 전부터 0% 를 찍고 있었는데 출력일 뿐이라
    # 그냥 지나갔고, 라벨 2.04M 행이 정수 등급으로 만들어졌다. 그 대가는 작지
    # 않다 -- 등급 하나당 |정수 - 기댓값| 이 평균 0.550 칸이고, 게이트 tau=0.5
    # 에서 판정 23.8% 가 뒤집힌다. 게다가 정수 conf 는 값이 몇 개뿐이라
    # tau=0.50/0.517/0.55 가 같은 설정이 된다(압축률 전부 32.9%).
    # 옛 라벨 파일을 일부러 처리할 때만 ALLOW_INT_GRADES=1 로 넘긴다.
    if not ngp and os.environ.get("ALLOW_INT_GRADES", "") not in ("1", "true"):
        raise SystemExit(
            "등급 분포(gp)가 한 행도 없다. 라벨러가 grade_probs 를 적지 않았다 -- "
            "정수 등급으로 떨어지면 conf 가 계단이 되고 역치가 작동하지 않는다. "
            "라벨러를 고치고 다시 만들거나, 옛 파일을 일부러 쓸 때는 "
            "ALLOW_INT_GRADES=1 을 준다.")

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
                                "conf": round(conf_of(r, SIGN, WEIGHT, NGRADE), 4)}) + "\n")

    # 배포용 parquet 도 같이 낸다. `apply_ratio_labels.py` 가 읽는 형식이고,
    # 다른 기계에서는 이것 하나만 받으면 된다 -- 등급도 jsonl 도 필요 없다.
    try:
        pq = out.replace(".jsonl", ".parquet")
        # **등급을 같이 담는다.** 비싼 것은 VLM 이 매기는 등급뿐이고 그 뒤는
        # 전부 산수다. 등급이 있으면 띠나 규칙을 바꿔 CPU 만으로 몇 초 만에
        # 다시 라벨링할 수 있다 -- VLM 을 다시 돌릴 필요가 없다.
        #
        # **한 DataFrame 으로 올리지 않는다.** 204만 행 x 14열을 통째로 만들면
        # 이미 메모리에 있는 원본 행과 겹쳐 로그인 노드에서 exit 137 로 죽는다
        # (robocasa 에서 두 번 같은 자리에서 죽었다). 덩이로 흘려 쓴다.
        #
        # **행마다 키 집합이 다르다.** `guard` 는 접촉이 걸린 행에만 붙는다.
        # 첫 줄로 스키마를 잡으면 열 길이가 어긋나므로 열을 미리 못 박는다.
        # `fixed` 는 접촉이 없는 판에도 0 으로 넣는다 -- 두 판을 나란히 놓는 것이
        # 목적인데 열이 갈리면 "0.00% 대 16.56%" 라는 대조를 아예 못 쓴다.
        import pyarrow as pa
        import pyarrow.parquet as pqw
        pq = out.replace(".jsonl", ".parquet")
        CH = 200_000
        # 덩이로 흘려 쓰므로 **미리 정렬한다.** 통째로 만들 때는 마지막에
        # sort_values 로 맞췄는데, 흘려 쓰면 그 자리가 없다. 붙이는 쪽이
        # (episode_index, frame_index) 로 찾으므로 순서가 서 있어야 빠르다.
        order = sorted(range(len(rows)),
                       key=lambda i: (int(rows[i]["ep"]), int(rows[i]["f"])))
        w, sch = None, None
        for b0 in range(0, len(order), CH):
            idxs = order[b0:b0 + CH]
            sl = [rows[i] for i in idxs]
            cols = {
                "episode_index": np.array([int(r["ep"]) for r in sl], dtype=np.int32),
                "frame_index": np.array([int(r["f"]) for r in sl], dtype=np.int32),
                "task": [ep2t.get(r["ep"]) for r in sl],
                "instruction": [ep2i.get(r["ep"]) for r in sl],
                "ratio": np.array([ratio[i] for i in idxs], dtype=np.float32),
                "conf": np.array([conf_of(r, SIGN, WEIGHT, NGRADE) for r in sl],
                                 dtype=np.float32),
                "fixed": np.array([1 if contact(r) else 0 for r in sl], dtype=np.int8),
            }
            for k in SLOTS:
                cols[k] = np.array([int(r[k]) for r in sl], dtype=np.int8)
            for k in ("grip_transition", "precise_hold", "guard"):
                cols[k] = np.array([float(r.get(k, 0.0)) for r in sl], dtype=np.float32)
            # 기대 등급. gp 가 없으면 정수 등급이 그대로 들어간다 -- 받는 쪽이
            # 신뢰도가 어디서 나왔는지 되짚을 수 있어야 한다.
            for i, k in enumerate(SLOTS):
                cols["eg_" + k] = np.array(
                    [(expected_grades(r.get("gp"), NGRADE) or [float(r[k])] * len(SLOTS))[i]
                     for r in sl], dtype=np.float32)
            t = pa.table(cols)
            if w is None:
                sch = t.schema
                w = pqw.ParquetWriter(pq, sch, compression="zstd")
            w.write_table(t.cast(sch))
        if w is not None:
            w.close()
        print(f"-> {pq}  (배포용. docs/RATIO_LABELS.md 참고)")
    except ImportError:
        print("[!] pyarrow 가 없어 parquet 은 안 만들었다. jsonl 은 나왔다 -- "
              "ratio_jsonl_to_parquet.py 로 따로 만들 수 있다.")

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
