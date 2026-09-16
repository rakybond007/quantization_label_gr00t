"""libero v3d 계산 사실 = v3c 와 같고 **속도 줄만 백분위로, raw 숫자는 뺐다.**

libero 는 절대 임계 두 개(`SLOW_POS`=0.2, 0.7)로 속도를 세 구간으로 나눈다. 실측
분포는 **3.6% / 50.1% / 46.3%** 다 -- "barely moving" 은 사실상 안 쓰이고 나머지가
반반이라 세 구간이 두 구간처럼 동작한다. robocasa 가 같은 이유로 백분위로 갈아탄
자리이고(`robocasa_descriptors.py` 의 주석), libero 는 그 업그레이드를 못 받았다.

표: `configs/libero_speed_percentiles.json` (libero_gr00t_delta 120 에피 · stride 16
· 1,102 표본 · 101 경계). 표가 없으면 **죽는다** -- 조용히 옛 문구로 강등되면
어떤 문구로 만든 라벨인지 알 수 없게 된다.

`(mean step 0.67, peak 0.75)` 도 뺐다. 단위가 없어 0.67 이 무엇인지 알 수 없고,
백분위 문구가 이미 같은 뜻을 전달한다(`prompts/FORMAT.md` §3).
"""
import json
import os
import re

import numpy as np

from libero_descriptors import *  # noqa: F401,F403
from libero_descriptors import facts_text as _v3c_facts

_PCT_PATH = (f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
             f"/configs/libero_speed_percentiles.json")
_PCT = np.asarray(json.load(open(_PCT_PATH))["percentiles"], dtype=float)
_OLD = re.compile(r"it is (barely moving|moving at a normal pace|moving fast)"
                  r" \(mean step (-?\d+\.\d+), peak -?\d+\.\d+\)")


def _band(pc):
    return ("among the slowest motions in this dataset" if pc < 20 else
            "on the slow side for this dataset" if pc < 40 else
            "middling in speed for this dataset" if pc < 60 else
            "on the fast side for this dataset" if pc < 80 else
            "among the fastest motions in this dataset")


def facts_text(x):
    t = _v3c_facts(x)
    m = _OLD.search(t)
    if m is None:
        raise AssertionError("속도 줄 문구가 바뀌었다. 백분위로 못 바꾸면 "
                             "뭉개진 절대 임계 문구가 그대로 나간다: " + t[:200])
    pc = float(np.clip(np.searchsorted(_PCT, float(m.group(2)), side="right") - 1, 0, 100))
    return t[:m.start()] + (f"it is {_band(pc)} -- faster than {pc:.0f}% of all "
                            f"moments here") + t[m.end():]
