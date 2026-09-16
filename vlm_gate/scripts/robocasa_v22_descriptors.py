"""robocasa v22 계산 사실 = v8 과 같고 **단위 없는 raw 숫자만 뺐다.**

v7/v8 은 속도 줄에 `(mean step 0.41, peak 0.52)` 를 붙인다. 단위가 없어서 0.41 이
무엇인지 알 수 없고, 앞의 백분위 문구("faster than 34% of all moments here")가 이미
같은 뜻을 다 전달한다. `prompts/FORMAT.md` §3 이 "한계값이 아니라 분포 구간으로"
라고 정한 자리이고, 척도 없는 숫자는 판정기가 없는 척도를 발명하게 만든다.

**머리글 교체를 검증한다.** v8 은 v7 문장의 "over the next ~1 second" 를 문자열
replace 로 고쳤다. 원본 문구가 바뀌면 replace 가 조용히 아무것도 안 하고 거짓
1초 주장이 그대로 나간다(창 16스텝은 20fps 에서 0.80초다). 여기서는 둘 중 하나가
반드시 있어야 하고 아니면 죽는다.
"""
import re

from robocasa_descriptors import *  # noqa: F401,F403
from robocasa_descriptors import facts_text as _v7_facts

_OLD = "over the next ~1 second"
_NEW = "over the chunk ahead"
_NUM = re.compile(r" \(mean step -?\d+\.\d+, peak -?\d+\.\d+\)")


def facts_text(x):
    t = _v7_facts(x)
    if _OLD not in t and _NEW not in t:
        raise AssertionError(
            "계산 사실 머리글이 바뀌었다. 기간 주장을 지우지 못하면 거짓이 나간다: "
            + t[:120])
    t = t.replace(_OLD, _NEW)
    t, n = _NUM.subn("", t)
    if n != 1:
        raise AssertionError(f"속도 raw 숫자 제거 실패 (제거 {n}개): {t[:200]}")
    return t
