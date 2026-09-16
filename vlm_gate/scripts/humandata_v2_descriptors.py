"""human_data v2 계산 사실 = v1 에서 **배속을 지목하는 두 문장을 빼고** 접속사를 고쳤다.

v1 의 마지막 두 문장은 이랬다:

    at 2x this stretch would move 0.068 rad in one step, smaller than most;
    at 3x 0.099 rad, smaller than most

**2배·3배를 우리가 정하지 않는다.** 묻는 것은 "이 구간을 빨리 지나가도 되느냐" 이고,
문장이 특정 배율을 지목하면 판정기에게 우리가 안 고른 정책을 준다. 새로 라벨을
만드는 중이므로 남길 이유가 없다.

접속사도 고쳤다. v1 은 그리퍼 줄을 두 조각으로 나눠 `"; "` 로 이어

    the hand stays open throughout; and its opening barely changes

가 되는데, 목록 항목이 `; and` 로 시작해 문법이 깨진다. `, and` 로 한 항목으로 붙인다.

속도 세 구간은 건드리지 않는다 -- `analysis/human_data/quantiles.json` 의
`speed_tertile` 에서 나오므로 데이터셋별로 33/33/33 이다.
"""
import re

from humandata_descriptors import *  # noqa: F401,F403
from humandata_descriptors import facts_text as _v1_facts

_JOINS = ("; and it is closing", "; and it is opening", "; and its opening barely changes")
_KX = re.compile(r"; at 2x this stretch would move .*?(?=\.$)", re.S)


def facts_text(x):
    t = _v1_facts(x)
    for j in _JOINS:
        if j in t:
            t = t.replace(j, "," + j[1:], 1)
            break
    else:
        raise AssertionError("그리퍼 줄 접속 문구를 못 찾았다 -- v1 문장이 바뀌었나: " + t[:180])
    t, n = _KX.subn("", t)
    if n != 1:
        raise AssertionError(f"배속 문장(at 2x/3x) 제거 실패 (제거 {n}개): {t[:220]}")
    return t
