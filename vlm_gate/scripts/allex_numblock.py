"""allex 액션 콜에 넣을 숫자 블록 — 양팔 관절 + 손 평균 + 양 손목 위치."""
import numpy as np, json
from allex_common import RA, LA, RH, LH, NUM_HEADER
def build(action, wr, wl, f, n=16):
    """action:(T,48), wr/wl:(T,9) 손목 pose. -> 헤더 포함 문자열"""
    w=slice(f, f+n)
    cols=[action[w,RA], action[w,LA],
          action[w,RH].mean(1)[:,None], action[w,LH].mean(1)[:,None],
          wr[w,:3], wl[w,:3]]
    M=np.column_stack(cols)
    return f"{NUM_HEADER}\n{json.dumps(np.round(M,3).tolist())}"
