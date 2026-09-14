"""새 allex(택배 분류) 데이터셋의 국면 유도. robocasa `judge_ab.phase_of` 자리.

**손가락으로 국면을 가를 수 없다.** 이 로봇은 편 손으로 소포를 감싸거나 밀어
옮긴다 -- 12에피 25,450프레임에서 오른손 굽힘 p90 이 0.079, 왼손이 0.232 로
사실상 움직이지 않는다. robocasa 의 그리퍼 개폐, libero 의 그리퍼 차원에 해당하는
신호가 여기엔 없다.

대신 **두 손바닥 사이에 물건이 있는가**(`held`)로 가른다. 이건 이 데이터셋의
기존 기술자가 이미 쓰는 정의이고(간격 < 0.42 이면서 한 팔만 쓰는 게 아닐 때),
위험 모델 전체가 "그 두 손 사이 유지를 잃는 것" 을 다룬다.

    접근    아직 안 쥠, 손바닥이 모이는 중
    감싸기  이 창에서 안 쥔 상태 -> 쥔 상태로 바뀐다   <- 가장 위험한 자리
    정렬    쥔 채로 손목이 크게 돈다 (바코드를 위로)
    운반    쥔 채로 옮긴다
    해제    이 창에서 쥔 상태 -> 안 쥔 상태로 바뀐다
    빈손    아무것도 안 쥐고 이동하거나 서 있다
"""
import os

import numpy as np

ARM = slice(0, 14)
RA, LA = slice(0, 7), slice(7, 14)
WIN = int(os.environ.get("ALLEX_WIN", "16"))
GAP_HELD = float(os.environ.get("ALLEX_GAP_HELD", "0.42"))
ROT_REORIENT = float(os.environ.get("ALLEX_ROT_REORIENT", "31.0"))  # 표본 p75
FINE = ("접근", "감싸기", "정렬", "운반", "해제", "빈손")


def _rot6_to_R(v):
    a, b = v[..., :3], v[..., 3:6]
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b - (a * b).sum(-1, keepdims=True) * a
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return np.stack([a, b, np.cross(a, b)], axis=-2)


def _ang(R1, R0):
    c = np.clip((np.trace(R1 @ R0.T) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))


def episode_phases(A, WR, WL, win=WIN):
    """프레임마다 국면 이름. A(n,48) · WR/WL(n,9)."""
    n = len(A)
    gap = np.linalg.norm(WR[:, :3] - WL[:, :3], axis=1)
    out = []
    for f in range(n):
        w = slice(f, min(f + win, n))
        Aw = A[w]
        if len(Aw) < 2:
            out.append(out[-1] if out else "빈손"); continue
        r = float(np.linalg.norm(np.diff(Aw[:, RA], axis=0), axis=1).sum())
        l = float(np.linalg.norm(np.diff(Aw[:, LA], axis=0), axis=1).sum())
        tot = r + l
        one_h = tot > 1e-6 and min(r, l) / tot < 0.25
        g = gap[w]
        held_now = bool(g[:max(1, len(g) // 4)].mean() < GAP_HELD and not one_h)
        held_end = bool(g[-max(1, len(g) // 4):].mean() < GAP_HELD and not one_h)
        Rr = _rot6_to_R(WR[w, 3:9]); Rl = _rot6_to_R(WL[w, 3:9])
        rot = max(_ang(Rr[-1], Rr[0]), _ang(Rl[-1], Rl[0]))
        closing = bool(g[-1] < g[0] - 0.01)
        if not held_now and held_end:
            p = "감싸기"
        elif held_now and not held_end:
            p = "해제"
        elif held_now:
            p = "정렬" if rot >= ROT_REORIENT else "운반"
        elif closing:
            p = "접근"
        else:
            p = "빈손"
        out.append(p)
    return out


def load_episode(ds, ep, chunks_size=1000):
    import pandas as pd
    ch = ep // chunks_size
    d = pd.read_parquet(f"{ds}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet",
                        columns=["action", "action.right_wrist_wrt_base",
                                 "action.left_wrist_wrt_base"])
    return (np.stack(d["action"].values),
            np.stack(d["action.right_wrist_wrt_base"].values),
            np.stack(d["action.left_wrist_wrt_base"].values))
