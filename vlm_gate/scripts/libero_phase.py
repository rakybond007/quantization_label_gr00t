"""LIBERO 국면 정의. 계산만으로, 문항의 답을 미리 알려주지 않는 값만 쓴다.

robocasa 의 `judge_ab.phase_of` 와 같은 구조인데 축이 다르다 -- LIBERO 액션은 7차원
(0-2 위치 델타 · 3-5 회전 · 6 그리퍼)이고 robocasa 는 12차원(5-7 위치)이다. 그리고
`libero_descriptors` 가 `vert` 를 연속값으로, `down`/`up` 을 나눠서 내므로 "내려가며
다가감" 과 "들어올림" 을 robocasa 보다 잘 가른다.

그리퍼 전이는 방향까지 가른다 -- robocasa 에서 "opens or closes" 로 뭉쳐 두었더니
모델의 답이 파지와 해제에서 사실상 같았고(차이 -0.003, p=0.86), 방향을 주자
-0.121 (p=0.0009) 로 갈렸다. 같은 실수를 반복하지 않는다.
"""
import numpy as np

PHASES = ("접근", "파지", "해제", "전이(둘다)", "운반", "들어올림", "내려놓기")


def phase_of(a, f, n=16):
    """a:(T,7) · f 시작 프레임 -> 국면 이름."""
    w = np.asarray(a[f:f + n], dtype=np.float64)
    if len(w) < 2:
        return "접근"
    g = w[:, 6]
    dg = np.diff(g)
    cl = bool((dg > 0.5).any())      # 열림(-1) -> 닫힘(+1)
    op = bool((dg < -0.5).any())
    if cl and op:
        return "전이(둘다)"
    if cl:
        return "파지"
    if op:
        return "해제"
    pos = w[:, 0:3]
    closed = float((g > 0).mean()) > 0.5
    dz = float(pos[:, 2].sum())
    span = float(np.linalg.norm(pos.sum(axis=0))) + 1e-9
    vert = dz / span
    if vert < -0.55:
        return "내려놓기" if closed else "접근"
    if vert > 0.55:
        return "들어올림"
    return "운반" if closed else "접근"


def gripper_fact(a, f, n=16):
    """계산 사실 첫 줄. 방향과 시점을 준다 (robocasa 에서 검증된 형태)."""
    w = np.asarray(a[f:f + n], dtype=np.float64)
    if len(w) < 2:
        return "the gripper stays open throughout"
    g = w[:, 6]
    dg = np.diff(g)
    cl = np.nonzero(dg > 0.5)[0]
    op = np.nonzero(dg < -0.5)[0]
    idx = list(cl) + list(op)
    when = f" {int(min(idx))} of {n} steps in" if idx else ""
    if len(cl) and len(op):
        return f"the gripper closes and then opens again{when}"
    if len(cl):
        return (f"the gripper CLOSES on something{when} -- the grasp is made "
                f"inside this window")
    if len(op):
        return (f"the gripper OPENS to let go{when} -- the release happens "
                f"inside this window")
    return ("the gripper stays closed throughout" if float((g > 0).mean()) > 0.5
            else "the gripper stays open throughout")
