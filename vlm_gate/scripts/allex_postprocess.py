"""게이트 결함 교정 — 셋은 성격이 달라 서로 다른 층에 속한다.

  라벨 층 (label_risk): 하드 차단과 누적 회전. 계획된 청크와 직전 청크들만으로
    계산되므로 라벨링 시점에 이미 알 수 있다. 학생이 이것까지 배우게 한다.
  런타임 층 (runtime_gate): 히스테리시스와 인과적 평활. 이전 *판정*이 필요해
    프레임 독립인 라벨에는 정의상 넣을 수 없다. 라벨에 구워넣으면 학생 출력이
    이미 평활된 상태가 되어 추론 시 이중 적용된다.


측정된 결함 (ep0, 165청크):
  1) 판정이 36% 확률로 매 청크 뒤집힘. 연속 구간 중앙 2청크, 33%가 1청크짜리.
     조작 국면이 0.53초마다 바뀌지 않으므로 노이즈다.
  2) 회전 28.9도·병합요구 0.325(한계의 2배)인데 rank 0.53 -> 통과. 위험 신호가
     포화되어 상위가 뭉치면 순위정규화가 경계로 밀어낸다.
  3) 청크당 10도 이하지만 여러 청크에 걸쳐 도는 '느린 회전' 24개를 놓침
     (회전을 청크 경계에서만 재기 때문).

셋 다 과거만 참조한다 — 런타임에 그대로 쓸 수 있다.
"""
import numpy as np

# 속도 그 자체는 차단 사유가 아니다. 시연은 사람 원격조작이라 그 최대치는
# 사람이 얼마나 빨리 움직였는지를 반영할 뿐이고, 이 데모는 최대 4배 가속까지
# 봐야 한다. 예전 merge_demand 하드 차단은 165청크 중 34개(20.6%)를 오로지
# '시연보다 빠르다'는 이유만으로 잘랐다 — 제거한다.
#
# 남기는 차단 사유는 결과가 달라지는 것들뿐이다:
#   - 파지 중 누적 회전: 두 손이 서로 다르게 움직여야 물체가 돌고, 그 상대 운동이
#     곧 파지력이다. 중간 목표를 건너뛰면 손바닥이 스텝당 더 멀리 이동한다.
#   - 파지 중 손바닥 간격 변화율: K2는 이 값을 2배로 만든다. 간격이 벌어지는
#     방향으로 빠르게 어긋나면 마찰 파지가 풀린다.
GAP_RATE_LIMIT = 0.010     # m/step. K2 적용 시 20 mm/step 이상이 되는 구간
ROT_ACCUM_LIMIT = 18.0     # deg, 직전 3청크 누적

def hard_block(x, rot_accum):
    """순위와 무관하게 차단할 조건. 속도가 아니라 파지 유지 가능성으로만 판단한다."""
    if not x.get("held"):
        return False                                  # 물체를 안 들었으면 잃을 것이 없다
    if rot_accum > ROT_ACCUM_LIMIT:
        return True                                   # 든 채로 방향을 바꾸는 중
    if x.get("gap_rate", 0.0) * 2 > GAP_RATE_LIMIT * 2:
        return x.get("gap_rate", 0.0) > GAP_RATE_LIMIT
    return False

def smooth_causal(conf, k=3):
    """인과적 이동 중앙값. 미래를 보지 않고 직전 k-1개만 사용한다."""
    out = np.empty_like(conf)
    for i in range(len(conf)):
        out[i] = np.median(conf[max(0, i - k + 1): i + 1])
    return out

def runtime_gate(descs, conf, tau=0.5, hyst=0.08, k=3, rot_window=3):
    """descs: 청크별 기술자 dict 리스트, conf: 순위정규화된 confidence.
    반환: 압축 여부 배열."""
    n = len(conf)
    # 느린 회전: 직전 rot_window 청크의 회전각 누적 (과거만)
    wr = np.array([d.get("wrist_rot", 0.0) for d in descs])
    accum = np.array([wr[max(0, i - rot_window + 1): i + 1].sum() for i in range(n)])
    sm = smooth_causal(np.asarray(conf, float), k)
    out = np.zeros(n, dtype=bool)
    on = False                                      # 히스테리시스 상태
    for i in range(n):
        if hard_block(descs[i], accum[i]):
            on = False; out[i] = False; continue
        # 압축 중이면 낮은 문턱까지 유지, 아니면 높은 문턱을 넘어야 진입
        thr = tau - hyst if on else tau + hyst
        on = sm[i] >= thr
        out[i] = on
    return out


def label_risk(descs, rot_window=3):
    """라벨 층: 청크별 하드 차단 위험도(0/1). 학생이 배울 대상에 포함시킨다."""
    wr = np.array([d.get("wrist_rot", 0.0) for d in descs])
    accum = np.array([wr[max(0, i - rot_window + 1): i + 1].sum() for i in range(len(descs))])
    return np.array([1.0 if hard_block(d, a) else 0.0 for d, a in zip(descs, accum)])
