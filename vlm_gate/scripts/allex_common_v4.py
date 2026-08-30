"""allex 질문 v4 — 계산 가능한 것은 계산해 '사실'로 진술하고, VLM에는 의미만 묻는다.

v1~v3의 실패 원인: 손목 간격·팔 속도처럼 정확히 계산되는 값을 VLM에게 '질문'해
눈대중시켰다. 실측 결과 VLM의 답과 직접 계산의 이진 일치도가 37%(E), 53%(F),
44%(G)로 무너졌고, AUC도 0.804 -> 0.619로 뭉개졌다.

v4는 같은 값을 질문이 아니라 전제로 넣는다. 로보카사에서 SCALE REFERENCE를
넣었을 때 그리퍼 검출이 36%->84%로 오른 것과 같은 장치를, 전역 통계가 아니라
청크별 계산값으로 확장한 것이다. VLM에게 남기는 것은 계산으로 알 수 없는 것뿐이다.
"""
import numpy as np, os
T=os.path.expanduser("~/quantization_agent_workspace/assets/datasets/allex_hires_v1")
RA=slice(0,7); LA=slice(7,14); ARM=slice(0,14); RH=slice(14,29); LH=slice(29,44)

# ---- 2층: 결정적 계산 -------------------------------------------------------
def descriptors(action, wr, wl, f, n=16):
    """계획된 청크에서 기술자를 정확히 계산한다. 반환값은 사람이 읽는 문장."""
    w=slice(f, min(f+n, len(action)))
    gap=np.linalg.norm(wr[w,:3]-wl[w,:3], axis=1)
    sp=np.linalg.norm(np.diff(action[w,ARM],axis=0),axis=1)
    hand=np.array([action[w,RH].mean(1), action[w,LH].mean(1)])
    merged=np.linalg.norm(action[w][2::2]-action[w][0:-2:2], axis=1)[:, ] if len(action[w])>2 else np.array([0.0])
    d={
      "gap_mean":float(gap.mean()), "gap_min":float(gap.min()),
      "gap_change":float(gap.max()-gap.min()),
      "closing":bool(gap[-1] < gap[0]-0.01), "opening":bool(gap[-1] > gap[0]+0.01),
      "arm_speed":float(sp.mean()) if len(sp) else 0.0,
      "arm_speed_max":float(sp.max()) if len(sp) else 0.0,
      "hand_change":float(np.abs(hand[:,-1]-hand[:,0]).max()),
      "merge_demand":float(np.max(merged)) if len(merged) else 0.0,
    }
    return d

def facts_text(d):
    """계산값을 문장으로. 임계값 판정까지 여기서 끝내고 모델에는 결론만 준다."""
    close = "pinched between the two hands" if d["gap_mean"]<0.42 else \
            ("within reach of each other" if d["gap_mean"]<0.55 else "far apart")
    move  = "almost stationary" if d["arm_speed"]<0.010 else \
            ("moving slowly" if d["arm_speed"]<0.025 else "moving fast")
    trend = "the hands are closing on something" if d["closing"] else \
            ("the hands are opening away from something" if d["opening"] else "the hand separation is steady")
    hand  = "the fingers change pose noticeably" if d["hand_change"]>0.008 else "the fingers barely move"
    feas  = (" Halving the control rate would demand a single-step joint move of "
             f"{d['merge_demand']:.3f} rad, which is beyond anything this robot performed in the "
             "demonstrations." ) if d["merge_demand"]>0.159 else ""
    return ("MEASURED FROM THE PLANNED MOTION (these are computed facts, not guesses): "
            f"over the next 0.53 s the two wrists are {d['gap_mean']:.2f} m apart on average "
            f"({close}), the separation changes by {d['gap_change']:.3f} m and {trend}; the arms are "
            f"{move} at {d['arm_speed']:.3f} rad per step; {hand}.{feas}")

# ---- 3층: VLM에게 남기는 질문 ----------------------------------------------
# 계산으로 알 수 없는 것만: 무엇을 다루고 있고, 그 물체·국면이 압축을 견디는가.
ASK=("You can see the scene, and the measured facts above tell you exactly what the arms and hands "
 "are doing. Judge only what the measurements cannot tell you. Answer each check on its own line as "
 "\"A) YES\" or \"A) NO\", in order, nothing else:\n"
 "A) Is the object being handled soft or unstable - a plastic bag, a loose pile, something that would\n"
 "   shift or deform if moved abruptly - rather than a rigid box?\n"
 "B) Does the object's final position or facing matter here - being set down on the conveyor, lined\n"
 "   up, or turned so its label shows - rather than just being moved somewhere?\n"
 "C) Is the robot at a moment where losing the object would ruin the attempt - taking its weight,\n"
 "   or letting go of it - rather than already holding it securely?\n"
 "D) Would running this half-second at double speed still produce the same result?\nAnswer:")

SCALE=("SCALE REFERENCE (measured on this robot over 12 episodes). The two wrists sit 0.30-0.75 m apart; "
 "below 0.42 m a package is pinched between them. Each 7-joint arm moves 0.012 rad per step at the "
 "median and 0.042 at the 90th percentile. One step is 1/30 s. The arms take absolute joint targets, "
 "so halving the control rate means executing every second target and skipping the one between.")
