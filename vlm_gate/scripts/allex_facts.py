"""판정기에 나가는 계산 사실. **배치 생성기와 API 판정기가 같은 것을 쓴다.**

둘이 다른 문구를 쓰면 비교가 안 되므로 한 곳에 둔다.

압축 요구량은 좋다/나쁘다가 아니라 **이 녹화본의 분포에 견주어** 말한다.
우리는 느린 시연보다 빨리 움직이는 것이 목적이고, 시연 범위를 넘는 것 자체는
감점 사유가 아니다 -- 시뮬 벤치마크에서 clip 을 푸는 것과 같은 이치다.
"""
import allex_v3_checks as TH   # 문턱값

def facts(x):
    """압축 판단에 쓰이는 결론만. 날숫자를 넘기면 해석이 판정기 몫이 된다."""
    p = []
    p.append("only one arm is working; the other is idle" if x["one_handed"]
             else "both arms are working together")
    if not x["one_handed"]:
        g = x["gap_mean"]
        p.append("the palms are close in" if g < TH.GAP_NEAR else
                 "the palms are a middling distance apart" if g < TH.GAP_FAR else
                 "the palms are well apart")
        p.append("and drawing together" if x["closing"] else
                 "and moving apart" if x["opening"] else "and holding that distance")
    h = x["hand_change"]
    p.append("the fingers are still" if h < TH.FING_STILL else
             "the fingers are shifting a little" if h < TH.FING_WORK else
             "the fingers are working")
    v = x["arm_speed"]
    p.append("the arms are barely moving" if v < TH.SPEED_SLOW else
             "the arms are moving at a normal pace" if v < TH.SPEED_FAST else
             "the arms are moving fast")
    r = x["wrist_rot"]
    p.append("the wrists barely turn" if r < TH.ROT_LITTLE else
             "the wrists turn a fair amount" if r < TH.ROT_LOT else
             "the wrists turn a great deal")
    # **압축 요구량은 이 녹화본의 분포에 견주어 말한다.**
    # 앞 판에서 "시연에 없던 크기" 를 한계 위반처럼 적었다가 뺐다. 우리는 느린
    # 시연보다 빨리 움직이는 것이 목적이고, 시연 범위를 넘는 것 자체는 감점
    # 사유가 아니다 -- 시뮬 벤치마크에서 clip 을 푸는 것과 같은 이치다.
    # 좋다/나쁘다 대신 **다른 순간들에 비해 큰가 작은가**만 말한다. 분위는
    # 9,179 청크에서 낸 것이다(k2: p20 .059 · p40 .084 · p60 .116 · p80 .173).
    k2, k3 = x["merge_demand_k2"], x["merge_demand_k3"]
    def band(d, q):
        return ("much smaller than most moments in this recording" if d < q[0] else
                "smaller than most" if d < q[1] else
                "about average" if d < q[2] else
                "larger than most" if d < q[3] else
                "much larger than most moments in this recording")
    Q2 = (0.059, 0.084, 0.116, 0.173)
    Q3 = (0.079, 0.114, 0.156, 0.234)
    p.append(f"at 2x this stretch would move {k2:.3f} rad in one step, {band(k2, Q2)}")
    p.append(f"at 3x {k3:.3f} rad, {band(k3, Q3)}")
    return "; ".join(p) + "."
