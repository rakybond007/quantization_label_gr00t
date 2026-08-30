"""allex 국면 기준 가이던스 v5 — 회전 국면을 위험으로 명시.

측정 근거 (파지 중 청크 703개, 회전각 상·하위 25% 비교):
  K2 실행 불가(>0.159 rad)   저회전 4.5%  vs  고회전 49.4%
  K2 적용 후 파지 간격 변화율  4.1        vs  7.5 mm/step
  양손 회전 비대칭            1.7도       vs  15.3도
이 로봇은 손가락이 아니라 두 손바닥의 상대 자세로 물체를 문다. 방향을 바꾸려면
양손이 서로 다르게 움직여야 하고(비대칭), 그 협응이 관절 공간에서 큰 변위가 된다.
스텝을 건너뛰면 절반이 실행 불가가 되고, 간격이 벌어지는 방향으로 어긋나면 미끄러진다.
"""
import numpy as np, os
T=os.path.expanduser("~/quantization_agent_workspace/assets/datasets/allex_hires_v1")
RA=slice(0,7); LA=slice(7,14); ARM=slice(0,14); RH=slice(14,29); LH=slice(29,44)
MERGE_LIMIT=0.159      # 시연에서 관측된 단일 스텝 최대 관절 변위

def _rot6_to_R(v):
    a,b=v[:,:3],v[:,3:6]
    e1=a/np.linalg.norm(a,axis=1,keepdims=True)
    b2=b-(e1*b).sum(1,keepdims=True)*e1
    e2=b2/np.linalg.norm(b2,axis=1,keepdims=True)
    return np.stack([e1,e2,np.cross(e1,e2)],axis=2)

def _ang(Ra,Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra@Rb.T)-1)/2,-1,1))))

def descriptors(A, WR, WL, f, n=16):
    w=slice(f, min(f+n, len(A)))
    gap=np.linalg.norm(WR[w,:3]-WL[w,:3],axis=1)
    mid=(WR[w,:3]+WL[w,:3])/2
    sp=np.linalg.norm(np.diff(A[w,ARM],axis=0),axis=1)
    rh=A[w,RH].mean(1); lh=A[w,LH].mean(1)
    Rr=_rot6_to_R(WR[w,3:9]); Rl=_rot6_to_R(WL[w,3:9])
    rr=_ang(Rr[-1],Rr[0]); rl=_ang(Rl[-1],Rl[0])
    merged=np.linalg.norm(A[w][2::2]-A[w][0:-2:2],axis=1) if len(A[w])>2 else np.array([0.0])
    return {"gap_mean":float(gap.mean()),"gap_change":float(gap.max()-gap.min()),
            "gap_rate":float(np.abs(np.diff(gap)).max()) if len(gap)>1 else 0.0,
            "closing":bool(gap[-1]<gap[0]-0.01),"opening":bool(gap[-1]>gap[0]+0.01),
            "held":bool(gap.mean()<0.42),
            "arm_speed":float(sp.mean()) if len(sp) else 0.0,
            "hand_change":float(max(abs(rh[-1]-rh[0]),abs(lh[-1]-lh[0]))),
            "wrist_rot":float(max(rr,rl)), "rot_asym":float(abs(rr-rl)),
            "translation":float(np.linalg.norm(mid[-1]-mid[0])),
            "merge_demand":float(merged.max()) if len(merged) else 0.0,
            "wrist_z":float((WR[w,2].mean()+WL[w,2].mean())/2)}

def facts_text(x):
    grip=("a package is pinched between the two palms" if x["held"] else
          "the hands are too far apart to be holding anything between them")
    trend=("the palms are closing" if x["closing"] else
           "the palms are separating" if x["opening"] else "the palm separation is steady")
    move=("almost stationary" if x["arm_speed"]<0.010 else
          "moving slowly" if x["arm_speed"]<0.025 else "moving fast")
    rot =(f"the wrists turn {x['wrist_rot']:.0f} deg across the window, the two of them differing "
          f"by {x['rot_asym']:.0f} deg" if x["wrist_rot"]>=5 else
          "the wrists barely turn")
    tr  =f"the grasp centre travels {x['translation']*100:.0f} cm"
    hand=("the fingers change pose noticeably" if x["hand_change"]>0.008 else "the fingers barely move")
    feas=(f" Halving the control rate would demand a single-step joint move of {x['merge_demand']:.3f} rad, "
          f"beyond the {MERGE_LIMIT} rad this robot ever produced in the demonstrations."
          if x["merge_demand"]>MERGE_LIMIT else "")
    return ("MEASURED FROM THE PLANNED MOTION over the next 0.53 s (computed facts, not estimates): "
            f"{grip} at {x['gap_mean']:.2f} m and {trend} (up to {x['gap_rate']*1000:.1f} mm per step); "
            f"the arms are {move} at {x['arm_speed']:.3f} rad per step; {rot}; {tr}; {hand}.{feas}")

GUIDANCE=(
"Compression is the default. Judge the MOMENT, not the task. This robot has no pinch grip: it "
"presses both palms against opposite faces of a parcel and holds it by the relative pose of the "
"two hands.\n"
"The question that separates phases: DOES THE ROBOT HAVE TO MAINTAIN ITS HOLD RIGHT NOW? "
"Compression makes the arms move twice as far per step and lets the palms drift a few millimetres. "
"That changes the outcome only while a parcel is being held between the palms and the hold is "
"changing; when the hands hold nothing, or hold something in a settled, unchanging pose, the error "
"is absorbed.\n"
"COARSE - safe to compress:\n"
"  - reaching toward a parcel before contact, and withdrawing after letting go;\n"
"  - carrying a parcel from one place to another while both palms keep the same relative pose and "
"the parcel's facing does not change;\n"
"  - sweeping over empty conveyor, or repositioning the torso between items.\n"
"DELICATE - needs full rate:\n"
"  - the palms converging onto a parcel until its weight is taken, and separating from one being "
"set down, until it stands on its own;\n"
"  - turning, rolling, or tipping a held parcel so a different face points a different way. The two "
"hands must move differently to turn it, and that relative motion is what holds it, so skipping "
"targets makes the palms travel further per step and the parcel can slip;\n"
"  - setting a parcel onto the conveyor where its position or facing decides the outcome;\n"
"  - handling a bag or sack, which shifts shape when moved abruptly.\n"
"When torn, ask what would be lost if the palms arrived a few millimetres off. If the answer is "
"'nothing', answer COARSE. Default to COARSE - compression is the point.")

ASK=("The measurements above are exact - do not re-estimate them. Judge only what the cameras show "
 "that the numbers cannot. Answer each check on its own line as \"A) YES\" or \"A) NO\", in order, "
 "nothing else. YES and NO refer only to the question asked.\n"
 "A) Is the item a bag, sack, or otherwise soft and liable to shift, rather than a rigid box?\n"
 "B) Is the package being turned or tipped to change which way it faces, rather than simply being\n"
 "   carried from one place to another?\n"
 "C) Is the package's final resting position or facing being established right now - lowered onto\n"
 "   the conveyor, straightened, or lined up?\n"
 "D) Are the hands empty, or reaching toward a package they have not yet touched?\nAnswer:")
