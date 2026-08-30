"""로보카사 v6 — 1콜 설계.

  2층(계산): 액션에서 그리퍼 전이·방향 반전·닫힘저속·감속·병합 실현가능성을 정확히 계산.
             기존 액션 문항 E~H를 전부 대체한다(VLM은 이걸 0.52~0.93으로밖에 못 읽었다).
  3층(VLM) : 3뷰 + 진화 가이던스 + 계산 결과를 '사실'로 진술 + 시각으로만 알 수 있는 4문항.
             정지화면 한계는 계산 사실이 메운다.
집계는 계산 위험 플래그와 VLM 위험 답을 함께 noisy-OR.
"""
import json, os, sys, numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlm_gate import VLMGate
from robocasa_descriptors import descriptors, facts_text, computed_risk
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
PORT=sys.argv[1]; SHARD=int(sys.argv[2]); NSH=int(sys.argv[3])
STRAT=(len(sys.argv)>4 and sys.argv[4]=="strat")
_SFX={"phase6":"_phase6","phase":"_phase","phase2":"_phase2","phase3":"_phase3","phase4":"_phase4","phase5":"_phase5"}.get(os.environ.get("GUIDANCE",""),"")
OUT=(f"{BASE}/output/_gate_distill/v6b{_SFX}_strat.jsonl" if STRAT
     else f"{BASE}/output/_gate_distill/v6b{_SFX}_s{NSH}_{SHARD}.jsonl")
info=json.load(open(f"{DS}/meta/info.json"))
instr={}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d=json.loads(l); c=[t for t in d.get("tasks",[]) if isinstance(t,str) and len(t.split())>1 and t!="Valid"]
    instr[d["episode_index"]]=c[0] if c else ""
# 극성 중립 가이던스 — 내용 동일, YES/NO 결속만 제거
# GUIDANCE=phase 이면 국면 기준 가이던스. 기존 가이던스는 규칙을 태스크 종류로 써놓아
# 태스크 전체가 한 덩어리로 처리됐다(OpenDrawer 통과율 0%, CoffeePressButton 92%).
_GF=("robocasa_guidance_phase_v5.txt" if os.environ.get("GUIDANCE","")=="phase6"
     else "robocasa_guidance_phase_v3.txt" if os.environ.get("GUIDANCE","")=="phase5"
     else "robocasa_guidance_phase_v2.txt" if os.environ.get("GUIDANCE","")=="phase4"
     else "robocasa_guidance_phase_v1.txt" if os.environ.get("GUIDANCE","").startswith("phase")
     else "robocasa_cosmos_ttl_best_guidance_aligned.txt")
if not os.environ.get("GUIDANCE"):
    # 기본값은 v6b — 태스크 종류로 규칙을 쓴 옛 가이던스다. 이걸 의도해서 고른
    # 사람은 거의 없으므로, 조용히 넘어가지 말고 무엇이 나올지 말해준다.
    print("[WARN] GUIDANCE 미설정 -> v6b (ttl_aligned, 4문항). "
          "phase5/phase6 를 원하면 GUIDANCE 를 지정할 것.", flush=True)
print(f"[gen] GUIDANCE={os.environ.get('GUIDANCE') or 'v6b(default)'} "
      f"guidance={_GF} out={os.path.basename(OUT)}", flush=True)
G=open(f"{BASE}/analysis/_evolver/_varkA/{_GF}").read().strip()

# 계산이 이미 답하는 것을 다시 물으면 모델은 사실을 복창할 뿐이다(v6 1차: 답 2종으로 붕괴).
# 남기는 질문은 액션 숫자로는 원리적으로 알 수 없는 것 — 무엇을 상대하고 있는 장면인가.
# 문항도 국면 기준이어야 한다. "문·서랍을 당겨 여는가"는 태스크를 묻는 질문이라
# OpenDrawer 전 구간에서 0.79로 켜졌고, 가이던스를 국면 기준으로 바꿔도 소용이 없었다.
# 계산이 답하는 것(그리퍼 개폐·속도·반전·병합 실현가능성)은 사실로만 진술하고 묻지 않는다.
# 문항도 "잃을 것이 있는가" 축으로. CoffeePressButton은 압축하면 성공률 0.82->1.00,
# 스텝 291->71(76% 절감)이다 — 그리퍼가 닫히지만 누르려고 쥐는 것이라 놓칠 물건이 없다.
# 반대로 당겨 열기·운반은 힘에 맞서 파지를 유지해야 한다.
# 축은 "그리퍼가 지금 파지를 유지해야 하는 국면인가". CoffeePressButton은 압축하면
# 성공률 0.82->1.00, 스텝 291->71이다 — 손가락이 닫히지만 누르려고 쥐는 것이라
# 유지해야 할 파지가 없다. 반대로 당겨 열기·운반은 힘에 맞서 파지를 유지해야 한다.
ASK=("The measurements above already tell you how the arm and gripper move; do not repeat them. "
 "Answer only what the cameras show about the MOMENT in front of you. Answer each check on its own "
 "line as \"A) YES\" or \"A) NO\", in order, nothing else. YES and NO refer only to the question asked.\n"
 "A) Is the gripper acting on something fixed in place - pressing a button or switch, turning a knob\n"
 "   or dial, pushing a door or drawer shut - while holding nothing that it could drop?\n"
 "B) Is the gripper taking up an object's weight right now, or setting the object down and letting\n"
 "   go of it?\n"
 "C) Is an object the gripper is holding being lined up with a confined space it must fit into - a\n"
 "   sink basin, a shelf, the microwave interior, a burner - and not yet resting in it?\n"
 "D) Is the gripper closing onto a handle that it will then have to pull against, at the moment the\n"
 "   hold is being established?\nAnswer:")

# phase6 확정본 — 다섯 축에 문항 하나씩.
# 파일럿에서 D·E 가 차단 결정을 0.007/0.004 밖에 못 바꿨지만, 원인은 문항이 아니라
# 그때 쓰던 가이던스가 그 두 축을 아예 서술하지 않은 것이었다. 설명 없이 물으면
# 모델이 답할 근거가 없다. v5 가 다섯 축을 모두 서술하므로 D·E 를 되살렸다.
# (한때 셋으로 줄였다가 되돌렸다. 낮게 나온다는 이유만으로 문항을 버리면 안 된다 —
#  값은 연속이라 낮아도 noisy-OR 에 기여하고, 폐루프만이 문항을 퇴출할 수 있다.)
ASK6=("The measurements above already tell you how the arm and gripper move; do not repeat them. "
 "Answer only what the cameras show about the MOMENT in front of you. Answer each check on its own "
 "line as \"A) YES\" or \"A) NO\", in order, nothing else. YES and NO refer only to the question asked.\n"
 "A) Is the gripper touching nothing at all, moving through empty space?\n"
 "B) Is the gripper holding a handle, a knob, a lever, or the edge of a door or drawer?\n"
 "C) Is the gripper closing onto an object until its weight is taken, or setting an object down\n"
 "   and letting go of it?\n"
 "D) Is there a rim, an edge, a shelf or a wall in the way, so the gripper or what it carries\n"
 "   would hit it going straight and has to lift over or go around?\n"
 "E) Is the target so far from the arm that the robot has to drive its base while the arm is\n"
 "   still moving?\nAnswer:")

gate=VLMGate(f"http://127.0.0.1:{PORT}", timeout=180)
TIL=f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
MAN=os.environ.get("MANIFEST", f"{BASE}/output/_gate_distill/tiles_manifest.txt")
_P6 = os.environ.get("GUIDANCE","")=="phase6"
_ASK, _NQ, _SLOTS = (ASK6, 5, "ABCDE") if _P6 else (ASK, 4, "ABCD")
done=set()
if os.path.exists(OUT):
    for l in open(OUT):
        try: r=json.loads(l); done.add((r["ep"],r["f"]))
        except Exception: pass
WANT=None
if STRAT:
    WANT=set()
    for l in open(f"{BASE}/output/_gate_distill/cosmos_2call_bits.jsonl"):
        try: r=json.loads(l); WANT.add((r["ep"],r["f"]))
        except Exception: pass
acts={}
def A(ep):
    if ep not in acts:
        ch=ep//info["chunks_size"]
        try: acts[ep]=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
        except Exception: acts[ep]=None
        if len(acts)>40:
            for k in list(acts)[:20]: acts.pop(k,None)
    return acts[ep]
out=open(OUT,"a"); n=0
for nm in sorted(open(MAN).read().split()):
    ep=int(nm[2:6]); f=int(nm.split("_f")[1][:3])
    if WANT is not None and (ep,f) not in WANT: continue
    if WANT is None and ep % NSH != SHARD: continue
    if (ep,f) in done: continue
    a=A(ep)
    if a is None or f>=len(a)-4: continue
    x=descriptors(a,f)
    im=np.array(Image.open(f"{TIL}/{nm}").convert("RGB")); h,w,_=im.shape
    views=[Image.fromarray(im[:, k*w//3:(k+1)*w//3]) for k in range(3)]
    ins=f"{instr.get(ep,'')}\n{facts_text(x)}"
    r=gate.judge(views, ins, G, question=_ASK, n_ask=_NQ)
    c=r.get("confidences") or [0.0]*_NQ
    rec={"ep":ep,"f":f, **{k:float(v) for k,v in zip(_SLOTS,c)}, **computed_risk(x),
         "speed_mean":x["speed_mean"], "ans":r.get("answer","")}
    out.write(json.dumps(rec)+"\n"); n+=1
    if n%200==0: print(f"shard{SHARD}: {n}", flush=True); out.flush()
out.close(); print(f"shard{SHARD} 완료 {n} -> {OUT}")
