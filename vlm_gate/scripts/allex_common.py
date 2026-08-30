"""allex(양팔 휴머노이드) 라벨링 공용 정의 — 질문, 스케일 기준, 관절 슬라이스.

cosmos(슬롯 스캐폴드)와 API(같은 형식 + top_logprobs)가 같은 질문을 쓰도록 한 곳에 둔다.

로보카사에서 그대로 옮겨오면 안 되는 이유:
  - 그리퍼 0/1이 아니라 손마다 15관절. "닫힘/열림"이 이진 신호가 아니다.
  - 팔이 둘이다. 한쪽만 보면 양손 협응(잡아 건네기, 한 손으로 받치고 다른 손이 조정)을
    통째로 놓친다. 이 태스크의 "바코드를 위로 향하게" 구간이 정확히 그것이다.
  - 액션이 절대 관절각이라 K2가 "델타 합산"이 아니라 "격 스텝 건너뛰기"다.
스케일 기준은 이 데이터셋 6에피소드 18,867스텝에서 직접 측정했다.
"""
import os
T=os.path.expanduser("~/quantization_agent_workspace/assets/datasets/allex_frontier_demo_v1")

RA=slice(0,7); LA=slice(7,14); ARM=slice(0,14); RH=slice(14,29); LH=slice(29,44)

VIS_ASK=("Answer these four checks about the camera views. Answer each on its own line as\n"
 "\"A) YES\" or \"A) NO\", in order, nothing else:\n"
 "A) Are the fingers of either hand closing onto an object right now, or opening to release it?\n"
 "B) Is a held object being reoriented in place - turned, rolled, or regrasped - or passed\n"
 "   between the two hands?\n"
 "C) Is an object being set down onto a target with its pose mattering - aligned, levelled,\n"
 "   or turned to face a particular way?\n"
 "D) Is this plain gross motion - reaching toward something, carrying a firmly held object,\n"
 "   retracting, or repositioning the torso?\nAnswer:")

ACT_ASK=("Answer these four checks about the planned joint trajectory. Answer each on its own\n"
 "line as \"A) YES\" or \"A) NO\", in order, nothing else:\n"
 "A) Do either hand's finger joints change by more than 0.008 rad within the next 16 steps\n"
 "   (a grasp closing or a release opening)?\n"
 "B) Are BOTH arms active at the same time - each moving more than 0.024 rad per step for\n"
 "   several consecutive steps (two-handed manipulation)?\n"
 "C) Does either wrist rotate more than 30 degrees in total across the window while that\n"
 "   hand's fingers stay closed (reorienting a held object)?\n"
 "D) Do the arms stay slower than 0.008 rad per step for most of the window while the\n"
 "   fingers hold a closed pose (slow in-contact work)?\nAnswer:")

SCALE=("SCALE REFERENCE (measured on this robot, 18,867 steps). Per step: each 7-joint arm moves "
 "0.011-0.013 rad at the median, 0.002 at the 10th percentile, 0.040-0.043 at the 90th; both arms "
 "are active at once in only 6% of steps, so simultaneous two-handed motion is unusual. Each wrist "
 "rotates 0.61 deg per step at the median and 2.1-2.3 deg at the 90th, so a 16-step window turns "
 "about 30 deg at the 90th percentile. The 15 finger joints of a hand move 0.0003-0.0004 rad per "
 "step at the median, so any sustained finger motion above 0.008 rad over the window is a real "
 "grasp or release, not noise. One step is 1/30 s; the 16-step window below is 0.53 s.")

NUM_HEADER=("Planned targets for the next 16 steps, one row per step. Columns: 7 right-arm joints, "
 "7 left-arm joints, right-hand finger mean, left-hand finger mean, right wrist xyz, left wrist xyz "
 "(absolute radians and metres):")
