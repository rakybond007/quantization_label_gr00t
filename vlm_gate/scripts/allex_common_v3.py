"""allex 질문 v3 — 60프레임 육안 라벨과 실제 파지 방식 관찰에 기반해 재작성.

관찰 (720p 콘택트 시트 60장 직접 확인):
  이 로봇은 손가락으로 쥐지 않는다. 두 손을 물체 양옆에 대고 집게처럼 조인다.
  그래서 v1/v2의 "손가락 개폐"(AUC 0.55), "손목 회전"(전 프레임 0.00),
  "양팔 동시 활성"(0.46)은 전부 사문이었다. 살아남은 축은
  자세를 맞춰 놓는 구간(0.75), 단순 이동 여부(역방향 0.68), 저속 접촉(0.65)이다.

v3는 접촉 자체를 직접 묻는다 — 이 로봇에서 위험 = 손이 물체에 닿아 있거나
닿기/떼기 시작하는 순간이기 때문이다.
"""
import os
T=os.path.expanduser("~/quantization_agent_workspace/assets/datasets/allex_hires_v1")
RA=slice(0,7); LA=slice(7,14); ARM=slice(0,14); RH=slice(14,29); LH=slice(29,44)

VIS_ASK=("Answer these four checks about the camera views. Answer each on its own line as\n"
 "\"A) YES\" or \"A) NO\", in order, nothing else:\n"
 "A) Is either hand touching an object right now - resting on it, pressed against its side,\n"
 "   or squeezing it between both hands?\n"
 "B) Is a hand just now making contact with an object, or just now letting go of one?\n"
 "C) Is an object being set down or lined up so that its position or facing matters -\n"
 "   placed on the conveyor, straightened, or turned so its label shows?\n"
 "D) Are both hands clear of every object - moving through open space, approaching but not\n"
 "   yet touching, or withdrawn?\nAnswer:")

ACT_ASK=("Answer these four checks about the planned motion. Answer each on its own line as\n"
 "\"A) YES\" or \"A) NO\", in order, nothing else:\n"
 "A) Do the two wrists stay closer together than 0.42 m for most of the window, as they do\n"
 "   when a package is pinched between them?\n"
 "B) Does the gap between the two wrists close or open by more than 0.03 m across the window\n"
 "   (the hands clamping onto something or letting it go)?\n"
 "C) Do the arms stay slower than 0.010 rad per step for most of the window?\n"
 "D) Do the arms move faster than 0.040 rad per step for most of the window?\nAnswer:")

SCALE=("SCALE REFERENCE (measured on this robot over 12 episodes). The two wrists sit 0.30-0.75 m "
 "apart; below about 0.42 m they are close enough to have a package pinched between them. Each "
 "7-joint arm moves 0.012 rad per step at the median, 0.002 at the 10th percentile and 0.042 at "
 "the 90th. One step is 1/30 s, so the 16-step window below is 0.53 s. The arms are commanded as "
 "absolute joint targets, so halving the control rate means executing every second target and "
 "skipping the one in between.")

NUM_HEADER=("Planned targets for the next 16 steps, one row per step. Columns: 7 right-arm joints, "
 "7 left-arm joints, right-hand finger mean, left-hand finger mean, right wrist xyz, left wrist xyz "
 "(absolute radians and metres):")
