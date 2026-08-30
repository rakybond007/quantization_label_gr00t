import random
"""초소규모 confidence 프로토콜 비교: (A) top20 YES/NO 합산, (B) 5단계 언어척도"""
import json, base64, os, sys, io
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0,"/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/scripts")
from vlm_gate import SYSTEM
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
mode=sys.argv[1]
TAG=mode
if mode.startswith(("u_","s_")): mode=mode[2:]   # 접두사는 출력 디렉토리 구분용
OUT=f"{BASE}/output/_gate_distill/exp_cp_{TAG}"; os.makedirs(OUT, exist_ok=True)
G=open(f"{BASE}/analysis/_evolver/_varkA/robocasa_cosmos_ttl_best_guidance.txt").read().strip()
TIL=f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
info=json.load(open(f"{DS}/meta/info.json"))
instr={}
for l in open(f"{DS}/meta/episodes.jsonl"):
    d=json.loads(l); c=[t for t in d.get("tasks",[]) if isinstance(t,str) and len(t.split())>1 and t!="Valid"]
    instr[d["episode_index"]]=c[0] if c else ""
have=set(os.listdir(TIL))
byep={}
for n in sorted(have): byep.setdefault(int(n[2:6]),[]).append(int(n.split("_f")[1][:3]))
# 태스크 층화: 24개 환경 클래스에서 골고루
import re as _re
_envof={}
for _l in open(f"{DS}/meta/episodes.jsonl"):
    _d=json.loads(_l)
    _c=[t for t in _d.get("tasks",[]) if isinstance(t,str) and _re.fullmatch(r"[A-Z][A-Za-z]+",t) and t!="Valid"]
    if _c: _envof[_d["episode_index"]]=_c[0]
_bycls={}
for _e in sorted(byep):
    if len(byep[_e])<20: continue
    _bycls.setdefault(_envof.get(_e,"?"),[]).append(_e)
random.seed(11)
eps=[]
for _cls in sorted(_bycls):
    eps += random.sample(_bycls[_cls], min(9, len(_bycls[_cls])))
print("층화 표본: 태스크", len(_bycls), "종, 에피소드", len(eps))
view_note=("You are shown 3 camera views (concatenated left-to-right): agentview-left, agentview-right, wrist close-up.")
if mode=="seq8":
    ask=("Answer in TWO stages.\n"
         "STAGE 1 - look ONLY at the camera views and decide these four, ignoring the numbers for now:\n"
         "  A) Is the gripper closing on an object or a handle right now, or opening to release one?\n"
         "  B) Is a carried object being precisely inserted, aligned, or lowered into a confined receptacle?\n"
         "  C) Is a door or drawer being PULLED OPEN with the grasped handle under load?\n"
         "  D) Is this plain gross motion - reaching, transporting a firmly held object, retracting, a broad "
         "sweep, or pressing a rigidly mounted button or knob?\n"
         "STAGE 2 - now read the action numbers (7 per step: EE delta x,y,z, rotation x,y,z, gripper last, "
         "0=open 1=closed) and decide these four:\n"
         "  E) Does the gripper command change value within the next 16 steps?\n"
         "  F) Is there a real direction reversal - two consecutive steps both with |d| > 0.10 turning more "
         "than 90 degrees?\n"
         "  G) Is the gripper closed while |d| stays below 0.12 for most of the window?\n"
         "  H) Do the magnitudes decrease steadily and end below 0.15?\n"
         "Answer with the two stages separated by a slash, four characters each, Y or N, no other text. "
         "Example: NNNY/NYNN")
    K=5
elif mode=="seq8":
    ask=("Answer in TWO stages.\n"
         "STAGE 1 - look ONLY at the camera views and decide these four, ignoring the numbers for now:\n"
         "  A) Is the gripper closing on an object or a handle right now, or opening to release one?\n"
         "  B) Is a carried object being precisely inserted, aligned, or lowered into a confined receptacle?\n"
         "  C) Is a door or drawer being PULLED OPEN with the grasped handle under load?\n"
         "  D) Is this plain gross motion - reaching, transporting a firmly held object, retracting, a broad "
         "sweep, or pressing a rigidly mounted button or knob?\n"
         "STAGE 2 - now read the action numbers (7 per step: EE delta x,y,z, rotation x,y,z, gripper last, "
         "0=open 1=closed) and decide these four:\n"
         "  E) Does the gripper command change value within the next 16 steps?\n"
         "  F) Is there a real direction reversal - two consecutive steps both with |d| > 0.10 turning more "
         "than 90 degrees?\n"
         "  G) Is the gripper closed while |d| stays below 0.12 for most of the window?\n"
         "  H) Do the magnitudes decrease steadily and end below 0.15?\n"
         "Answer with the two stages separated by a slash, four characters each, Y or N, no other text. "
         "Example: NNNY/NYNN")
    K=5
if mode=="vis4":
    ask=("Look ONLY at the camera views and answer four checks about what is happening right now:\n"
         "  A) Is the gripper closing on an object or a handle, or opening to release one?\n"
         "  B) Is a carried object being precisely inserted, aligned, or lowered into a confined receptacle "
         "(sink basin, cabinet shelf, microwave, burner)?\n"
         "  C) Is a door or drawer being PULLED OPEN with the grasped handle under load?\n"
         "  D) Is this plain gross motion - reaching, transporting a firmly held object, retracting, a broad "
         "sweep, or pressing a rigidly mounted button or knob?\n"
         "Answer with EXACTLY four characters, one per check in order A,B,C,D, each Y or N. No other text.")
    K=5
elif mode=="bits8cal":
    ask=("Answer eight yes/no checks. A-D come from the CAMERA VIEWS, E-H from the ACTION NUMBERS "
         "(7 per step: EE delta x,y,z, rotation x,y,z, gripper command last, 0=open 1=closed).\n"
         "SCALE REFERENCE for this dataset: a step magnitude |d| is typically 0.34; 0.12 is slow (10th pct), "
         "0.73 is fast (90th pct); consecutive steps turn by about 11 degrees on average and a turn beyond "
         "90 degrees occurs in only ~1% of steps; while carrying an object |d| is still about 0.33, so a "
         "closed gripper does NOT imply slow motion.\n"
         "FROM THE VIEWS:\n"
         "  A) Is the gripper closing on an object or handle right now, or opening to release one?\n"
         "  B) Is a carried object being precisely inserted, aligned, or lowered into a confined receptacle "
         "(sink basin, cabinet shelf, microwave, burner)?\n"
         "  C) Is a door or drawer being PULLED OPEN with the grasped handle under load, so the arm must "
         "track the hinge or slide?\n"
         "  D) Is this plain gross motion - reaching, transporting a firmly held object, retracting, a broad "
         "sweep, or pressing a rigidly mounted button or knob?\n"
         "FROM THE NUMBERS (next 16 steps):\n"
         "  E) Does the gripper command change value?\n"
         "  F) Is there a real direction reversal - two consecutive steps both with |d| > 0.10 turning by "
         "more than 90 degrees? (ignore sign changes of tiny components)\n"
         "  G) Is the gripper closed while |d| stays below 0.12 for most of the window?\n"
         "  H) Do the magnitudes decrease steadily and end below 0.15?\n"
         "Answer with EXACTLY eight characters, one per check in order A..H, each Y or N. No other text.")
    K=5
elif mode=="bits4cal":
    ask=("Each action step has 7 numbers: end-effector delta x,y,z, then rotation x,y,z, then the GRIPPER "
         "command (last, 0=open 1=closed).\n"
         "SCALE REFERENCE for this robot and dataset (computed from the demonstrations):\n"
         "  the magnitude |d| = sqrt(dx^2+dy^2+dz^2) of one step is typically 0.34; 0.12 is slow (10th "
         "percentile), 0.73 is fast (90th percentile);\n"
         "  consecutive steps normally point in almost the same direction - the turn angle between them is "
         "about 11 degrees on average, and a turn beyond 90 degrees happens in only ~1% of steps;\n"
         "  a closed gripper does NOT mean slow motion: while carrying an object |d| is still about 0.33.\n"
         "Using those references, check four conditions over the next 16 steps:\n"
         "  A) Does the gripper command (last number) change value?\n"
         "  B) Is there a REAL direction reversal - two consecutive steps that both have |d| > 0.10 and turn "
         "by more than 90 degrees? (ignore sign changes of tiny components; those are noise)\n"
         "  C) Is the gripper closed (1) while |d| stays below 0.12 for most of the window (i.e. genuinely "
         "slow, not ordinary carrying)?\n"
         "  D) Do the magnitudes decrease steadily and end below 0.15 (settling onto a target)?\n"
         "Answer with EXACTLY four characters, one per condition in order A,B,C,D, each Y or N. "
         "No spaces, no other text. Example: NYNN")
    K=5
elif mode=="bits8":
    ask=("Answer eight yes/no checks. The first four are read from the CAMERA VIEWS (what is happening "
         "semantically), the last four from the ACTION NUMBERS (7 per step: EE delta x,y,z, rotation x,y,z, "
         "gripper command last, 0=open 1=closed).\n"
         "FROM THE VIEWS:\n"
         "  A) Is the gripper closing on an object or a handle right now, or opening to release one?\n"
         "  B) Is a carried object being precisely inserted, aligned, or lowered into a confined receptacle "
         "(sink basin, cabinet shelf, microwave, burner)?\n"
         "  C) Is a door or drawer being PULLED OPEN, with the grasped handle under load so the arm must "
         "track the hinge or slide?\n"
         "  D) Is this instead plain gross motion - reaching, transporting a firmly held object, retracting, "
         "a broad sweep, or pressing a rigidly mounted button/knob?\n"
         "FROM THE NUMBERS:\n"
         "  E) Does the gripper command change value within the next 16 steps?\n"
         "  F) Does the sign of any end-effector delta flip within the window?\n"
         "  G) Is the gripper closed while the deltas stay small?\n"
         "  H) Do the delta magnitudes shrink steadily toward the end of the window?\n"
         "Answer with EXACTLY eight characters, one per check in order A..H, each Y or N. "
         "No spaces, no other text. Example: NNNYNYNN")
    K=5
elif mode=="ladder":
    ask=("Each action step has 7 numbers: EE delta x,y,z, rotation x,y,z, then the GRIPPER command "
         "(last, 0=open 1=closed).\n"
         "Below are four claims about compressing the next ~1 second (running it at half rate by averaging "
         "consecutive pairs of steps). They get progressively stronger. State whether you AGREE with each:\n"
         "  A) At least the first quarter of this window is gross motion that could be safely compressed.\n"
         "  B) At least half of this window could be safely compressed.\n"
         "  C) At least three quarters of this window could be safely compressed.\n"
         "  D) The ENTIRE window could be compressed with no effect on the outcome — no grasp or release, "
         "no path reversal, no fine placement, no settling onto a target occurs in it.\n"
         "Answer with EXACTLY four characters, one per claim in order A,B,C,D, each Y (agree) or N (disagree). "
         "No spaces, no other text. Example: YYNN")
    K=5
elif mode=="bits4":
    ask=("Each action step has 7 numbers: EE delta x,y,z, rotation x,y,z, then the GRIPPER command "
         "(last, 0=open 1=closed). Check these four conditions over the next 16 steps and report each one:\n"
         "  A) Does the gripper command change value (grasp or release happening)?\n"
         "  B) Does the sign of any EE delta flip (the path bends)?\n"
         "  C) Is the gripper closed while the EE deltas stay small (fine placement/insertion)?\n"
         "  D) Do the EE delta magnitudes shrink steadily toward the end (settling onto a target)?\n"
         "Answer with EXACTLY four characters, one per condition in order A,B,C,D, each Y (condition holds) "
         "or N (does not hold). No spaces, no other text. Example: NYNN")
    K=5
elif mode=="vote":
    ask=("Each action step has 7 numbers: EE delta x,y,z, rotation x,y,z, then the GRIPPER command "
         "(last, 0=open 1=closed). Answer NO if within the next 16 steps the gripper command changes, "
         "the sign of an EE delta flips, the gripper is closed while deltas stay small, or the delta "
         "magnitudes shrink steadily toward the end. Otherwise answer YES.\n"
         "Can the next ~1 second of motion be compressed (run at half rate)? Answer YES or NO.")
    K=5
elif mode=="grade10r":
    ask=("Each action step has 7 numbers: end-effector delta x,y,z, rotation x,y,z, then the GRIPPER command "
         "(last, 0=open 1=closed).\n"
         "Rate how much of the next ~1 second can be safely executed at half rate, on a 1-10 scale anchored to "
         "what the numbers and views show:\n"
         "  1-2  = the gripper command CHANGES in this window (grasp or release is happening) — never compress\n"
         "  3-4  = the gripper is closed while the deltas stay small (fine placement, insertion, alignment), "
         "or the delta magnitudes shrink steadily toward the end (settling onto a target)\n"
         "  5-6  = the sign of an end-effector delta flips (the path bends) but no contact event is imminent\n"
         "  7-8  = ordinary transit with moderate, mostly steady deltas\n"
         "  9-10 = clear gross motion: large steady deltas, free-space reaching, carrying a firmly held object, "
         "retracting, or pressing a rigidly mounted button\n"
         "Answer with exactly ONE integer from 1 to 10.")
    K=5
elif mode=="pub":
    ask=("Respond with exactly one word: YES or NO.")
    K=5
elif mode=="generic":
    ask=("The action list gives the robot's planned motion for the next 16 control steps "
         "(7 numbers per step: end-effector delta x,y,z, rotation x,y,z, and the gripper command as the last "
         "number, 0=open 1=closed). Read it together with the images and apply the guidance above — the "
         "numbers tell you motion magnitude, direction changes and gripper transitions that a single frame "
         "cannot show.\n"
         "Can the next ~1 second of motion be compressed (run at half rate)? "
         "Answer YES (compress) or NO (needs precise full-rate control).")
    K=5
elif mode=="binary4":
    ask=("Each action step has 7 numbers: EE delta x,y,z then rotation x,y,z then the GRIPPER command (last).\n"
         "Answer NO if ANY of these holds over the next 16 steps:\n"
         "  (1) GRIPPER: the last number changes value (the gripper opens or closes during this window).\n"
         "  (2) REVERSAL: the sign of any EE delta (x, y or z) flips — the path bends, so averaging pairs "
         "would cut the corner.\n"
         "  (3) FINE CONTACT: the gripper is closed (last number 1) AND the EE deltas are small — this is "
         "precise placement, insertion or alignment, not transport.\n"
         "  (4) DECELERATION: the EE delta magnitudes shrink steadily toward the end of the window — the arm "
         "is settling onto a target.\n"
         "Otherwise answer YES (gross motion: large steady deltas, or carrying a firmly held object).\n"
         "Can the next ~1 second of motion be compressed (run at half rate)? Answer YES or NO.")
    K=5
elif mode=="binaryg":
    ask=("The last number of each action step is the GRIPPER command (0 or 1). If that value CHANGES "
         "anywhere in the next 16 steps, the gripper is opening or closing during this window — answer NO. "
         "Dimensions 5-7 are the end-effector delta xyz: large, smooth values mean gross transit (YES), "
         "small values with sign changes mean fine adjustment (NO).\n"
         "Can the next ~1 second of motion be compressed (run at half rate)? "
         "Answer YES (compress) or NO (needs precise full-rate control).")
    K=5
elif mode=="grade20g":
    ask=("Rate THIS moment on a 1-20 compressibility scale, relative to other moments in such episodes:\n"
         "  1-4  = the gripper is opening or closing on an object/handle right now, or is within ~0.3s of doing so\n"
         "  5-9  = fine insertion or alignment; decelerating into contact\n"
         "  10-14 = ordinary manipulation motion\n"
         "  15-20 = gross motion (free transit, carrying a firmly held object, retracting)\n"
         "The single most important check is the gripper: look at the wrist view and at the planned gripper "
         "command in the action list (last number of each step). If that gripper value CHANGES within the next "
         "16 steps, the grade must be 4 or lower. Grades should spread over the full range.\n"
         "Answer with exactly ONE integer from 1 to 20.")
    K=5
elif mode=="grade20":
    ask=("Rate THIS moment on a 1-20 compressibility scale, judged RELATIVE to all other moments in this "
         "kind of manipulation episode:\n"
         "  1-4  = the most precision-critical instants (gripper closing/opening on an object, "
         "fine insertion or alignment where millimeters matter)\n"
         "  5-9  = approaching or decelerating into contact, delicate but not the critical instant\n"
         "  10-14 = ordinary manipulation motion, moderate care\n"
         "  15-20 = pure gross motion (free-space transit, transporting a firmly held object, retracting)\n"
         "Across a whole episode these grades should SPREAD over the full range — roughly a quarter of "
         "moments fall below 10. Do not default to the top of the scale.\n"
         "Answer with exactly ONE integer from 1 to 20.")
    K=5
elif mode=="ordinal":
    ask=("Of the next 16 control steps, how many could be safely executed at half rate "
         "(merging consecutive pairs) without changing the outcome?\n"
         "Answer with exactly ONE number from: 0, 4, 8, 12, 16.\n"
         "0 = none (precise control needed throughout), 16 = all (pure gross motion).")
    K=5
elif mode=="scale5":
    ask=("How confident are you that the next ~1 second of motion can be compressed (run at half rate)?\n"
         "Answer with exactly ONE word from this scale:\n"
         "CERTAIN (definitely compressible) / LIKELY / UNSURE / DOUBTFUL / IMPOSSIBLE (definitely needs full rate).")
    K=5
else:
    ask=("Can the next ~1 second of motion be compressed (run at half rate)? "
         "Answer YES (compress) or NO (needs precise full-rate control).")
    K=20
if mode=="seq8":
    ask=("Answer in TWO stages.\n"
         "STAGE 1 - look ONLY at the camera views and decide these four, ignoring the numbers for now:\n"
         "  A) Is the gripper closing on an object or a handle right now, or opening to release one?\n"
         "  B) Is a carried object being precisely inserted, aligned, or lowered into a confined receptacle?\n"
         "  C) Is a door or drawer being PULLED OPEN with the grasped handle under load?\n"
         "  D) Is this plain gross motion - reaching, transporting a firmly held object, retracting, a broad "
         "sweep, or pressing a rigidly mounted button or knob?\n"
         "STAGE 2 - now read the action numbers (7 per step: EE delta x,y,z, rotation x,y,z, gripper last, "
         "0=open 1=closed) and decide these four:\n"
         "  E) Does the gripper command change value within the next 16 steps?\n"
         "  F) Is there a real direction reversal - two consecutive steps both with |d| > 0.10 turning more "
         "than 90 degrees?\n"
         "  G) Is the gripper closed while |d| stays below 0.12 for most of the window?\n"
         "  H) Do the magnitudes decrease steadily and end below 0.15?\n"
         "Answer with the two stages separated by a slash, four characters each, Y or N, no other text. "
         "Example: NNNY/NYNN")
    K=5
elif mode=="seq8":
    ask=("Answer in TWO stages.\n"
         "STAGE 1 - look ONLY at the camera views and decide these four, ignoring the numbers for now:\n"
         "  A) Is the gripper closing on an object or a handle right now, or opening to release one?\n"
         "  B) Is a carried object being precisely inserted, aligned, or lowered into a confined receptacle?\n"
         "  C) Is a door or drawer being PULLED OPEN with the grasped handle under load?\n"
         "  D) Is this plain gross motion - reaching, transporting a firmly held object, retracting, a broad "
         "sweep, or pressing a rigidly mounted button or knob?\n"
         "STAGE 2 - now read the action numbers (7 per step: EE delta x,y,z, rotation x,y,z, gripper last, "
         "0=open 1=closed) and decide these four:\n"
         "  E) Does the gripper command change value within the next 16 steps?\n"
         "  F) Is there a real direction reversal - two consecutive steps both with |d| > 0.10 turning more "
         "than 90 degrees?\n"
         "  G) Is the gripper closed while |d| stays below 0.12 for most of the window?\n"
         "  H) Do the magnitudes decrease steadily and end below 0.15?\n"
         "Answer with the two stages separated by a slash, four characters each, Y or N, no other text. "
         "Example: NNNY/NYNN")
    K=5
if mode=="vis4":
    ask=("Look ONLY at the camera views and answer four checks about what is happening right now:\n"
         "  A) Is the gripper closing on an object or a handle, or opening to release one?\n"
         "  B) Is a carried object being precisely inserted, aligned, or lowered into a confined receptacle "
         "(sink basin, cabinet shelf, microwave, burner)?\n"
         "  C) Is a door or drawer being PULLED OPEN with the grasped handle under load?\n"
         "  D) Is this plain gross motion - reaching, transporting a firmly held object, retracting, a broad "
         "sweep, or pressing a rigidly mounted button or knob?\n"
         "Answer with EXACTLY four characters, one per check in order A,B,C,D, each Y or N. No other text.")
    K=5
elif mode=="bits8cal":
    ask=("Answer eight yes/no checks. A-D come from the CAMERA VIEWS, E-H from the ACTION NUMBERS "
         "(7 per step: EE delta x,y,z, rotation x,y,z, gripper command last, 0=open 1=closed).\n"
         "SCALE REFERENCE for this dataset: a step magnitude |d| is typically 0.34; 0.12 is slow (10th pct), "
         "0.73 is fast (90th pct); consecutive steps turn by about 11 degrees on average and a turn beyond "
         "90 degrees occurs in only ~1% of steps; while carrying an object |d| is still about 0.33, so a "
         "closed gripper does NOT imply slow motion.\n"
         "FROM THE VIEWS:\n"
         "  A) Is the gripper closing on an object or handle right now, or opening to release one?\n"
         "  B) Is a carried object being precisely inserted, aligned, or lowered into a confined receptacle "
         "(sink basin, cabinet shelf, microwave, burner)?\n"
         "  C) Is a door or drawer being PULLED OPEN with the grasped handle under load, so the arm must "
         "track the hinge or slide?\n"
         "  D) Is this plain gross motion - reaching, transporting a firmly held object, retracting, a broad "
         "sweep, or pressing a rigidly mounted button or knob?\n"
         "FROM THE NUMBERS (next 16 steps):\n"
         "  E) Does the gripper command change value?\n"
         "  F) Is there a real direction reversal - two consecutive steps both with |d| > 0.10 turning by "
         "more than 90 degrees? (ignore sign changes of tiny components)\n"
         "  G) Is the gripper closed while |d| stays below 0.12 for most of the window?\n"
         "  H) Do the magnitudes decrease steadily and end below 0.15?\n"
         "Answer with EXACTLY eight characters, one per check in order A..H, each Y or N. No other text.")
    K=5
elif mode=="bits4cal":
    ask=("Each action step has 7 numbers: end-effector delta x,y,z, then rotation x,y,z, then the GRIPPER "
         "command (last, 0=open 1=closed).\n"
         "SCALE REFERENCE for this robot and dataset (computed from the demonstrations):\n"
         "  the magnitude |d| = sqrt(dx^2+dy^2+dz^2) of one step is typically 0.34; 0.12 is slow (10th "
         "percentile), 0.73 is fast (90th percentile);\n"
         "  consecutive steps normally point in almost the same direction - the turn angle between them is "
         "about 11 degrees on average, and a turn beyond 90 degrees happens in only ~1% of steps;\n"
         "  a closed gripper does NOT mean slow motion: while carrying an object |d| is still about 0.33.\n"
         "Using those references, check four conditions over the next 16 steps:\n"
         "  A) Does the gripper command (last number) change value?\n"
         "  B) Is there a REAL direction reversal - two consecutive steps that both have |d| > 0.10 and turn "
         "by more than 90 degrees? (ignore sign changes of tiny components; those are noise)\n"
         "  C) Is the gripper closed (1) while |d| stays below 0.12 for most of the window (i.e. genuinely "
         "slow, not ordinary carrying)?\n"
         "  D) Do the magnitudes decrease steadily and end below 0.15 (settling onto a target)?\n"
         "Answer with EXACTLY four characters, one per condition in order A,B,C,D, each Y or N. "
         "No spaces, no other text. Example: NYNN")
    K=5
elif mode=="bits8":
    ask=("Answer eight yes/no checks. The first four are read from the CAMERA VIEWS (what is happening "
         "semantically), the last four from the ACTION NUMBERS (7 per step: EE delta x,y,z, rotation x,y,z, "
         "gripper command last, 0=open 1=closed).\n"
         "FROM THE VIEWS:\n"
         "  A) Is the gripper closing on an object or a handle right now, or opening to release one?\n"
         "  B) Is a carried object being precisely inserted, aligned, or lowered into a confined receptacle "
         "(sink basin, cabinet shelf, microwave, burner)?\n"
         "  C) Is a door or drawer being PULLED OPEN, with the grasped handle under load so the arm must "
         "track the hinge or slide?\n"
         "  D) Is this instead plain gross motion - reaching, transporting a firmly held object, retracting, "
         "a broad sweep, or pressing a rigidly mounted button/knob?\n"
         "FROM THE NUMBERS:\n"
         "  E) Does the gripper command change value within the next 16 steps?\n"
         "  F) Does the sign of any end-effector delta flip within the window?\n"
         "  G) Is the gripper closed while the deltas stay small?\n"
         "  H) Do the delta magnitudes shrink steadily toward the end of the window?\n"
         "Answer with EXACTLY eight characters, one per check in order A..H, each Y or N. "
         "No spaces, no other text. Example: NNNYNYNN")
    K=5
elif mode=="ladder":
    ask=("Each action step has 7 numbers: EE delta x,y,z, rotation x,y,z, then the GRIPPER command "
         "(last, 0=open 1=closed).\n"
         "Below are four claims about compressing the next ~1 second (running it at half rate by averaging "
         "consecutive pairs of steps). They get progressively stronger. State whether you AGREE with each:\n"
         "  A) At least the first quarter of this window is gross motion that could be safely compressed.\n"
         "  B) At least half of this window could be safely compressed.\n"
         "  C) At least three quarters of this window could be safely compressed.\n"
         "  D) The ENTIRE window could be compressed with no effect on the outcome — no grasp or release, "
         "no path reversal, no fine placement, no settling onto a target occurs in it.\n"
         "Answer with EXACTLY four characters, one per claim in order A,B,C,D, each Y (agree) or N (disagree). "
         "No spaces, no other text. Example: YYNN")
    K=5
elif mode=="bits4":
    ask=("Each action step has 7 numbers: EE delta x,y,z, rotation x,y,z, then the GRIPPER command "
         "(last, 0=open 1=closed). Check these four conditions over the next 16 steps and report each one:\n"
         "  A) Does the gripper command change value (grasp or release happening)?\n"
         "  B) Does the sign of any EE delta flip (the path bends)?\n"
         "  C) Is the gripper closed while the EE deltas stay small (fine placement/insertion)?\n"
         "  D) Do the EE delta magnitudes shrink steadily toward the end (settling onto a target)?\n"
         "Answer with EXACTLY four characters, one per condition in order A,B,C,D, each Y (condition holds) "
         "or N (does not hold). No spaces, no other text. Example: NYNN")
    K=5
elif mode=="vote":
    ask=("Each action step has 7 numbers: EE delta x,y,z, rotation x,y,z, then the GRIPPER command "
         "(last, 0=open 1=closed). Answer NO if within the next 16 steps the gripper command changes, "
         "the sign of an EE delta flips, the gripper is closed while deltas stay small, or the delta "
         "magnitudes shrink steadily toward the end. Otherwise answer YES.\n"
         "Can the next ~1 second of motion be compressed (run at half rate)? Answer YES or NO.")
    K=5
elif mode=="grade10r":
    ask=("Each action step has 7 numbers: end-effector delta x,y,z, rotation x,y,z, then the GRIPPER command "
         "(last, 0=open 1=closed).\n"
         "Rate how much of the next ~1 second can be safely executed at half rate, on a 1-10 scale anchored to "
         "what the numbers and views show:\n"
         "  1-2  = the gripper command CHANGES in this window (grasp or release is happening) — never compress\n"
         "  3-4  = the gripper is closed while the deltas stay small (fine placement, insertion, alignment), "
         "or the delta magnitudes shrink steadily toward the end (settling onto a target)\n"
         "  5-6  = the sign of an end-effector delta flips (the path bends) but no contact event is imminent\n"
         "  7-8  = ordinary transit with moderate, mostly steady deltas\n"
         "  9-10 = clear gross motion: large steady deltas, free-space reaching, carrying a firmly held object, "
         "retracting, or pressing a rigidly mounted button\n"
         "Answer with exactly ONE integer from 1 to 10.")
    K=5
elif mode=="pub":
    sys_text=open(f"{BASE}/paper_prompts/final/robocasa_publication_v1.txt").read().strip()
else:
    sys_text=SYSTEM+"\n\nAdditional learned guidance (from prior evaluations):\n"+G
try:
    sys_text
except NameError:
    sys_text=SYSTEM+"\n\nAdditional learned guidance (from prior evaluations):\n"+G
acts={}
def A(ep):
    if ep not in acts:
        ch=ep//info["chunks_size"]
        acts[ep]=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
    return acts[ep]
rows=[]
for ep in eps:
    aa=A(ep)
    ff=sorted(byep[ep]); idxs=np.linspace(0,len(ff)-1,7).astype(int)
    for f in [ff[i] for i in sorted(set(idxs))]:
        if f>=len(aa)-4: continue
        im=Image.open(f"{TIL}/ep{ep:04d}_f{f:03d}.png").convert("RGB")
        b=io.BytesIO(); im.save(b,format="JPEG",quality=90)
        if mode=="vis4":
            txt=(f"Task: {instr.get(ep,'')}\n{view_note}\n"+ask)     # 영상 전용: 액션 수치 제외
        else:
            txt=(f"Task: {instr.get(ep,'')}\n{view_note}\n"
                 "Planned actions for the next 16 control steps (7 numbers per step: EE delta x,y,z, rot x,y,z, gripper):\n"
                 +json.dumps(np.round(aa[f:f+16,5:],2).tolist())+"\n"+ask)
        rows.append({"custom_id":f"ep{ep:04d}_f{f:03d}","method":"POST","url":"/v1/chat/completions",
          "body":{"model":"gpt-5.6-luna","max_completion_tokens":(24 if mode in ("bits4","bits8","bits4cal","bits8cal","seq8","vis4") else 8),"reasoning_effort":"none",
                  "logprobs":True,"top_logprobs":K,
                  **({"n":3,"temperature":1.0} if mode=="vote" else {}),
                  "messages":[{"role":"system","content":[{"type":"text","text":sys_text}]},
                              {"role":"user","content":[
                                {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()}},
                                {"type":"text","text":txt}]}]}})
p=f"{OUT}/part_00.jsonl"
with open(p,"w") as fo:
    for r in rows: fo.write(json.dumps(r)+"\n")
json.dump([p], open(f"{OUT}/files.json","w"))
print(mode, "요청", len(rows), round(os.path.getsize(p)/1e6,1),"MB")
