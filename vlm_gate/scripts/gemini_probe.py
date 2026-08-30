"""Gemini 3.7 Flash 대량 테스트 — 처리량·오류율·토큰 실측, 2회 분리 호출 중 영상 호출"""
import json, os, base64, urllib.request, time, sys, threading, queue
import numpy as np, pandas as pd
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
KEY=open(os.path.expanduser("~/quantization_agent_workspace/gemini_key")).read().strip()
N=int(sys.argv[1]) if len(sys.argv)>1 else 200
WORKERS=int(sys.argv[2]) if len(sys.argv)>2 else 4
TIL=f"{BASE}/output/_gate_distill/luna_robocasa_full/tiles"
frames=[]
for l in open(f"{BASE}/output/_gate_distill/exp_cp_s_vis4/labels.jsonl"):
    r=json.loads(l); frames.append((r['ep'],r['f']))
frames=frames[:N]
ASK=("Look ONLY at the camera views (3 panels: agentview-left, agentview-right, wrist close-up) and answer:\n"
     "  A) Is the gripper closing on an object or a handle right now, or opening to release one?\n"
     "  B) Is a carried object being precisely inserted, aligned, or lowered into a confined receptacle?\n"
     "  C) Is a door or drawer being PULLED OPEN with the grasped handle under load?\n"
     "  D) Is this plain gross motion - reaching, transporting a firmly held object, retracting, a broad "
     "sweep, or pressing a rigidly mounted button or knob?\n"
     "Answer with EXACTLY four characters, one per check in order A,B,C,D, each Y or N. No other text.")
out=open(f"{BASE}/output/_gate_distill/gemini_vis4.jsonl","w")
lock=threading.Lock(); stats={"ok":0,"err":0,"in":0,"out":0,"503":0}
q=queue.Queue()
for fr in frames: q.put(fr)
def work():
    while True:
        try: ep,f=q.get_nowait()
        except queue.Empty: return
        p=f"{TIL}/ep{ep:04d}_f{f:03d}.png"
        if not os.path.exists(p): q.task_done(); continue
        b=base64.b64encode(open(p,"rb").read()).decode()
        body={"contents":[{"parts":[{"inline_data":{"mime_type":"image/png","data":b}},{"text":ASK}]}],
              "generationConfig":{"maxOutputTokens":320,"temperature":0,"thinkingConfig":{"thinkingBudget":0}}}
        for attempt in range(3):
            try:
                req=urllib.request.Request(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent?key={KEY}",
                    data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
                r=json.load(urllib.request.urlopen(req, timeout=120))
                u=r.get("usageMetadata",{}); c=r["candidates"][0]
                parts=c.get("content",{}).get("parts") or []
                txt="".join(x.get("text","") for x in parts).strip().upper()
                ys=[ch for ch in txt if ch in "YN"][:4]
                with lock:
                    stats["in"]+=u.get("promptTokenCount",0)
                    stats["out"]+=u.get("candidatesTokenCount",0)+u.get("thoughtsTokenCount",0)
                    if len(ys)==4:
                        stats["ok"]+=1
                        out.write(json.dumps({"ep":ep,"f":f,"bits":[x=="Y" for x in ys]})+"\n")
                    else: stats["err"]+=1
                break
            except Exception as e:
                code=getattr(e,"code",0)
                with lock:
                    if code==503: stats["503"]+=1
                if attempt==2:
                    with lock: stats["err"]+=1
                time.sleep(2*(attempt+1))
        q.task_done()
t0=time.time()
ths=[threading.Thread(target=work,daemon=True) for _ in range(WORKERS)]
[t.start() for t in ths]; [t.join() for t in ths]
out.close(); el=time.time()-t0
print(f"요청 {len(frames)}건 / 성공 {stats['ok']} / 실패 {stats['err']} / 503재시도 {stats['503']}")
print(f"소요 {el:.0f}초 → 처리량 {stats['ok']/max(el,1)*60:.0f}건/분 (워커 {WORKERS})")
if stats["ok"]:
    pi=stats["in"]/max(stats["ok"],1); po=stats["out"]/max(stats["ok"],1)
    print(f"요청당 입력 {pi:.0f} 토큰 / 출력 {po:.1f} 토큰")
    print(f"262k 환산: 입력 {pi*262329/1e6:.0f}M, 출력 {po*262329/1e6:.1f}M")
    print(f"  전량 소요시간 추정: {262329/max(stats['ok']/max(el,1),1e-9)/3600:.1f}시간 (동일 워커 수)")
