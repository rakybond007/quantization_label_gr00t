"""티처×도메인 2x2 내재적 품질 평가 (cosmos는 정답이 아님; 물리 신호 기준)
지표: (1) 그리퍼 개폐 구간 판별 AUC/억제율 (2) 접촉 구간 판별 (3) 인접 프레임 일관성(잔차) (4) qrate/분포"""
import json, glob, os, sys
import numpy as np, pandas as pd
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
def auc(y,s):
    y=np.asarray(y); s=np.asarray(s)
    o=np.argsort(s); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    n1=y.sum(); n0=len(y)-n1
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0) if n1*n0>0 else float('nan')

# ---------- 도메인별 물리 신호 ----------
def real_signals():
    DS="/sjw_alinlab/home/hojin2/taekwan/Isaac-GR00T/Data/human_data/MoSS/lerobot/pnp_objects"
    sig={}
    for f in sorted(glob.glob(f"{DS}/data/chunk-000/episode_*.parquet")):
        ep=int(f.split("_")[-1].split(".")[0]); d=pd.read_parquet(f)
        st=np.stack(d["observation.state"].values); g=st[:,7]
        tac=np.abs(np.stack(d["observation.tactile.right"].values)).mean(1)
        tq=np.abs(np.stack(d["observation.torque"].values)).mean(1)
        gd=np.abs(np.diff(g, prepend=g[0]))
        tr=gd > (0.05*max(g.max(),1e-6))
        near=np.zeros(len(g),bool)
        for t in np.where(tr)[0]: near[max(0,t-5):t+6]=True
        contact = tac > np.percentile(tac,70)          # 촉각 상위 30% = 접촉
        highT   = tq  > np.percentile(tq,70)
        # 실기: 그리퍼 토글 ±2 | 관절 속도 반전 | 촉각 상위30%+저속
        vel=np.stack(d["observation.velocity"].values)[:,:7]
        sp=np.linalg.norm(vel,axis=1)
        rev=np.concatenate([[False], np.sum(vel[:-1]*vel[1:],axis=1)<0])
        tog=np.zeros(len(g),bool)
        for t in np.where(tr)[0]: tog[max(0,t-2):t+3]=True
        fine=contact & (sp<np.percentile(sp,25))
        sig[ep]={"grip":near,"contact":contact,"risk":tog|rev|fine,"torque":highT,"n":len(g)}
    return sig
def rc_signals():
    DS="/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
    info=json.load(open(f"{DS}/meta/info.json")); sig={}
    return DS, info, sig
def rc_sig_ep(DS, info, ep, cache={}):
    if ep in cache: return cache[ep]
    ch=ep//info["chunks_size"]
    try: a=np.stack(pd.read_parquet(f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
    except Exception: return None
    grip=a[:,-1]; d=a[:,5:8]
    gd=np.abs(np.diff(grip, prepend=grip[0]))
    near=np.zeros(len(a),bool)
    for t in np.where(gd>0.5)[0]: near[max(0,t-8):t+9]=True
    # 접촉 근사: 그리퍼 닫힘 상태 + 저속 (물체 조작/삽입 국면)
    spd=np.linalg.norm(d,axis=1)
    closed = grip > np.median(grip)
    slow = spd < np.percentile(spd,40)
    # K2 병합이 실제로 손상을 주는 프레임: 다음 2스텝 내 그리퍼 토글 | 방향반전 | 닫힘+저속(정밀조작)
    tog=np.zeros(len(a),bool)
    for t in np.where(gd>0.5)[0]: tog[max(0,t-2):t+3]=True
    nrm=np.linalg.norm(d,axis=1)+1e-9
    cosang=np.sum(d[:-1]*d[1:],axis=1)/(nrm[:-1]*nrm[1:])
    rev=np.concatenate([[False], cosang<0.0])
    fine=(grip>np.median(grip)) & (spd<np.percentile(spd,25))
    cache[ep]={"grip":near,"contact":closed&slow,"risk":tog|rev|fine,"torque":None,"n":len(a)}
    return cache[ep]

def resid_consistency(d, stride):
    byep={}
    for (ep,f),p in d.items(): byep.setdefault(ep,{})[f]=p
    res={}
    for ep,v in byep.items():
        if len(v)<5: continue
        m=np.mean(list(v.values()))
        for f,p in v.items(): res[(ep,f)]=p-m
    a=[];b=[]
    for (ep,f),p in res.items():
        q=res.get((ep,f+stride))
        if q is not None: a.append(p); b.append(q)
    return np.corrcoef(a,b)[0,1] if len(a)>100 else float('nan')

def evaluate(d, dom, name):
    stride = 4 if dom=="real" else 8
    if dom=="real": SIG=real_signals()
    else: DS, info, _ = rc_signals()
    ys={"grip":[], "contact":[], "risk":[]}; ss={"grip":[], "contact":[], "risk":[]}
    for (ep,f),p in d.items():
        s = SIG.get(ep) if dom=="real" else rc_sig_ep(DS, info, ep)
        if s is None or f>=s["n"]: continue
        ys["grip"].append(int(s["grip"][f])); ss["grip"].append(1-p)
        ys["contact"].append(int(s["contact"][f])); ss["contact"].append(1-p)
        ys["risk"].append(int(s["risk"][f])); ss["risk"].append(1-p)
    v=np.array(list(d.values()))
    gA=auc(ys["grip"], ss["grip"]); cA=auc(ys["contact"], ss["contact"]); rA=auc(ys["risk"], ss["risk"])
    yg=np.array(ys["grip"]); sg=np.array(ss["grip"])
    yes_on = np.mean(1-sg[yg==1] >= 0.5) if (yg==1).any() else float('nan')   # 그리퍼 구간 YES율
    yes_off= np.mean(1-sg[yg==0] >= 0.5) if (yg==0).any() else float('nan')
    return dict(n=len(d), qrate=float(np.mean(v>=0.5)), std=float(v.std()),
                gripAUC=gA, contactAUC=cA, riskAUC=rA, yes_grip=yes_on, yes_other=yes_off,
                consist=resid_consistency(d, stride))
def load_jsonl(p):
    d={}
    if not os.path.exists(p): return d
    for l in open(p):
        try: r=json.loads(l); d[(r['ep'],r['f'])]=r['p_yes']
        except: pass
    return d
def load_parquet(p, ec="episode_index", fc="frame_index"):
    if not os.path.exists(p): return {}
    c=pd.read_parquet(p); return dict(zip(zip(c[ec].astype(int),c[fc].astype(int)), c['p_yes']))
if __name__=="__main__":
    ENTRIES=[]
    # ---- robocasa ----
    ENTRIES.append(("robocasa","cosmos 로컬 (sim-evolved 프롬프트)", load_parquet("/rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_robocasa_cosmos_v1/labels/full_merged.parquet")))
    gem=pd.concat([pd.read_parquet(f) for f in glob.glob("/rlwrld-unified-checkpoints/hojin2/checkpoints/gate_distill_robocasa_gemma_v2/labels/strat240_shard*.parquet")])
    ENTRIES.append(("robocasa","gemma 로컬", dict(zip(zip(gem['episode_index'].astype(int),gem['frame_index'].astype(int)), gem['p_yes']))))
    for tag,lab in [("fair_rc_none","frontier (sim 프롬프트 그대로)"),("fp_v1_counterbias","frontier +counterbias"),
                    ("fp_v2_rubric","frontier +rubric"),("fp_v3_ratehint","frontier +ratehint"),
                    ("rc_act","frontier +액션수치")]:
        ENTRIES.append(("robocasa", lab, load_jsonl(f"{BASE}/output/_gate_distill/exp_{tag}/labels.jsonl")))
    # ---- real ----
    ENTRIES.append(("real","cosmos 로컬 (sim-evolved+실기패치)", load_parquet(f"{BASE}/output/_gate_distill/cosmos_real_tiles_base.parquet")))
    ENTRIES.append(("real","cosmos +counterbias", load_parquet(f"{BASE}/output/_gate_distill/cosmos_real_tiles_v1_counterbias.parquet")))
    ENTRIES.append(("real","구독 luna (정지,6묶음)", load_jsonl(f"{BASE}/output/_gate_distill/luna_real_full/labels_luna.jsonl")))
    for tag,lab in [("real_motion","frontier 모션3f"),("real_motion_act","frontier 모션3f+액션"),("fair_real_none","frontier 공정세팅(정지)")]:
        ENTRIES.append(("real", lab, load_jsonl(f"{BASE}/output/_gate_distill/exp_{tag}/labels.jsonl")))
    print(f"{'도메인':8s} {'티처':34s} {'n':>5s} {'qrate':>6s} {'그리퍼AUC':>8s} {'접촉AUC':>7s} {'K2위험AUC':>9s} {'YES(그리퍼)':>10s} {'YES(그외)':>9s} {'일관성':>6s}")
    for dom,lab,d in ENTRIES:
        if not d: print(f"{dom:8s} {lab:34s} (없음)"); continue
        r=evaluate(d,dom,lab)
        print(f"{dom:8s} {lab:34s} {r['n']:5d} {r['qrate']:6.2f} {r['gripAUC']:8.3f} {r['contactAUC']:7.3f} {r['riskAUC']:9.3f} {r['yes_grip']:10.2f} {r['yes_other']:9.2f} {r['consist']:6.3f}")
