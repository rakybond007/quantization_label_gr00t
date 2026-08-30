"""실험 매트릭스 레지스트리: 도메인 x 티처 x 아키텍처 x (라벨/학습/폐루프/최적τ)"""
import json, os, glob, re
import numpy as np
BASE="/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
CKR="/rlwrld-unified-checkpoints/hojin2/checkpoints"
def n_parquet(p):
    try:
        import pandas as pd; return len(pd.read_parquet(p))
    except Exception: return 0
def n_jsonl(p):
    try: return sum(1 for _ in open(p))
    except Exception: return 0
def closed_loop(d):
    s=n=0; steps=[]
    for f in glob.glob(f"{BASE}/output/robocasa/{d}/*/prediction.txt"):
        for m in re.finditer(r"is_success: \[\s*(\w+)\] action_steps: (\d+)", open(f).read()):
            n+=1; ok=m.group(1)=="True"; s+=ok
            if ok: steps.append(int(m.group(2)))
    if not n: return None
    return {"success": round(s/n,3), "steps": round(float(np.mean(steps)),1), "episodes": n}
R={"labels":{}, "students":{}, "closed_loop":{}, "best_tau":{}, "gaps":[]}
# ---- 라벨 ----
R["labels"]["robocasa/cosmos3"]={"path":f"{CKR}/gate_distill_robocasa_cosmos_v1/labels/full_merged.parquet",
                                 "rows":n_parquet(f"{CKR}/gate_distill_robocasa_cosmos_v1/labels/full_merged.parquet"), "stride":8}
R["labels"]["robocasa/gemma4"]={"path":f"{CKR}/gate_distill_robocasa_gemma_v2/labels/gemma_full_merged.parquet",
                                "rows":n_parquet(f"{CKR}/gate_distill_robocasa_gemma_v2/labels/gemma_full_merged.parquet"), "stride":8}
fr_even=n_jsonl(f"{BASE}/output/_gate_distill/exp_s16_act/labels.jsonl")
fr_odd=n_jsonl(f"{BASE}/output/_gate_distill/exp_s8_odd_act/labels.jsonl")
R["labels"]["robocasa/frontier"]={"path":"exp_s16_act + exp_s8_odd_act", "rows_even":fr_even, "rows_odd":fr_odd,
                                  "target":260031, "stride":8, "status":"진행중"}
for k,p in [("real/cosmos3", f"{CKR}/gate_distill_real_droid_pnp_v1/labels/real_cosmos_full.parquet"),
            ("real/luna_batch6(폐기)", f"{CKR}/gate_distill_real_droid_pnp_v1/labels/real_luna_full.parquet")]:
    R["labels"][k]={"path":p,"rows":n_parquet(p)}
R["labels"]["real/cosmos3_phase_v2"]={"path":f"{BASE}/output/_gate_distill/cosmos_real_tiles_phase_v2.parquet",
                                      "rows":n_parquet(f"{BASE}/output/_gate_distill/cosmos_real_tiles_phase_v2.parquet")}
R["labels"]["real/frontier_api_act"]={"path":f"{BASE}/output/_gate_distill/exp_real_api_act/labels.jsonl",
                                      "rows":n_jsonl(f"{BASE}/output/_gate_distill/exp_real_api_act/labels.jsonl"), "target":3705}
R["labels"]["real/gemma4"]={"status":"미실시 (GAP)"}
# ---- 학생/모델 ----
for name,path in [("robocasa/cosmos3/A'", f"{CKR}/gate_distill_robocasa_cosmos_v1/module_A_tc_full/gate_module.pt"),
                  ("robocasa/gemma4/A'", f"{CKR}/gate_distill_robocasa_cosmos_v1/module_A_gemma_full_restored/gate_module.pt"),
                  ("robocasa/frontier/A'", f"{CKR}/gate_distill_robocasa_cosmos_v1/module_A_frontier_full/gate_module.pt"),
                  ("robocasa/cosmos3/C", f"{CKR}/gate_distill_robocasa_cosmos_v1/vla_gateC_v3_lam03_60k/checkpoint-60000"),
                  ("robocasa/gemma4/C", f"{CKR}/gate_distill_robocasa_cosmos_v1/vla_gateC_60k/checkpoint-60000"),
                  ("real/luna_batch6/C", f"{CKR}/gate_distill_real_droid_pnp_v1/vla_gateC_real_25k/checkpoint-25000"),
                  ("real/cosmos3/C", f"{CKR}/gate_distill_real_droid_pnp_v1/vla_gateC_real_cosmos_25k/checkpoint-25000"),
                  ("real/luna_batch6/A'", f"{CKR}/gate_distill_real_droid_pnp_v1/module_A_luna_25kstep/gate_module.pt"),
                  ("real/plain_VLA", f"{CKR}/gate_distill_real_droid_pnp_v1/vla_plain_real_25k/checkpoint-25000")]:
    R["students"][name]={"path":path, "exists":os.path.exists(path)}
    if not os.path.exists(path): R["gaps"].append(f"학습 필요: {name}")
# ---- 폐루프 ----
for name,d in [("robocasa/gemma4/A'@tau0.5","moduleA_gemma_gated"),
               ("robocasa/cosmos3/A'@tau0.5","moduleA_cosmos_gated"),
               ("robocasa/cosmos3/A'@tau0.42","moduleA_cosmos_tau042"),
               ("robocasa/cosmos3/A'@tau0.38","moduleA_cosmos_tau038"),
               ("robocasa/gemma4/B@tau0.5","moduleB_gemma_gated"),
               ("robocasa/cosmos3/B@tau0.5","moduleB_cosmos_gated"),
               ("robocasa/cosmos3/C_v3lam03@tau0.5","gateC_v3_lam03_internal_gated"),
               ("robocasa/cosmos3/C_v3@tau0.5","gateCv3_internal_gated"),
               ("robocasa/frontier9k/A'@tau0.5","moduleA_f9k_act_gated"),
               ("robocasa/cosmos9k/A'@tau0.5","moduleA_cosmos9k_gated"),
               ("robocasa/frontier_full/A'@tau0.5","moduleA_frontier_full_gated")]:
    r=closed_loop(d)
    if r: R["closed_loop"][name]=r
R["best_tau"]={"주의":"티처x아키텍처마다 다름. evolve로 탐색해야 하는 값",
               "robocasa/cosmos3/A'":"탐색중 (0.5=0.659/299, 0.42/0.38 대기)",
               "robocasa/gemma4/A'":"0.5 기준 0.667/258 (미탐색)",
               "robocasa/cosmos3/C":"0.5 기준 0.638~0.647 (미탐색)"}
for gap in ["real/gemma4 라벨링 미실시","real/frontier A' 학생 미학습","real/cosmos3 A' 학생 미학습",
            "robocasa/frontier C 미학습","real 도메인 폐루프 평가 수단 없음(오프라인 지표만)"]:
    R["gaps"].append(gap)
json.dump(R, open(f"{BASE}/analysis/EXPERIMENT_REGISTRY.json","w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1)[:2600])
