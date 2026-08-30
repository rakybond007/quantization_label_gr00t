"""프로브용 부분집합 인덱스를 만든다 (pandas 있는 환경에서 실행).
인코더 추출기는 pandas 없는 venv 에서도 돌아야 하므로 npz 로 넘긴다."""
import os, sys, numpy as np, pandas as pd
WS=os.path.expanduser("~/quantization_agent_workspace"); BASE=f"{WS}/vlm_gate"
MAN=f"{BASE}/output/_gate_distill/tiles_manifest.txt"
LAB=sys.argv[1] if len(sys.argv)>1 else f"{WS}/assets/labels/robocasa/v6b_phase5_1call_full.parquet"
N=int(sys.argv[2]) if len(sys.argv)>2 else 40000
OUT=sys.argv[3] if len(sys.argv)>3 else f"{WS}/assets/probe_features/index.npz"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
lab=pd.read_parquet(LAB, columns=["episode_index","frame_index","p_yes"])
key={(int(r.episode_index),int(r.frame_index)):float(r.p_yes) for r in lab.itertuples()}
names=[n for n in open(MAN).read().split() if (int(n[2:6]), int(n.split("_f")[1][:3])) in key]
rng=np.random.default_rng(0); rng.shuffle(names); names=sorted(names[:N])
ep=np.array([int(n[2:6]) for n in names]); fr=np.array([int(n.split("_f")[1][:3]) for n in names])
y=np.array([key[(e,f)] for e,f in zip(ep,fr)], dtype=np.float32)
np.savez(OUT, names=np.array(names), ep=ep, fr=fr, y=y)
print(f"[index] {len(names)}장 / 에피소드 {len(set(ep.tolist()))}개 -> {OUT}")
