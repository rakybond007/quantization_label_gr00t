"""로보카사 창 평균속도의 전체 분포를 재서 분위표로 굽는다.

facts_text 의 속도 줄은 0.12 / 0.50 이라는 절대 경계로 세 문구를 골랐다. 실제
분포를 재보니 그 경계가 하위 1.6% 와 46.6% 를 자른다 -- "barely moving" 은
1.6% 에서만 나오고 "moving fast" 가 과반(53.4%)에 붙는다. 세 갈래 중 하나는
죽어 있고 나머지는 반반이니 그 줄이 주는 정보가 거의 없다.

그래서 절대값 대신 분위를 준다. 이 스크립트가 굽는 101개 경계값으로
speed_mean 을 백분위로 옮기면, 다섯 구간이 각각 20% 씩 실제로 쓰인다.

    python vlm_gate/scripts/build_speed_percentiles.py [에피소드수]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
      "robocasa_mg_gr00t_300")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{BASE}/configs/robocasa_speed_percentiles.json"
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
WIN = 16      # descriptors() 의 창
STRIDE = 4    # 라벨을 뽑는 간격과 같게 -- 창을 균등하게 훑기만 하면 된다
TAIL = 4      # 라벨러가 액션 배열 끝에서 비워 두는 만큼

info = json.load(open(f"{DS}/meta/info.json"))
CS = info["chunks_size"]
rng = np.random.default_rng(0)
eps = rng.choice(7200, min(N_EP, 7200), replace=False)

S = []
ok = 0
for ep in eps:
    p = f"{DS}/data/chunk-{ep // CS:03d}/episode_{ep:06d}.parquet"
    try:
        a = np.stack(pd.read_parquet(p)["action"].values)
    except Exception:
        continue
    ok += 1
    # descriptors() 와 같은 정의: EE delta xyz 의 스텝 노름, 창 안 평균
    mag = np.linalg.norm(a[:, 5:8], axis=1)
    for f in range(0, max(len(a) - TAIL, 0), STRIDE):
        w = mag[f:f + WIN]
        if len(w):
            S.append(float(w.mean()))

S = np.asarray(S)
if len(S) < 10000:
    raise SystemExit(f"표본이 너무 적다: {len(S)}")
grid = [float(x) for x in np.percentile(S, np.arange(101))]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({"field": "speed_mean", "window": WIN, "stride": STRIDE,
           "n_episodes": ok, "n_windows": int(len(S)),
           "percentiles": grid}, open(OUT, "w"), indent=1)
print(f"에피 {ok}개 · 창 {len(S):,}개 -> {OUT}")
for q in (0, 10, 25, 50, 75, 90, 100):
    print(f"  {q:>3}% {grid[q]:.4f}")
