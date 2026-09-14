"""평가 롤아웃의 초기 장면을 뽑는다. 배속 예측의 관측이다.

**왜 초기 프레임인가.** 에피 중간 프레임은 로봇을 따라 크게 움직이는 시점이라
팔이 화면을 덮는 칸이 많고 에피끼리 비교가 안 된다. 첫 프레임은 모든 에피에서
같은 시점의 정지 장면이고, 에피마다 달라지는 것(주방 외형·대상 물체·배치)이
거기 다 있다. 압축 내성이 갈리는 원인도 대부분 거기 있다 -- 정책은 장면이
정해지면 거의 결정적이다.
"""
import os
import sys

import av
import numpy as np
import pandas as pd
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = f"{BASE}/output/robocasa/baseline_full_v2_with_action_steps"
OUT = f"{BASE}/output/_gate_distill/ratio_frames"
NF = int(os.environ.get("NFRAME", "1"))          # 에피마다 몇 장 (첫 장 기준)


def main():
    T = pd.read_parquet(f"{BASE}/analysis/ratio_prompt/episode_split.parquet")
    T.columns = [c if c != 0 else "max_ratio" for c in T.columns]
    os.makedirs(OUT, exist_ok=True)
    done = miss = 0
    for r in T.itertuples():
        t, ep = r.task, int(r.ep)
        dst = f"{OUT}/{t}__ep{ep:03d}.png"
        if os.path.exists(dst):
            done += 1; continue
        p = f"{RUN}/{t}/{t}-episode_{ep}.mp4"
        try:
            with av.open(p) as c:
                fr = [f.to_ndarray(format="rgb24")
                      for i, f in enumerate(c.decode(video=0)) if i < NF]
            if not fr:
                miss += 1; continue
            im = fr[0] if NF == 1 else np.concatenate(fr, axis=1)
            Image.fromarray(im).save(dst)
            done += 1
        except Exception as e:
            miss += 1
            if miss <= 3:
                print(f"  {t} ep{ep}: {e}", flush=True)
        if done % 100 == 0:
            print(f"  {done}/{len(T)}", flush=True)
    print(f"완료 {done} · 실패 {miss} -> {OUT}")


if __name__ == "__main__":
    main()
