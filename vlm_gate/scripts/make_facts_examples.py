"""각 벤치마크의 **실제 데이터**에서 계산 사실 예시를 뽑는다.

손으로 붙여넣으면 코드가 바뀌었을 때 조용히 어긋난다. 실제로 allex 예시는
"call allex_facts.py to see them" 이라는 빈 껍데기였다.

출력은 레포의 `prompts/<bench>_<ver>_facts_example.txt` 로 간다. 한 벤치마크에서
서로 다른 순간 셋을 뽑아 조건에 따라 문장이 어떻게 갈리는지 보이게 한다.

  python make_facts_examples.py <출력디렉터리>
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{BASE}/scripts")


def allex():
    from allex_facts import facts
    from allex_v2_common import descriptors
    DS = "/rlwrld2/home/david/frontier_demo_cumul/v1_v2_v3_v4"
    f = sorted(glob.glob(f"{DS}/data/chunk-000/*.parquet"))[0]
    d = pd.read_parquet(f)
    A = np.stack(d["action"].values)
    WR = np.stack(d["action.right_wrist_wrt_base"].values)
    WL = np.stack(d["action.left_wrist_wrt_base"].values)
    ep = int(f.split("episode_")[1][:6])
    return [(f"에피 {ep} · 프레임 {fr}", facts(descriptors(A, WR, WL, fr, 16)))
            for fr in (0, len(A) // 3, 2 * len(A) // 3)]


def _robocasa(mod):
    D = __import__(mod)
    DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
          "robocasa_mg_gr00t_300")
    info = json.load(open(f"{DS}/meta/info.json"))
    ep = 52
    ch = ep // info["chunks_size"]
    a = np.stack(pd.read_parquet(
        f"{DS}/data/chunk-{ch:03d}/episode_{ep:06d}.parquet")["action"].values)
    return [(f"에피 {ep} · 프레임 {fr}", D.facts_text(D.descriptors(a, fr)))
            for fr in (0, len(a) // 3, 2 * len(a) // 3)]


def robocasa_v7():
    return _robocasa("robocasa_descriptors")


def robocasa_v8():
    return _robocasa("robocasa_v8_descriptors")


def libero():
    import libero_descriptors as D
    DS = ("/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/"
          "libero_gr00t_delta")
    f = sorted(glob.glob(f"{DS}/data/chunk-000/*.parquet"))[0]
    a = np.stack(pd.read_parquet(f)["action"].values)
    ep = int(f.split("episode_")[1][:6])
    return [(f"에피 {ep} · 프레임 {fr}", D.facts_text(D.descriptors(a, fr)))
            for fr in (0, len(a) // 3, 2 * len(a) // 3)]


MAKERS = {("allex", "v4c"): allex, ("robocasa", "v7"): robocasa_v7,
          ("robocasa", "v8"): robocasa_v8, ("robocasa", "v9"): robocasa_v8,
          ("robocasa", "v10"): robocasa_v8, ("libero", "v3c"): libero}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else f"{BASE}/prompts"
    for (name, ver), fn in MAKERS.items():
        try:
            rows = fn()
        except Exception as e:
            print(f"  {name} {ver}: 실패 {type(e).__name__} {e}")
            continue
        txt = (f"# {name} {ver} 계산 사실 예시 -- **실제 데이터에서 뽑았다**\n"
               f"# make_facts_examples.py 가 만든다. 손으로 고치지 말 것.\n"
               f"# 전문의 {{computed facts}} 자리에 이런 문장이 들어간다.\n\n")
            
        for tag, s in rows:
            txt += f"[{tag}]\n{s}\n\n"
        open(f"{out}/{name}_{ver}_facts_example.txt", "w").write(txt)
        print(f"  {name} {ver}: {len(rows)}개 -> {name}_{ver}_facts_example.txt")


if __name__ == "__main__":
    main()
