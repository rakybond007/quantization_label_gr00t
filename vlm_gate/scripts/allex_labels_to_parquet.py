"""판정 records.jsonl 을 배포용 parquet 으로 굽는다.

`prehj/robocasa-ratio-labels-contact` 의 구조를 따른다:

    labels/<name>.parquet   프레임마다 한 행
    checks/<name>.py        부호·가중치 모듈 사본
    prompts/<name>.txt      조립된 프롬프트 전문
    README.md

열 (robocasa 릴리스와 맞춘다)

    episode_index int32 · frame_index int32 · A..D int8 · conf float32
    instruction object · valid int8

robocasa 에 있고 여기 없는 것:
  fixed      접촉 라벨을 쓰지 않았다 (frontier_demo_cumul 에 subtasks.jsonl 이 없다)
  eA..eE     Gemini 는 등급 확률을 안 준다 (logprobs 가 HTTP 400) -- 정수 등급뿐이라
             기댓값 등급이 없다. 그래서 conf 가 43단 계단이다.
  ratio      단일 배속 게이트라 태스크별 상한이 없다. K 는 역치를 정한 뒤 붙인다.

`conf` 는 청크 단위 값을 그 청크의 프레임 전체에 깐다(hold). **램프를 걸지 않는다** --
램프는 K 가 재생속도로 소비될 때 저크를 막는 장치이고, 회귀 타깃에 넣으면 라벨을
흐린다. 꼬리 프레임(청크가 에피 끝을 넘어가는 구간)은 valid=0 이다.

    python allex_labels_to_parquet.py <records.jsonl> <dataset> <out.parquet> [--chunk 16]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("dataset", help="프레임 수와 지시문을 읽을 LeRobot 데이터셋")
    ap.add_argument("out")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--questions", default="ABCD")
    args = ap.parse_args()

    by_ep = {}
    for line in open(args.records):
        try:
            r = json.loads(line)
        except Exception:
            continue
        by_ep.setdefault(int(r["ep"]), {})[int(r["f"])] = r

    tasks = {}
    with open(f"{args.dataset}/meta/tasks.jsonl") as f:
        for line in f:
            t = json.loads(line)
            tasks[t["task_index"]] = t["task"]
    lens, ep_task = {}, {}
    with open(f"{args.dataset}/meta/episodes.jsonl") as f:
        for line in f:
            e = json.loads(line)
            lens[e["episode_index"]] = e["length"]
            ts = [t for t in e.get("tasks", []) if isinstance(t, str)]
            ep_task[e["episode_index"]] = ts[0] if ts else ""

    Q = list(args.questions)
    rows = []
    for ep in sorted(lens):
        n = lens[ep]
        ch = by_ep.get(ep)
        if not ch:
            print(f"  [!] ep{ep} 라벨 없음 -- 건너뛴다")
            continue
        grade = {q: np.zeros(n, dtype=np.int8) for q in Q}
        conf = np.zeros(n, dtype=np.float32)
        valid = np.zeros(n, dtype=np.int8)
        for f, r in ch.items():
            w = slice(f, min(f + args.chunk, n))
            for q in Q:
                grade[q][w] = int(r[q])
            conf[w] = float(r["conf"])
            valid[w] = 1
        # 꼬리: 마지막 청크가 덮지 못한 구간. 값은 마지막 청크를 물려받되 valid=0 이다
        last = max(ch)
        tail = min(last + args.chunk, n)
        if tail < n:
            for q in Q:
                grade[q][tail:] = int(ch[last][q])
            conf[tail:] = float(ch[last]["conf"])
        d = {"episode_index": np.full(n, ep, dtype=np.int32),
             "frame_index": np.arange(n, dtype=np.int32)}
        d.update({q: grade[q] for q in Q})
        d["conf"] = conf
        d["valid"] = valid
        d["instruction"] = [ep_task.get(ep, "")] * n
        rows.append(pd.DataFrame(d))

    df = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df.to_parquet(args.out, compression="zstd", index=False)

    v = df.conf[df.valid == 1]
    print(f"{len(df):,}행 · 에피 {df.episode_index.nunique()} · "
          f"유효 {int(df.valid.sum()):,} ({df.valid.mean():.1%})")
    print(f"  conf {v.min():.3f}~{v.max():.3f} 평균 {v.mean():.4f} · "
          f"서로 다른 값 {v.round(4).nunique()}개")
    for q in Q:
        print(f"  {q}: 평균 {df[q][df.valid == 1].mean():.2f}")
    print(f"  dtype: " + " · ".join(f"{c}:{df[c].dtype}" for c in df.columns))
    print(f"-> {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
