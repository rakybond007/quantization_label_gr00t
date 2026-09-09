"""학습에 실제로 쓰일 행 수. 라벨과 프레임 캐시의 교집합이다."""
import glob, os
import pandas as pd
W  = os.path.expanduser("~/quantization_agent_workspace")
WS = f"{W}/vlm_gate"
lab_old = f"{W}/assets/labels/robocasa/phase9_cached.parquet"
cache = f"{W}/assets/frame_cache_robocasa"

keep = set()
for p in sorted(glob.glob(f"{cache}/index_shard*.parquet")):
    d = pd.read_parquet(p)
    keep |= set(zip(d.episode_index.tolist(), d.frame_index.tolist()))
print(f"프레임 캐시           {len(keep):,}")

if os.path.exists(lab_old):
    d = pd.read_parquet(lab_old)
    k = set(zip(d.episode_index.tolist(), d.frame_index.tolist()))
    print(f"이전 라벨(phase9_cached) {len(d):,}  캐시와 교집합 {len(k & keep):,}")

for nm in ("robocasa_v1_ratio", "robocasa_contact_ratio"):
    p = f"{WS}/output/_gate_distill/{nm}.parquet"
    if not os.path.exists(p):
        print(f"{nm}: 없음"); continue
    d = pd.read_parquet(p, columns=["episode_index", "frame_index", "ratio", "fixed"])
    k = set(zip(d.episode_index.tolist(), d.frame_index.tolist()))
    inter = k & keep
    print(f"{nm:24s} {len(d):,}  캐시와 교집합 {len(inter):,} = {len(inter)/len(d):.1%}")
    sub = d[[(e, f) in keep for e, f in zip(d.episode_index, d.frame_index)]]
    if len(sub):
        print("   그 교집합의 배속 분포 " +
              "  ".join(f"{v}x:{(sub.ratio==v).mean():.1%}" for v in sorted(sub.ratio.unique())))
