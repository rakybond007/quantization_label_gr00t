"""부호 검증용 타일. 위험 풀과 안정 풀의 태스크가 **둘 다** 들어가게 고른다.

기존 `gen_libero_tiles_shard.py` 는 에피소드를 번호로 나누므로, 앞쪽 몇 개만
만들면 한 수트에 몰린다. 실제로 190장을 만들었더니 전부 안정 풀이라 부호를
아예 못 봤다. 여기서는 지시문으로 태스크를 맞춘 뒤 두 풀에서 고르게 뽑는다.

    python vlm_gate/scripts/gen_libero_tiles_pools.py [태스크당_에피수] [스트라이드]
"""
import json, os, sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from derive_libero_questions import INSTR

DS = "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta"
BASE = "/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
OUT = f"{BASE}/output/_gate_distill/libero_pools"
MAN = f"{BASE}/output/_gate_distill/libero_pools_manifest.txt"
VK = ["observation.images.front_view", "observation.images.left_wrist_view"]

RISK = ["long/1", "spatial/3", "spatial/0", "goal/1", "spatial/5", "spatial/8",
        "goal/3", "goal/9", "long/8"]
STABLE = ["object/0", "object/7", "goal/5", "long/3", "object/9", "goal/4",
          "object/5", "object/3", "goal/2", "goal/7", "long/5", "long/7"]

NEP = int(sys.argv[1]) if len(sys.argv) > 1 else 2
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 12


def main():
    from decord import VideoReader
    info = json.load(open(f"{DS}/meta/info.json"))
    by_text = {v.strip().lower(): k for k, v in INSTR.items()}
    task_eps = {}
    for l in open(f"{DS}/meta/episodes.jsonl"):
        d = json.loads(l)
        for t in d.get("tasks", []):
            if isinstance(t, str) and t.strip().lower() in by_text:
                task_eps.setdefault(by_text[t.strip().lower()], []).append(d["episode_index"])
                break
    os.makedirs(f"{OUT}/tiles", exist_ok=True)
    names, per = [], {}
    for tk in RISK + STABLE:
        eps = sorted(task_eps.get(tk, []))[:NEP]
        if not eps:
            print(f"[!] {tk}: 에피소드 못 찾음"); continue
        cnt = 0
        for ep in eps:
            ch = ep // info["chunks_size"]
            try:
                vrs = [VideoReader(f"{DS}/" + info["video_path"].format(
                    episode_chunk=ch, episode_index=ep, video_key=k)) for k in VK]
            except Exception as e:
                print(f"ep{ep}: {e}", flush=True); continue
            n = min(len(v) for v in vrs)
            for fi in range(0, n, STRIDE):
                p = f"{OUT}/tiles/ep{ep:04d}_f{fi:03d}.png"
                if not os.path.exists(p):
                    t = np.concatenate([v[fi].asnumpy() for v in vrs], axis=1)
                    Image.fromarray(t).resize(
                        (t.shape[1] // 2, t.shape[0] // 2)).save(p)
                names.append(os.path.basename(p)); cnt += 1
        per[tk] = cnt
        print(f"  {tk:<12} 에피 {eps} -> 타일 {cnt}", flush=True)
    open(MAN, "w").write("\n".join(sorted(set(names))) + "\n")
    nr = sum(per.get(t, 0) for t in RISK)
    ns = sum(per.get(t, 0) for t in STABLE)
    print(f"\n타일 {len(set(names))}장  (위험 {nr} · 안정 {ns})  -> {MAN}")


if __name__ == "__main__":
    main()
