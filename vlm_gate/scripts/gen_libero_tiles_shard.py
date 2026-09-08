"""LIBERO 타일 생성 — 로보카사 gen_robocasa_tiles_shard.py 와 같은 모양.

차이 두 가지.
  * 뷰가 셋이 아니라 둘이다: front_view(정면) + left_wrist_view(손목).
    타일은 가로로 이어 붙인 256x512 -> 2배 축소 128x256.
  * 전량 라벨링은 stride 1로 모든 프레임을 만든다. `TILE_STRIDE`를 명시하면
    작은 진단용 표본의 간격만 바꿀 수 있다.

사용법:  python gen_libero_tiles_shard.py <shard> <nshards>
        python gen_libero_tiles_shard.py merge <nshards>   # 샤드 JSON -> 평문 매니페스트
"""
import io
import json
import os
import sys

import numpy as np
from PIL import Image

DS = os.environ.get(
    "LIBERO_DATASET",
    "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/libero_gr00t_delta",
)
BASE = os.environ.get(
    "VLM_GATE_ROOT", "/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate"
)
OUT = os.environ.get("LIBERO_TILE_OUT", f"{BASE}/output/_gate_distill/libero_full")
MANIFEST = os.environ.get("LIBERO_MANIFEST", f"{BASE}/output/_gate_distill/libero_tiles_manifest.txt")
# 전 프레임 라벨링이 기본이다. VLA 데이터로더는 모든 시점을 도는데 라벨이
# 간격마다만 있으면 그 시점들은 gate_valid=0 으로 손실에서 빠지고, 합동 학습에서
# 게이트가 받는 감독이 그만큼 성겨진다. STRIDE 로 줄일 수는 있게 남겨둔다.
STRIDE = int(os.environ.get("TILE_STRIDE", "1"))
LIMIT = int(os.environ.get("TILE_LIMIT", "0"))
# 정면 먼저, 손목 나중 — 라벨러가 왼쪽 절반을 정면으로 읽는다.
VK = ["observation.images.front_view", "observation.images.left_wrist_view"]


def merge(nsh):
    """샤드 매니페스트 JSON 을 파일명 한 줄짜리 평문 매니페스트로 합친다."""
    names = []
    for s in range(nsh):
        p = f"{OUT}/manifest_shard{s}.json"
        if not os.path.exists(p):
            raise FileNotFoundError(f"필수 샤드 매니페스트가 없다: {p}")
        names += [os.path.basename(r["path"]) for r in json.load(open(p))]
    names = sorted(set(names))
    with open(MANIFEST, "w") as f:
        f.write("\n".join(names) + "\n")
    print(f"manifest: {len(names)} tiles -> {MANIFEST}")


def embedded_image(cell):
    """LeRobot parquet의 image 셀(bytes/path)을 PIL 이미지로 연다."""
    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
        cell = cell.get("path")
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(cell))).convert("RGB")
    if isinstance(cell, str):
        path = cell if os.path.isabs(cell) else os.path.join(DS, cell)
        return Image.open(path).convert("RGB")
    if isinstance(cell, Image.Image):
        return cell.convert("RGB")
    raise TypeError(f"지원하지 않는 image 셀 형식: {type(cell).__name__}")


def main():
    if sys.argv[1] == "merge":
        return merge(int(sys.argv[2]))
    shard, nsh = int(sys.argv[1]), int(sys.argv[2])
    info = json.load(open(f"{DS}/meta/info.json"))
    ntotal = info["total_episodes"]
    os.makedirs(f"{OUT}/tiles", exist_ok=True)
    eps = [e for e in range(ntotal) if e % nsh == shard]
    man = []
    for i, ep in enumerate(eps):
        ch = ep // info["chunks_size"]
        if all(info["features"][k]["dtype"] == "image" for k in VK):
            import pandas as pd
            table = pd.read_parquet(f"{DS}/" + info["data_path"].format(
                episode_chunk=ch, episode_index=ep), columns=VK)
            n = len(table)
            get_frame = lambda k, fi: np.asarray(embedded_image(table[k].iloc[fi]))
        else:
            from decord import VideoReader
            vrs = [VideoReader(f"{DS}/" + info["video_path"].format(
                episode_chunk=ch, episode_index=ep, video_key=k)) for k in VK]
            lengths = [len(v) for v in vrs]
            if len(set(lengths)) != 1:
                raise RuntimeError(f"ep{ep}: view 길이가 서로 다르다: {lengths}")
            n = lengths[0]
            get_frame = lambda k, fi: vrs[VK.index(k)][fi].asnumpy()
        for fi in range(0, n, STRIDE):
            p = f"{OUT}/tiles/ep{ep:04d}_f{fi:03d}.png"
            if os.path.exists(p):                     # 재큐 시 다시 그리지 않는다
                man.append({"ep": ep, "f": fi, "path": p})
            else:
                t = np.concatenate([get_frame(k, fi) for k in VK], axis=1)
                Image.fromarray(t).resize((t.shape[1] // 2, t.shape[0] // 2)).save(p)
                man.append({"ep": ep, "f": fi, "path": p})
            if LIMIT and len(man) >= LIMIT:
                break
        if LIMIT and len(man) >= LIMIT:
            break
        if i % 50 == 0:
            print(f"shard{shard}: {i}/{len(eps)} eps, {len(man)} tiles", flush=True)
    json.dump(man, open(f"{OUT}/manifest_shard{shard}.json", "w"))
    print(f"shard{shard} done: {len(man)}")


if __name__ == "__main__":
    main()
