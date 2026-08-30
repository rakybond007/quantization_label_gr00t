"""n1.7 이 robocasa LeRobot 데이터셋을 그대로 읽는지 확인.

변환은 필요 없다 — n1.7 의 lerobot_episode_loader 가 meta/ + 청크 parquet + mp4 를
직접 읽는다. 필요한 것은 임베디먼트 모달리티 설정 하나뿐이다.

확인 항목
  ① 모달리티 키가 데이터셋 meta/modality.json 과 맞물리는가
  ② 액션이 (16, D) 로 나오는가, D 가 n1.5 구성과 일치하는가
  ③ 3뷰 비디오가 실제로 디코딩되는가
"""
import os, sys, numpy as np
DS = os.environ.get("DS", os.path.expanduser(
    "~/quantization_agent_workspace/assets/datasets/robocasa_n17_mirror"))
TAG = "robocasa_panda_gripper"

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.dataset.sharded_single_step_dataset import (
    ShardedSingleStepDataset, extract_step_data)

mc = MODALITY_CONFIGS[TAG]
print(f"① 모달리티 설정 '{TAG}'")
for k, v in mc.items():
    print(f"   {k:9s} keys={v.modality_keys} delta={len(v.delta_indices)}개")

ds = ShardedSingleStepDataset(
    dataset_path=DS,
    embodiment_tag=EmbodimentTag("new_embodiment"),
    modality_configs=mc,
    video_backend=os.environ.get("VIDEO_BACKEND", "decord"),
    shard_size=int(os.environ.get("SHARD_SIZE", 2**10)),      # 기본값 1024
    episode_sampling_rate=float(os.environ.get("EP_RATE", 0.1)),  # 기본값 0.1
    seed=0,
    allow_padding=True,
)
print(f"   데이터셋 길이 {len(ds)} 스텝")

print("② 에피소드 하나 로드")
ep_df = ds.episode_loader[0]   # __getitem__ 이 에피소드 DataFrame 을 준다
print(f"   에피소드 0: {len(ep_df)} 스텝, 컬럼 {len(ep_df.columns)}개")

print("③ 스텝 하나 추출")
step = extract_step_data(ep_df, 0, mc, EmbodimentTag("new_embodiment"), True)
def shp(x):
    a = np.asarray(x)
    return tuple(a.shape) + (str(a.dtype),)

print(f"   images  뷰 {list(step.images)}")
for k, v in step.images.items():
    print(f"     {k}: {len(v)}프레임 · {shp(v[0])}")
print(f"   states  {list(step.states)}")
tot_s = 0
for k, v in step.states.items():
    a = np.asarray(v); tot_s += a.reshape(a.shape[0], -1).shape[-1] if a.ndim > 1 else a.size
    print(f"     {k}: {shp(v)}")
print(f"   actions {list(step.actions)}")
tot_a = 0
for k, v in step.actions.items():
    a = np.asarray(v); tot_a += a.shape[-1] if a.ndim > 1 else 1
    print(f"     {k}: {shp(v)}")
print(f"   text: {str(step.text)[:70]!r}")
print(f"   embodiment: {step.embodiment}")
print()
print(f"=> 액션 청크 지평 {next(iter(step.actions.values())).shape[0]} 스텝, 합산 차원 {tot_a}")
print(f"   (n1.5 는 12차원: 0-4 미사용, 5-7 EE delta, 8-10 회전, 11 그리퍼)")
