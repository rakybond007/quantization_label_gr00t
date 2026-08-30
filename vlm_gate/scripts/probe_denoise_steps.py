"""1스텝 디노이징 값이 게이트에 쓸 만한 정보를 담는지 측정.

게이트 모듈은 액션 헤드의 디노이징과 동시에 돌아야 latency가 숨는다. 최종 청크를
입력으로 받으면 디노이징을 기다려야 하므로 순수 오버헤드가 된다. 대안은 1스텝만
지난 중간 상태를 받는 것 — 그 시점 값이 최종 청크의 '거친 구조'를 이미 담고
있다면 게이트가 필요한 판단(파지 전이·반전·저속 파지)에는 충분할 수 있다.

측정: 같은 관측에 대해 각 디노이징 스텝의 중간 상태를 저장하고, 거기서 계산한
기술자가 최종(4스텝) 기술자와 얼마나 일치하는지 본다.
"""
import json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/quantization_agent_workspace/Isaac-GR00T"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robocasa_descriptors import descriptors

CKPT = os.environ.get("CKPT",
    os.path.expanduser("~/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/"
                       "groot_n1_5_bs64_baseline/checkpoint-60000"))
DS = "/sjw_alinlab2/home/myungkyu/.cache/huggingface/lerobot/kimtaey/robocasa_mg_gr00t_300"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60

from gr00t.model.policy import Gr00tPolicy
from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.experiment.data_config import DATA_CONFIG_MAP

cfg = DATA_CONFIG_MAP["single_panda_gripper"]
policy = Gr00tPolicy(model_path=CKPT, embodiment_tag="new_embodiment",
                     modality_config=cfg.modality_config(),
                     modality_transform=cfg.transform(), denoising_steps=4)
ds = LeRobotSingleDataset(dataset_path=DS, modality_configs=cfg.modality_config(),
                          transforms=cfg.transform(), embodiment_tag="new_embodiment")
print(f"데이터 {len(ds)}프레임, {N}개 표본")

head = policy.model.action_head
traces = {}
orig = head.get_action

def traced(*a, **k):
    """디노이징 중간 상태를 가로챈다."""
    caught = []
    real_step = torch.Tensor.add_
    out = orig(*a, **k)
    return out

# Euler 루프에 훅: x_t 갱신 지점을 감싸 중간 상태 저장
import types
snap = {}
def make_hook():
    def hook(module, inp, out):
        pass
    return hook

# 간단히: num_inference_timesteps를 1~4로 바꿔가며 같은 관측을 반복 추론
res = {k: [] for k in (1,2,3,4)}
rng = np.random.default_rng(0)
idxs = rng.choice(len(ds), N, replace=False)
for i in idxs:
    step = ds[int(i)]
    outs = {}
    for ns in (1,2,3,4):
        head.num_inference_timesteps = ns
        torch.manual_seed(1234)                     # 같은 노이즈에서 출발
        with torch.no_grad():
            a = policy.get_action(step)["action_pred"]
        outs[ns] = np.asarray(a, dtype=np.float64)
    for ns in (1,2,3):
        res[ns].append((outs[ns], outs[4]))
head.num_inference_timesteps = 4

print(f"\n{'스텝':>4}{'액션 상관':>10}{'그리퍼전이 일치':>14}{'반전 일치':>10}{'닫힘저속 일치':>13}")
for ns in (1,2,3):
    pairs = res[ns]
    cors=[]; agr={"grip_change":[], "reversal":[], "closed_slow":[]}
    for p, f in pairs:
        cors.append(np.corrcoef(p[:, 5:11].ravel(), f[:, 5:11].ravel())[0,1])
        dp = descriptors(p, 0, n=min(16, len(p)))
        df = descriptors(f, 0, n=min(16, len(f)))
        for k in agr: agr[k].append(float(dp[k] == df[k]))
    print(f"{ns:>4}{np.nanmean(cors):>10.3f}{np.mean(agr['grip_change']):>13.0%}"
          f"{np.mean(agr['reversal']):>10.0%}{np.mean(agr['closed_slow']):>13.0%}")
