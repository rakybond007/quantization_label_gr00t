# Proprio + past executed-action student: Claude 인계

2026-09-06. 사용자 요청: **원래 코드를 수정하지 않고 복사본에 현재 이미지 +
지시문 + 현재 proprio + 직전 실행 액션 입력을 구현하고, srun GPU 1~2개로 학습
테스트한 뒤 설명을 남길 것.** 이 디렉터리만 새로 추가했다. 기존 학습/서빙/
평가 코드, 라벨, 이미지 캐시, 체크포인트는 수정하지 않았다. 본학습이나 폐루프
평가는 제출하지 않았으며 기존 클라이언트로 이 체크포인트를 바로 쓸 수는 없다.

## 구현과 판단

`model.py`의 이미지 trunk와 기존 baseline class는
`vlm_gate/scripts/train_gate_module.py:SmallGate`를 복사했다. 원본을 런타임에
import하지 않으므로 원본의 이후 수정과 이 실험이 서로 영향을 주지 않는다.
새 모델은 처음부터 학습했다. 기존 checkpoint를 변환하거나 덮어쓰지 않았다.

| 입력 | 배치 shape | 의미 |
|---|---|---|
| images | B,9,128,128 | 현재 RGB 3뷰, left/right/wrist 순서, uint8에서 /255 |
| text | B,384 | 기존 MiniLM 지시문 embedding |
| state | B,53 | 현재 observation.state, 원본 modality.json 순서 |
| previous_actions | B,16,12 | 현재 관측 이전 명령, 오래된 것부터 최신 순서 |
| history_mask | B,16 | 앞쪽 padding=0, 실제 이력=1 |

이미지 → 기존 CNN 4층(32/64/128/128) → 평균 pooling 128차원.
proprio 53 + flatten(actions) 192 + mask 16 = 261차원 → MLP 261→64→64.
시각 128 + 지시문 384 + motion 64 = 576차원 → 기존 형태 MLP 576→128→64→1.
출력은 sigmoid한 교사 점수의 예측값이며 실제 성공확률로 보정된 값이 아니다.
이미지 이력, GRU, 미래 계획 액션, 새로운 언어 모델은 추가하지 않았다.

기존 복사 baseline: **317,249 parameters**. 새 모델: **346,369 parameters**.
증가량은 **29,120 parameters (+9.18%)**다. 현재 실행 중인 액션 생성이 끝나기를
기다릴 필요 없는 입력만 사용한다. 실제 동시 실행은 기존 평가 클라이언트의
별도 배선 작업이며 이 실험이 자동으로 구현하지 않는다.

## 과거 액션의 정확한 의미 — 통합 전에 반드시 읽기

관측 f로 판단할 때 학습 입력은 `state[f]`와
`action[max(0,f-16):f]`다. `action[f]`나 그 이후는 절대 입력하지 않는다.
오프라인 데이터의 action은 기록된 demonstration 명령이다. GR00T가 예측했던
직전 청크를 복원한 것이 아니고, 물체의 실제 운동/관절 측정값과도 다르다.
데이터가 observation-before-action convention이라는 전제를 사용한다.

추론에서는 `ExecutedActionHistory`를 **환경마다 하나씩** 소유하고,
`env.step`이 성공적으로 실행된 뒤 실제 전달한 명령을 `append_executed()`에
기록한다. 에피소드 reset 시 이력도 reset한다. 게이트를 부르기 전에 방금 생성한
미래 청크 전체를 append하면 미래 정보 유입이므로 안 된다.

이력은 '직전 계획 청크 전체'가 아니라 **최근 최대 16개 실제 실행 명령**이다.
K2로 16개 계획을 8개 실행 명령으로 만들면 그 8개를 저장한다. 남은 슬롯에는
더 이전의 실행 명령이 들어갈 수 있다. 처음에는 왼쪽 zero padding + mask다.
컨트롤러 명령이 원본 12차원 표현과 다르면 명시적으로 변환해야 한다. 특히
그리퍼 0/1과 -1/+1 표현을 섞지 말 것. 정렬한 dict key를 이어붙이지 말 것.

원본 12차원 순서:
`base_motion[0:4], control_mode[4:5], EE_position_delta[5:8],
EE_rotation_axis_angle_delta[8:11], gripper_close[11:12]`.
현재 proprio는 원본 `meta/modality.json`의 53차원 전체다. 레이블로부터 추정한
state를 쓰지 않는다. normalization 평균/표준편차는 train split에서만 추정하고
checkpoint buffer로 저장한다. padding은 action 통계에서 제외하며, 정규화 후에도
mask를 곱해 실제 제로 이력과 구분한다. std<1e-5이면 1로 대체한다.

**아직 해결되지 않은 분포 차이:** 학습은 demonstration의 비압축 명령, 배포는
정책 출력의 압축/비압축 실행 명령이다. K2 합산 때문에 명령 범위도 바뀔 수 있다.
현재 스모크는 이 차이를 평가하지 않았다. 성공률 향상이나 배포 적합성을
주장할 수 없고, 이후 같은 라벨/분할로 baseline을 맞춰 학습하고 폐루프로
확인해야 한다. 임의의 K2 history 증강은 현재 관측과 물리적으로 일치하지 않을
수 있으므로 자동 추가하지 않았다.

## 파일

| 파일 | 역할 |
|---|---|
| model.py | 복사한 CNN + motion branch + causal history + 실행 이력 ring buffer |
| train.py | 엄격한 cache join, 원본 state/action 로딩, 분할/정규화, 실제 학습 |
| predict.py | 새로운 체크포인트 전용 Python 추론 adapter |
| verify.py | 실제 데이터 검증, 인과성/복원 검사, 고정 batch 학습, 비용 측정 |
| run_smoke.sh | 할당 내부에서 실행할 학습 + 검증 명령 |
| jobs.jsonl | 이번 srun의 별도 기록(원래 jobs.jsonl은 유지) |
| artifacts/smoke_159734_v1/ | 이번 체크포인트와 기계 판독 결과, git ignore |

데이터는 `assets/labels/robocasa/v6b_phase6_softA.parquet`의 `p_yes`를 사용했다.
완성돼 있고 기존 이미지 캐시와 연결 가능한 라벨로 배선을 검증하기 위한 선택이다.
**phase9 학습/비교가 아니다.** 새 라벨로 바꾸려면 CLI의 --labels를 바꾸고
episode_index, frame_index, task, p_yes 컬럼 및 이미지 캐시 커버리지를 확인한다.
누락 이미지와 중복 키는 원본처럼 조용히 건너뛰지 않고 실패시킨다.

지시문별로 episode를 고정 seed로 분리한다. default split 25% val이고 한 episode가
양쪽에 걸치지 않는다. 스모크 제한은 max 8 instructions × max 6 episodes ×
max 32 frames이며, 실 데이터에서 **46 episodes / 1,435 rows**가 선택됐다.
학습 31 episodes / 965 rows, 검증 15 episodes / 470 rows. cache match 1435/1435.
라벨과 사용한 source episode parquet, embedding, 코드 SHA256 및 modality/info는
`provenance.json`, 정확한 split은 `split.parquet`에 저장했다. 이미지 캐시 전체의
내용 hash를 검증한 것은 아니다.

학습 loss는 기존처럼 soft BCE, AUC는 sklearn의 정렬 기반 구현으로 교체했다.
교사 p_yes>=0.5로 AUC의 이진 정답을 만든다. **best checkpoint는 val soft BCE가
가장 낮은 epoch**로 선택한다(원본은 AUC). 새 입력의 효과만 비교하려면 baseline도
동일 split/seed/학습량/선택 기준으로 맞춰야 한다. 이번 baseline은 시간 측정용이며
학습 성능 대조군으로 훈련하지 않았다.

## 실제 GPU 테스트 결과

- srun job **159734**, GPU 1개, NVIDIA A100 80GB PCIe, torch 2.5.1+cu124.
- 요청 명령은 아래와 같으며 --time/-p/CPU/메모리 옵션은 사용하지 않았다.
  할당 시간은 시스템 기본값에 맡겼다. 테스트 후 interactive shell을 종료해
  **GPU를 반환했다**. sacct 확인: COMPLETED, exit 0:0, 할당 사용 9분 51초.
- 6 epochs, batch 64, lr 3e-4, FP32, num_workers=0, bounded uint8 RAM preload.
- 학습 BCE **0.68114 → 0.62142** (epoch 1→6).
- 초기 val BCE **0.68843**, best(epoch 5) **0.63695**, 그때 AUC **0.82443**.
- 마지막 epoch AUC 0.82626이지만 BCE가 0.64617이므로 best는 epoch 5다.
- 학습 초기 준비 50.57초, 준비 포함 train 프로그램 58.40초. 이후 verify는 별도.
  NFS/캐시의 영향을 받는 작은 preload 스모크라 전량 소요 시간으로 외삽하지 말 것.
- 비전/새 motion/head 가중치 모두 finite nonzero gradient 확인.
- 현재 proprio와 과거 액션 각각에 대한 입력 gradient nonzero 확인.
- 현재/미래 액션을 변경해도 과거 입력 불변, reset 시 빈 이력, padding 무영향,
  episode split 분리, train-only 통계, checkpoint exact reload 검사 통과.
- RGB 배열부터 state/action까지 받는 Predictor와 학습 경로의 출력 일치 확인.
- 실제 고정 16표본에서 100 optimizer steps: BCE **0.64874 → 0.53859**.
- 빠른 AUC의 tie 처리가 기존 pairwise 정의와 일치함을 확인.

FP32 resident GPU tensors, CUDA 동기화 후 반복 시간의 중앙값이다.
forward는 각 100회 × 5묶음, 학습은 각 30회 × 5묶음, warmup 10회.
학습 step에는 forward/backward/AdamW가 포함된다.

| 측정 | 기존 복사 baseline | proprio/history | 증가 |
|---|---:|---:|---:|
| 추론 batch 1 | 0.526 ms | 0.692 ms | +0.166 ms, 약 32% |
| 추론 batch 64 | 0.674 ms | 0.730 ms | +0.057 ms, 약 8% |
| 학습 step batch 64 | 3.031 ms | 3.186 ms | +0.155 ms, 약 5% |

HTTP/base64/이미지 전처리/데이터 로딩/정책과 동시 실행 시 GPU 경합은 제외했다.
따라서 전체 학습이 5%만 느려진다거나 배포 오버헤드가 항상 0.166ms라고
보장할 수 없다. GPU 산술량이 작아도 연산 호출 비용으로 batch 1 비율은 커진다.
수치는 학습 배선 정상과 작은 절대 추가 비용을 보여주며, 모델 품질 우위는 아니다.

## 재현

현재 서버 workspace에서, GPU 할당은:

```bash
srun --gpus=1 --job-name=codex_gate_proprio_executed_action_training_smoke --wckey=project-short-name:sub_fast bash --noprofile --norc
```

할당된 셸 안에서 새 출력 경로를 지정한다. 존재하는 출력 경로는 실패하도록 했다.

```bash
bash /sjw_alinlab/home/hojin2/quantization_agent_workspace/quantization_label_gr00t/vlm_gate/experiments/proprio_history/run_smoke.sh /sjw_alinlab/home/hojin2/quantization_agent_workspace/quantization_label_gr00t/vlm_gate/experiments/proprio_history/artifacts/my_new_smoke
```

검증만 다시 실행하려면:

```bash
/sjw_alinlab/home/hojin2/miniconda3/envs/quant_gate_eval/bin/python -u /sjw_alinlab/home/hojin2/quantization_agent_workspace/quantization_label_gr00t/vlm_gate/experiments/proprio_history/verify.py --run-dir /sjw_alinlab/home/hojin2/quantization_agent_workspace/quantization_label_gr00t/vlm_gate/experiments/proprio_history/artifacts/smoke_159734_v1
```

첫 학습 종료 시 verify.py를 아직 작성 중이어서 wrapper의 후속 검증 명령은 파일
미존재로 실패했다. 학습 산출물은 정상 완료됐고, 파일 작성 후 위 명령으로 검증을
별도 실행해 전 항목을 통과했다. 현재 wrapper가 참조하는 파일은 모두 존재한다.

원본 Mac→서버 tools/dev 매핑은 이 새 experiments 폴더를 보낼지 보장하지 않는다.
Mac에서 이어받는 에이전트는 이 폴더를 보존해 원본 클론으로 가져오고 sync 범위를
확인해야 한다. 현재 파일은 서버의 quantization_label_gr00t 안에 있고 커밋/푸시는
하지 않았다. 기존 scripts 경로에 복사해 덮어쓰지 말 것.

## 추론 통합 예시

이 폴더를 Python import 경로에 둔 뒤:

```python
from model import ExecutedActionHistory
from predict import Predictor

gate = Predictor(".../checkpoint.pt")
history = ExecutedActionHistory(history=16, action_dim=12)  # environment마다 별개

# env.reset() 직후
history.reset()

# 현재 관측으로 판단. 배열 순서/단위는 위 스키마대로 구성한다.
actions, mask = history.arrays()
result = gate.predict(current_rgb_views, instruction, current_state53, actions, mask)

# 이후 실제 env.step에 쓴 명령마다 기록. 미래 계획을 미리 기록하지 않는다.
obs, reward, done, info = env.step(controller_command)
history.append_executed(command_in_dataset_12d_order)
```

Python adapter는 known instruction embedding만 허용한다. 없는 지시문을 유사한
다른 지시문으로 대체하지 않는다. 추가 지시문은 같은 MiniLM 방식으로 embedding을
생성해야 한다. 체크포인트는 embedding 파일 경로를 저장하므로 이관 시 경로도
맞춰야 한다. client의 카메라 flip/뷰 순서도 기존 학습과 맞춰야 한다.

현재 train.py는 1 GPU 학습이다. 사용자 허용 범위 내에서 1 GPU로 테스트를 끝냈으며
2 GPU DDP 구현/성능은 검증하지 않았다. 본학습으로 확대하려면 --preload를 빼고
표본 제한을 0으로 설정할 수 있지만, 이번 작업에서 본학습은 실행하지 않았다.

다음 통합의 우선순위는 기존 클라이언트에서 state53/실행 명령12의 단위와 순서를
명시적으로 연결하는 일이다. 그 뒤 동일 라벨/동일 split으로 image+text baseline을
맞춰 비교하고, 동일한 폐루프 조건에서 성공률–실행 스텝–실시간 지연을 비교한다.
