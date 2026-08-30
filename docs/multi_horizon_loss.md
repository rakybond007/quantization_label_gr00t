# Multi-Horizon Auxiliary Loss for GR00T-N1.5

## 1. 개요

기존 모델은 16-step action chunk만 예측한다. Multi-horizon loss는 **같은 backbone+DiT body**를 공유하면서 별도의 decoder head 두 개를 추가해, 다음 두 가지 압축된 target에 대해서도 flow matching loss를 추가로 학습한다:

- **Factor 2**: GT actions 32-step → 인접 2개씩 합산 → 16-step compressed target
- **Factor 4**: GT actions 64-step → 인접 4개씩 합산 → 16-step compressed target

각 step이 더 큰 delta를 표현하도록 학습되어, 추론 시 **adaptive execution length**가 가능해진다 (자세한 generation 전략은 §6 참조).

## 2. 설계 선택

세 가지 옵션 중 **Option A (separate decoder heads)**를 선택:

| 옵션 | 방식 | 단점 |
|---|---|---|
| A. Separate heads | factor별 별도 decoder head, 각자 independent forward | 학습 시 3x 비용 |
| B. Mixed target | 한 head에 random target 할당 | 신호 충돌, 모델이 어떤 출력을 낼지 모름 |
| C. Conditioning | granularity_id를 input으로 | 너무 일반적이지 않음 (사용자 의견) |

## 3. 구현된 loss 흐름

각 forward step마다 **3번의 독립 flow matching pass** (각자 다른 noise/timestep):

```
shared inputs: vl_embs, state_features, embodiment_id (DiT body 공유)

[main path]
  noise_1, t_1 ~ random
  noisy_1 = (1-t_1)*noise_1 + t_1*action_16
  pred_1 = action_decoder( DiT(noisy_1, t_1, ...) )
  loss_main = MSE(pred_1, action_16 - noise_1) * mask_16

[aux path: factor=2]
  clean_2 = sum(GT[0:32] grouped by 2)  -> (16, D)
  noise_2, t_2 ~ random
  noisy_2 = (1-t_2)*noise_2 + t_2*clean_2
  pred_2 = aux_decoder_f2( DiT(noisy_2, t_2, ...) )
  loss_f2 = MSE(pred_2, clean_2 - noise_2) * mask_2

[aux path: factor=4]
  clean_4 = sum(GT[0:64] grouped by 4)  -> (16, D)
  ... (analogous)
  loss_f4 = ...

total = w_main * loss_main + w_f2 * loss_f2 + w_f4 * loss_f4
```

**중요한 mask 처리**: 압축 target의 mask는 그룹 내 **모든** 위치가 valid해야 valid (`mask.all(dim=group_axis)`). 에피소드 끝에서 padding된 step이 섞인 그룹은 자동으로 무시된다.

## 4. 변경된 파일

| 파일 | 변경 |
|---|---|
| `gr00t/model/action_head/flow_matching_action_head.py` | Config 필드 추가, `aux_action_decoders` ModuleDict, `_compress_actions`, `_flow_matching_loss`, multi-loss forward |
| `gr00t/model/transforms.py` | `extended_action_horizon` 필드, `_prepare_action_extended`, apply_single에서 `action_extended` 키 추가 |
| `gr00t/model/gr00t_n1.py` | `validate_inputs`가 `action_extended` 검증 |
| `gr00t/experiment/data_config.py` | `SinglePandaGripperMultiHorizonDataConfig` 추가, `DATA_CONFIG_MAP` 등록 |
| `gr00t/experiment/trainer.py` | aux loss 자동 logging |
| `scripts/gr00t_finetune.py` | `--use-multi-horizon-loss`, `--multi-horizon-factors`, weights CLI flags |
| `run_scripts/train/robocasa/finetune_gr00t_n1_5_multi_horizon.sh` | 학습 스크립트 |

## 5. 사용법

### 5.1 학습

```bash
sbatch run_scripts/train/robocasa/finetune_gr00t_n1_5_multi_horizon.sh
```

스크립트 내용 핵심:
```bash
python scripts/gr00t_finetune.py \
    --data-config single_panda_gripper_multi_horizon \
    --use-multi-horizon-loss \
    --multi-horizon-factors 2 4 \
    --multi-horizon-loss-weights 1.0 1.0 \
    --multi-horizon-main-weight 1.0 \
    ...
```

### 5.2 Smoke test

```bash
# 1) Tiny model + synthetic data
python scripts/smoke_test_multi_horizon.py

# 2) Real dataset transform
python scripts/smoke_test_multi_horizon_data.py

# 3) Real model + real data E2E (GPU + flash-attn 필요)
python scripts/smoke_test_multi_horizon_e2e.py
```

### 5.3 Logging

학습 중 wandb/tensorboard에 자동으로 다음이 logged:
- `loss` (total)
- `loss_main` (factor=1)
- `loss_f2`, `loss_f4` (각 aux head)

## 6. Generation Strategy (학습 후 활용 방안)

학습 후 모델은 **3개의 decoder head** (main + f2 + f4)를 갖는다. 각 head는 16-step 출력을 내지만, **각 step이 표현하는 시간 단위가 다르다**:

- main: 1 step = 1 unit time → 총 16 units 커버
- f2: 1 step = 2 unit time → 총 32 units 커버 (2x macro)
- f4: 1 step = 4 unit time → 총 64 units 커버 (4x macro)

이를 활용하는 방법 (제안):

### 옵션 1: Static head selection
Task에 따라 한 head를 고정 사용:
- 정밀 manipulation: main head (16 fine actions)
- 빠른 이동: f2 또는 f4 head (큰 delta로 빠른 motion)

### 옵션 2: Adaptive execution
Episode 중에 동적으로 head 전환:
1. 현재 step에서 main head로 16-step 생성
2. **별도로** f2 head로 16-step 생성 (32 unit time 분의 plan)
3. 두 trajectory가 일치하면 (main의 첫 8 step ≈ f2의 첫 4 step의 unfold) → f2의 macro로 빠르게 실행
4. 불일치하면 main의 fine action으로 정밀 실행

### 옵션 3: Hierarchical execution
- f4 head로 long-horizon coarse plan 생성
- 그 plan을 따라가면서 main head로 매 step fine correction
- 실행: `executed_action = main_pred[0] + alpha * (f4_pred[t//4] / 4 - main_pred[0])`
  여기서 alpha는 작은 값 (예: 0.1) — 주로 fine 사용, coarse는 가이드만

### 옵션 4: Test-time ensemble (간단, 추천 1순위)
- 매 step inference 시 main + f2 + f4 head 모두 forward
- Main의 첫 step과 f2의 첫 step의 절반, f4의 첫 step의 1/4을 평균
  → variance reduction (offline validation에서 6.3% MSE 감소를 보았던 그 효과)

```python
def step(obs):
    # Single forward through DiT body
    main_chunk = action_decoder(features)        # (16, D), 16 fine steps
    f2_chunk = aux_f2(features)                   # (16, D), each = 2x time
    f4_chunk = aux_f4(features)                   # (16, D), each = 4x time

    # Convert to "per-unit-time" actions for the immediate future
    main_step_0 = main_chunk[0]                   # next 1 unit
    f2_step_0_unfold = f2_chunk[0] / 2            # next 1 unit (averaged of 2)
    f4_step_0_unfold = f4_chunk[0] / 4            # next 1 unit (averaged of 4)

    # Ensemble (different temporal "context" per estimator)
    return (main_step_0 + f2_step_0_unfold + f4_step_0_unfold) / 3
```

이건 우리 이전 "candidate averaging"과 비슷한 효과를 같은 forward pass 비용으로 얻는다 (decoder만 추가, DiT body는 1번만 forward).

### 추천 우선순위

1. **옵션 1 (정밀 vs 빠름)**: 가장 단순, baseline과 비교 용이. 학습이 잘 되었는지 sanity check 용도.
2. **옵션 4 (ensemble unfold)**: 추가 비용 없이 quality 향상.
3. **옵션 2 (adaptive)**: 가장 흥미롭지만 disagreement signal이 의미가 있는지부터 검증 필요.

학습이 끝나면 먼저 옵션 1로 baseline-style eval을 돌려서 main/f2/f4 각각의 성공률을 비교하고, 그 결과를 보고 다음 옵션으로 진행하는 것을 추천.

## 7. Protective Measures (사전학습 보호)

새로 추가된 aux decoder들은 random initialization이라, 학습 초반에 큰 gradient가
공유 DiT body로 흘러 사전학습된 representation을 망가뜨릴 위험이 있다. 이를
완화하기 위해 두 가지 보호 장치가 들어있다:

### 7.1 Gradient scaling (`aux_grad_scale_to_body`)

`GradScale` autograd function을 통해 **aux loss → DiT body로 흐르는 gradient**를
`scale`배 dampening한다. Aux decoder 자체는 full gradient를 받는다.

```python
# Forward: identity. Backward: grad * scale (aux path만).
model_output_for_decoder = GradScale.apply(model_output, scale)
pred_aux = aux_decoder(model_output_for_decoder, ...)
```

| scale | 동작 |
|---|---|
| 1.0 | 보호 없음 (기본 multi-task) |
| 0.5 | DiT grad 절반으로 dampen |
| **0.1** (권장 default) | DiT는 main에서 1.0, aux에서 0.1 받음 → main이 dominance |
| 0.0 | Aux head는 frozen DiT 위에서만 학습 (DiT 완전 보호) |

검증된 거동 (smoke test):
```
scale=1.0 -> DiT grad=201.71  aux_decoders grad=62.41
scale=0.5 -> DiT grad=100.86  aux_decoders grad=62.41
scale=0.1 -> DiT grad= 20.17  aux_decoders grad=62.41
scale=0.0 -> DiT grad=  0.00  aux_decoders grad=62.41
```
→ DiT grad만 scale에 비례, aux decoder grad는 영향 없음.

### 7.2 Warmup (`aux_loss_warmup_steps`)

학습 초반 N step 동안 aux loss weight를 0 → configured value로 linear ramp.
Aux head가 sensible한 prediction을 시작할 때까지 main path가 stable하게 유지됨.

```python
warmup = min(1.0, current_forward_step / warmup_steps)
total = w_main * loss_main + sum((w_aux * warmup) * loss_aux for ...)
```

기본 5000 step (default). 0이면 disabled.

### 7.3 두 장치 조합 효과

- 학습 초반 aux head가 random일 때: warmup이 weight를 거의 0으로 → grad 거의 안 들어감
- 어느 정도 aux head가 학습되면: warmup이 1.0으로 수렴, aux grad가 DiT로 흘러감
- 안정화 후에도 gradient scaling이 main을 우선시

### 7.4 모니터링

학습 로그에 자동으로 추가됨:
- `loss_main`: 메인 head의 loss (저하 여부 모니터링)
- `loss_f2`, `loss_f4`: 각 aux head의 loss
- `aux_warmup`: 현재 warmup factor (0.0 → 1.0)

`loss_main`이 baseline 학습 (multi-horizon 끄고) 대비 비슷하거나 더 낮으면 보호 성공.
크게 올라가면 scale을 더 낮추거나 warmup을 늘려야 함.

## 8. 주의사항

### 학습 비용 증가
- 3-forward 방식이라 DiT body가 **3번** 호출됨
- 학습 시간 약 3x (체감) — backbone/projector는 그대로지만 DiT가 클 경우 영향 큼
- 비용 줄이기: **stochastic mode** — 매 step에서 main + 1 random aux만 forward (학습 시간 ~2x)

### Discrete dimension (gripper, control_mode)
- 현재 구현은 모든 dim에 대해 **합산**
- gripper (binary)의 합은 0/1/2 — 0/1 표현과 어긋남
- 1차 학습은 그대로 진행하고, 결과가 나쁘면 discrete dim 별도 처리 (예: max 또는 last) 추가 권장

### Episode 끝 padding
- mask가 모든 그룹원 valid를 요구해서 자동으로 짧은 에피소드 끝은 무시됨
- 단, 너무 짧은 에피소드(< 64 step)는 거의 모든 f4 mask가 invalid → loss_f4가 0에 가까움
- Robocasa-300 데이터셋은 평균 200+ step이라 무시 가능

### Resume 호환성
- Aux decoder들은 새 weight (학습 시 attach)
- Pretrained checkpoint에서 시작 → 첫 step 후 자동으로 saved (config에 multi_horizon flags 저장됨)
- Resume 시 config에서 자동으로 aux head 재구성됨

## 9. 검증 결과

| 테스트 | 상태 |
|---|---|
| Tiny model + synthetic data forward+backward | ✅ |
| Real dataset transform (`action`, `action_extended` 동시 fetch) | ✅ |
| Real model + real data E2E forward | ⏳ (flash_attn 설치 필요) |
| 학습 1 step 시도 | ⏳ |
| 학습 끝까지 안정성 | ⏳ |

## 10. 환경 / 의존성 메모

- README의 conda env 생성 (`gr00t`, python 3.10, `pip install -e .[base]`).
- `pip install qwen-vl-utils` 추가 필요.
- `flash-attn` 설치 시 `TMPDIR`이 cache와 같은 filesystem이어야 함 (cross-device link 회피):
  ```bash
  mkdir -p $HOME/tmp_build
  export TMPDIR=$HOME/tmp_build
  pip install --no-build-isolation --no-cache-dir flash-attn==2.7.1.post4
  ```

## 11. 다음 단계

1. **flash_attn 설치 마무리** → e2e smoke test 통과 확인.
2. **1 step 학습 시도** (`max_steps=10`, `batch_size=2`)로 trainer 흐름 검증.
3. **본격 학습** (`max_steps=60000`, `batch_size=32`).
4. 학습 완료 후 **§6의 옵션 1** (head별 baseline-style eval)부터 진행.
