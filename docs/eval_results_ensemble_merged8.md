# RoboCasa Eval — Ensemble→merged8 (8-step effective execution)

작성일: 2026-04-25
대상 잡: 292742 (mh_m8) / 292743 (mh_m8+econsist) — 6개 eval 스크립트 중 ensemble 두 가지

## 비교 두 변수

| 축 | 변수 | 값 |
|----|------|------|
| **학습 모델** | ensemble-consistency loss 사용 여부 | `mh_m8` (no consist) vs `mh_m8+econsist` |
| **추론 헤드** | 동일 — server side `ensemble_fix` (main+f2+f4 WLS) | 16-step output |
| **클라이언트** | `robocasa_service_merged.py`로 16→8 pair merge | effective horizon = 8 |

→ **둘 다 effective execution horizon = 8** (8-step 압축 액션). 차이는 학습 시 ensemble-consistency loss 추가 유무.

## Config

| | mh_m8 model | mh_m8+econsist model |
|---|---|---|
| 학습 스크립트 | `finetune_gr00t_n1_5_mh_m8.sh` | `finetune_gr00t_n1_5_mh_m8_econsist.sh` |
| 체크포인트 | `groot_n1_5_bs64_mh_m8_discfix/checkpoint-60000` | `groot_n1_5_bs64_mh_m8_econsist_discfix/checkpoint-60000` |
| Loss | `1.0·L_main + 1.0·L_m8 + warmup·(1.0·L_f2 + 1.0·L_f4)` | 위 + `c_warmup·0.1·L_consist` |
| Inference head | `ensemble_fix` | `ensemble_fix` |
| Server output H | 16 (main+f2+f4 WLS combined) | 16 |
| Client merge | even+odd sum (continuous), odd-of-pair (discrete dims 6,11) → H=8 | 동일 |
| n_episodes/task | 50 | 50 |
| seed | 42 | 42 |
| max_episode_steps | 1500 | 1500 |
| 24 task across | 8 array jobs × 3 tasks each | 동일 |

## Per-task 결과 (성공/롤아웃, 비율)

`a/b (c)` = 성공 a회 / 롤아웃 b회 / 성공률 c. 모든 task는 50 ep target이고 실제 돌아간 ep 수는 b로 표기.

| Task | mh_m8 + ens→8 | mh_m8+econsist + ens→8 | Δ |
|------|------:|------:|------:|
| CloseDoubleDoor | 36/50 (0.72) | 37/50 (0.74) | +0.02 |
| CloseDrawer | 50/50 (1.00) | 50/50 (1.00) | 0.00 |
| CloseSingleDoor | 49/50 (0.98) | 47/50 (0.94) | -0.04 |
| CoffeePressButton | 43/50 (0.86) | 42/50 (0.84) | -0.02 |
| CoffeeServeMug | 38/50 (0.76) | 37/50 (0.74) | -0.02 |
| CoffeeSetupMug | 12/50 (0.24) | 16/50 (0.32) | **+0.08** |
| OpenDoubleDoor | 36/50 (0.72) | 35/50 (0.70) | -0.02 |
| OpenDrawer | 29/50 (0.58) | 32/50 (0.64) | +0.06 |
| OpenSingleDoor | 36/50 (0.72) | 37/50 (0.74) | +0.02 |
| PnPCabToCounter | 25/50 (0.50) | 25/50 (0.50) | 0.00 |
| PnPCounterToCab | 24/50 (0.48) | 23/50 (0.46) | -0.02 |
| PnPCounterToMicrowave | 14/50 (0.28) | 11/50 (0.22) | -0.06 |
| PnPCounterToSink | 1/1 (1.00) | 34/50 (0.68) | — (sample 부족, 비교 무의미) |
| PnPCounterToStove | 25/50 (0.50) | 25/50 (0.50) | 0.00 |
| PnPMicrowaveToCounter | 11/50 (0.22) | 10/50 (0.20) | -0.02 |
| PnPSinkToCounter | 20/50 (0.40) | 22/50 (0.44) | +0.04 |
| PnPStoveToCounter | 36/50 (0.72) | 36/50 (0.72) | 0.00 |
| TurnOffMicrowave | 47/50 (0.94) | 50/50 (1.00) | +0.06 |
| TurnOffSinkFaucet | 41/50 (0.82) | 43/50 (0.86) | +0.04 |
| TurnOffStove | 9/50 (0.18) | 9/50 (0.18) | 0.00 |
| TurnOnMicrowave | 27/50 (0.54) | 26/50 (0.52) | -0.02 |
| TurnOnSinkFaucet | 35/50 (0.70) | 31/50 (0.62) | -0.08 |
| TurnOnStove | 15/50 (0.30) | 22/50 (0.44) | **+0.14** |
| TurnSinkSpout | 38/50 (0.76) | 40/50 (0.80) | +0.04 |
| **TOTAL** | **697/1151 (0.606)** | **740/1200 (0.617)** | +0.011 |

## 평균 / 분석

| 모델 | 성공/롤아웃 | 성공률 |
|---|---:|---:|
| **mh_m8 + ensemble→8** | 697 / 1151 | **0.606** |
| **mh_m8+econsist + ensemble→8** | 740 / 1200 | **0.617** |

거의 동일 (≈0.011 차). ensemble-consistency loss 추가가 본 평가 셋팅(ensemble→8) 전체 평균 성공률에 큰 영향 없음.

**Note**: mh_m8 의 PnPCounterToSink 는 array job 일부 미완으로 1 ep만 돌아 직접 비교에서 제외함.

태스크별로 보면:
- 큰 개선: CoffeeSetupMug (+0.08), TurnOnStove (+0.14)
- 큰 후퇴: TurnOnSinkFaucet (-0.08), PnPCounterToMicrowave (-0.06)
- 대부분 ±0.02 내 일치

## 노트
- PnPCounterToSink 의 mh_m8 결과(1.00)는 1 episode 만 기록되어 의미 없음 — array job 일부가 미완으로 종료된 흔적
- `prediction.txt` summary line이 누락된 task가 다수 — 직접 per-episode `is_success: [True/False]` 라인을 카운트해 계산
- 다른 head 비교(예: m8 head 단독, main head 단독)는 진행 예정 (m8 head는 server-side validation 버그로 실패 → 수정 후 재제출 = 293603/293604)

## 성공 에피소드 평균 동영상 길이 (rollout 효율성)

베이스라인 (16-step main) 대비 8-step 압축 액션 사용 시 rollout이 얼마나 빨라지는지.
환경 fps=20, 길이 = 성공한 에피소드 mp4 의 ffprobe nb_read_packets 평균.

> **주의**: `n=`는 **비디오로 녹화된 성공 에피소드 수** (RecordVideo trigger sparsity로 actual success 횟수 ≠ 녹화 수). 위 per-task 결과 표의 success/총 ep와 다름. 평균 길이 자체는 신뢰 가능.

| Task | baseline (16-step main) | baseline + client merge 16→8 | mh_m8 + ens→8 | mh_m8+econsist + ens→8 |
|------|------|------|------|------|
| CloseDoubleDoor | 21.5s (n=42) | 17.5s (n=39) | 15.3s (n=10) | 14.6s (n=36) |
| CloseDrawer | 10.2s (n=42) | 6.0s (n=50) | 6.5s (n=50) | 5.1s (n=16) |
| CloseSingleDoor | 13.0s (n=47) | 8.8s (n=46) | 8.9s (n=49) | 7.6s (n=47) |
| CoffeePressButton | 6.0s (n=31) | 4.4s (n=45) | 3.4s (n=43) | 3.1s (n=11) |
| CoffeeServeMug | 15.2s (n=38) | 9.4s (n=39) | 9.7s (n=38) | 9.2s (n=14) |
| CoffeeSetupMug | 13.2s (n=7)  | 8.0s (n=12) | 7.7s (n=10) | 5.9s (n=4)  |
| OpenDoubleDoor | 36.8s (n=39) | 25.1s (n=34) | 22.7s (n=35) | 22.8s (n=12) |
| OpenDrawer | 11.3s (n=37) | 7.9s (n=23) | 6.8s (n=29) | 7.1s (n=7) |
| OpenSingleDoor | 13.8s (n=39) | 13.8s (n=35) | 17.5s (n=14) | 12.6s (n=37) |
| PnPCabToCounter | 22.1s (n=7) | 12.0s (n=26) | 14.1s (n=9) | 11.3s (n=6) |
| PnPCounterToCab | 14.9s (n=25) | 10.2s (n=25) | 9.2s (n=15) | 11.1s (n=23) |
| PnPCounterToMicrowave | 19.3s (n=4) | 14.3s (n=11) | 13.6s (n=6) | 25.4s (n=3) |
| PnPCounterToSink | 25.0s (n=37) | 15.0s (n=31) | 18.9s (n=1) | 14.2s (n=9) |
| PnPCounterToStove | 18.5s (n=19) | 12.1s (n=26) | 11.0s (n=22) | 20.4s (n=5) |
| PnPMicrowaveToCounter | 15.3s (n=12) | 9.3s (n=7) | 10.3s (n=11) | 8.0s (n=4) |
| PnPSinkToCounter | 16.2s (n=35) | 11.3s (n=19) | 10.2s (n=20) | 11.8s (n=8) |
| PnPStoveToCounter | 15.8s (n=39) | 9.3s (n=35) | 8.4s (n=8) | 10.0s (n=36) |
| TurnOffMicrowave | 12.5s (n=30) | 8.0s (n=48) | 8.6s (n=47) | 7.1s (n=14) |
| TurnOffSinkFaucet | 10.9s (n=42) | 11.6s (n=38) | 8.6s (n=37) | 8.2s (n=43) |
| TurnOffStove | 13.4s (n=5)  | 14.5s (n=11) | 35.4s (n=3) | 5.2s (n=4) |
| TurnOnMicrowave | 12.4s (n=25) | 7.2s (n=28) | 8.4s (n=27) | 4.8s (n=6) |
| TurnOnSinkFaucet | 12.4s (n=13) | 8.7s (n=29) | 8.0s (n=35) | 7.6s (n=6) |
| TurnOnStove | 16.5s (n=19) | 11.8s (n=18) | 10.5s (n=14) | 11.3s (n=9) |
| TurnSinkSpout | 7.5s (n=38) | 5.1s (n=38) | 4.6s (n=38) | 4.7s (n=17) |

### 전체 평균 (모든 task 성공 episode pool 가중평균)

| 모델 | 평균 길이 | vs 베이스라인 | 총 녹화된 성공 ep |
|------|---------:|-----:|-----:|
| baseline (16-step main) | **15.50s** (310.0f) | 1.00× | 672 |
| baseline + client merge 16→8 | **10.55s** (210.9f) | **0.68×** | 713 |
| mh_m8 + ens→8 | **9.45s** (188.9f) | **0.61×** | 571 |
| mh_m8+econsist + ens→8 | **10.05s** (200.9f) | **0.65×** | 377 |

**관찰**:
- 8-step 압축으로 평균 rollout 길이 **30~40% 감소** (이론치 50%엔 못 미치지만 압축이 task progress와 1:1로 비례 안 함)
- mh_m8 (no econsist) 가 가장 짧음 (9.45s, 0.61×). econsist 추가하면 약간 길어짐 (10.05s)
- 단순 client-side merge baseline (10.55s) 도 학습 모델(mh_m8 9.45s)보다 약간 김 → 학습 시 m8 head 직접 supervision이 약간 더 효율적
- 성공률 (606 vs 617%) 와 길이 (9.45s vs 10.05s) 모두 다 mh_m8 가 econsist보다 좋음 → 본 셋팅에선 econsist 효과 부재

**Raw data**: [analysis/rollout_durations/per_task_durations.json](../analysis/rollout_durations/per_task_durations.json)

## 관련 파일

```
run_scripts/eval/eval_mh_m8_head_ensemble_fix.sh           — mh_m8 + ensemble→8 sbatch
run_scripts/eval/eval_mh_m8_econsist_head_ensemble_fix.sh  — mh_m8+econsist + ensemble→8 sbatch
output/robocasa/mh_m8/ensemble_fix_merged8/                — mh_m8 결과
output/robocasa/mh_m8_econsist/ensemble_fix_merged8/       — mh_m8+econsist 결과
gr00t/model/action_head/flow_matching_action_head.py:800   — ensemble_fix 헤드 구현
scripts/robocasa_service_merged.py:32                       — client-side 16→8 merge 함수
```
