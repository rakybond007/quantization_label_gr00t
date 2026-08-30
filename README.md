# quantization label — GR00T 행동 청크 압축 가능성 라벨링

행동 청크(16스텝)마다 **이 구간을 절반 속도로 실행해도 되는가**를 판단하는 게이트를
만든다. VLM 교사로 정답지를 만들고, 작은 학생 모듈에 증류하고, 시뮬레이터 폐루프에서
검증한다. GR00T-N1.5 기준.

되는 구간만 골라 압축하면 성공률을 지키면서 스텝을 줄인다. RoboCasa 24태스크 기준
무압축 0.656 / 330스텝, 무지성 K2 0.599 / 216스텝, 게이트 0.627~0.642 / 252~276스텝.
**얼마나가 아니라 어디를 압축하냐**가 성능을 가른다.

**이 레포 하나면 된다.** GR00T 본체(`gr00t/`, `scripts/`)와 게이트 작업(`vlm_gate/`)이
같이 들어 있어 따로 받아 싱크할 것이 없다. 업스트림 NVIDIA README 는
[README_upstream_gr00t.md](README_upstream_gr00t.md).

## 네 층

```
1. 임베디먼트 어댑터   액션이 델타냐 절대 목표냐에 따라 압축 연산이 갈린다
                       RoboCasa·LIBERO = 인접 스텝을 더함 (델타)
                       dexjoco·allex    = 중간 목표를 버림 (절대)
2. 결정론적 계산층     액션 숫자만으로 알 수 있는 위험 — 그리퍼 개폐, 경로 꺾임,
                       정밀 유지, 병합 실현가능성. VLM 에게 묻지 않는다.
3. VLM 판단층          계산으로는 원리적으로 알 수 없는 것만 묻는다 — 지금 무엇을
                       상대하고 있는 장면인가. 한 청크에 한 번 호출.
4. 집계·증류           noisy-OR 로 합치고 순위정규화 → 학생 모듈 학습
```

3층에서 계산이 이미 아는 것을 다시 물으면 모델은 사실을 복창할 뿐이고 답이 두 종류로
붕괴한다. 그래서 계산값은 **사실 문장으로 알려주고 묻지는 않는다.**

## 어디에 뭐가 있나

| 경로 | 내용 |
|---|---|
| `gr00t/`, `scripts/` | GR00T 본체. 학습(`scripts/gr00t_finetune.py`)·서빙(`scripts/serve_policy.py`) |
| `vlm_gate/scripts/*_descriptors*.py` | 2층. 벤치마크별 결정론적 위험 기술자 |
| `vlm_gate/analysis/_evolver/` | 3층. 가이던스와 문항 |
| `vlm_gate/scripts/cosmos_1call_v6.py` | 3층 실행. 답 슬롯 확률을 읽는다 |
| `vlm_gate/scripts/aggregate_*.py` | 4층. 집계 → 라벨 파케이 |
| `vlm_gate/scripts/recompute_soft_*.py` | 계산 플래그 연속화 (VLM 재호출 없음) |
| `vlm_gate/scripts/train_gate_module.py` | 학생 모듈 학습 (CNN / DINOv3) |
| `vlm_gate/scripts/*_service_compress.py` | 폐루프 평가. 게이트를 물려 압축 실행 |
| `vlm_gate/run_scripts/` | 위를 실제로 돌린 slurm 잡 스크립트 |
| `bin/qgate`, `vlm_gate/qgate/` | 결과를 읽고 검증하는 도구 |

## 프롬프트

라벨링 품질을 가르는 건 프롬프트다. `GUIDANCE` 로 세대를 고른다
(`vlm_gate/scripts/cosmos_1call_v6.py`).

| `GUIDANCE` | 가이던스 | 문항 |
|---|---|---|
| `phase5` | `_varkA/robocasa_guidance_phase_v3.txt` | 4문항 |
| `phase6` | `_varkA/robocasa_guidance_phase_v5.txt` | 5문항 (다섯 축) |

LIBERO 와 dexjoco 는 `_libero/`, `_dexjoco/` 아래에 가이던스와 문항이 따로 있다.

## 도는 순서

```bash
# 1. 라벨링 (slurm 배열)
sbatch vlm_gate/run_scripts/label/sbatch_phase6_full.sh

# 2. 학습 전에 반드시 검증한다 — 잡이 COMPLETED 여도 증거가 아니다
bin/qgate labels v6b_phase6_s16 --expected 247887

# 3. 집계 → 계산 플래그 연속화
python vlm_gate/scripts/aggregate_phase6.py
python vlm_gate/scripts/recompute_soft_phase6.py

# 4. 학생 학습 (스모크 먼저)
bash  vlm_gate/run_scripts/train/_smoke_train_phase6.sh
sbatch vlm_gate/run_scripts/train/sbatch_train_phase6_softA.sh

# 5. 폐루프 평가 → 판정
bin/qgate tradeoff robocasa --fast baseline_compress_K2 \
    --slow baseline_full_v2_with_action_steps
```

2번을 건너뛰면 안 된다. 라벨링 잡은 마지막 명령이 `kill` 이라 무슨 일이 있었든
종료코드 0 으로 끝나고, 선점 후 재큐된 샤드는 이미 쓴 줄을 다시 뱉는다. 판정은
디스크의 행에서 나와야 한다.

## 결과 읽기

결과는 이 레포에 담지 않는다. 워크스페이스를 가리키면 된다.

```bash
export QGATE_WS=/path/to/quantization_agent_workspace
bin/qgate results robocasa
```

**성공률만으로는 게이트 순위를 매길 수 없다.** 압축은 성공률을 속도와 맞바꾸므로
아무것도 압축하지 않는 게이트가 1등이 되고 그건 쓸모가 없다. 무압축과 무지성 압축을
잇는 직선이 공짜로 얻는 거래이고, 그 위로 얼마나 올라갔는지만 근거가 된다. 함정들은
[docs/TOOLKIT.md](docs/TOOLKIT.md), 가중치 위치는 [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md).

## 담지 않은 것

평가 결과, 라벨 파케이, 학습된 가중치, 로그. 가중치는 HuggingFace 에 있고
`docs/CHECKPOINTS.md` 가 경로와 주소를 모두 적어 둔다.
