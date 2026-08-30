# quantization label — GR00T 행동 청크 압축 가능성 라벨링

행동 청크(16스텝)마다 **이 구간을 절반 속도로 실행해도 되는가**를 판단하는 게이트를
만들기 위한 코드다. VLM 교사로 정답지를 만들고, 작은 학생 모듈에 증류하고,
시뮬레이터 폐루프에서 검증한다. GR00T-N1.5 기준이다.

되는 구간만 골라 압축하면 성공률을 지키면서 스텝을 줄인다. RoboCasa 24태스크 기준
무압축 0.656 / 330스텝, 무지성 K2 0.599 / 216스텝, 게이트 0.627~0.642 / 252~276스텝.
**얼마나가 아니라 어디를 압축하냐**가 성능을 가른다.

## 네 층

```
1. 임베디먼트 어댑터   액션이 델타냐 절대 목표냐에 따라 압축 연산이 갈린다
                       RoboCasa·LIBERO = 인접 스텝을 더함 (델타)
                       dexjoco·allex    = 중간 목표를 버림 (절대)
2. 결정론적 계산층     액션 숫자만으로 알 수 있는 위험 — 그리퍼 개폐, 경로 꺾임,
                       정밀 유지, 병합 실현가능성. VLM 에게 묻지 않는다.
3. VLM 판단층          계산으로는 원리적으로 알 수 없는 것만 묻는다 — 지금 무엇을
                       상대하고 있는 장면인가. 한 이미지에 한 번 호출.
4. 집계·증류           noisy-OR 로 합치고 순위정규화 → 학생 모듈 학습
```

3층에서 계산이 이미 아는 것을 다시 물으면 모델은 사실을 복창할 뿐이고 답이 두 종류로
붕괴한다. 그래서 계산값은 **사실 문장으로 알려주고 묻지는 않는다.**

## 구성

| 경로 | 내용 |
|---|---|
| `vlm_gate/scripts/*_descriptors*.py` | 2층. 벤치마크별 결정론적 위험 기술자 |
| `vlm_gate/analysis/_evolver/` | 3층. 실제로 쓰는 가이던스와 문항 |
| `vlm_gate/scripts/cosmos_1call_v6.py` | 3층 실행. 한 청크 한 호출, 답 슬롯 확률을 읽는다 |
| `vlm_gate/scripts/aggregate_*.py` | 4층. 집계 → 라벨 파케이 |
| `vlm_gate/scripts/recompute_soft_*.py` | 계산 플래그 연속화 (VLM 재호출 없음) |
| `vlm_gate/scripts/train_gate_module.py` | 학생 모듈 학습 (CNN / DINOv3) |
| `vlm_gate/scripts/*_service_compress.py` | 폐루프 평가. 게이트를 물려 압축 실행 |
| `vlm_gate/run_scripts/` | 위를 실제로 돌린 slurm 잡 스크립트 |
| `bin/qgate` · `vlm_gate/qgate/` | 결과를 읽는 도구 |

## 프롬프트

라벨링 품질을 가르는 건 프롬프트다. 현재 쓰는 것만 담았고 진화 이력은 뺐다.

| 파일 | 쓰임 |
|---|---|
| `_varkA/robocasa_guidance_phase_v3.txt` | RoboCasa phase5 — 4문항 |
| `_varkA/robocasa_guidance_phase_v5.txt` | RoboCasa phase6 — 다섯 축, 5문항 |
| `_libero/libero_guidance_v1.txt`, `libero_questions_v1.txt` | LIBERO |
| `_dexjoco/dexjoco_guidance_v1.txt`, `dexjoco_questions_v1.txt` | dexjoco |

`GUIDANCE=phase6` 이면 v5 + 5문항, `phase5` 면 v3 + 4문항으로 갈린다
(`scripts/cosmos_1call_v6.py`).

## 결과 읽기

```bash
export QGATE_WS=/path/to/quantization_agent_workspace   # 결과가 이 체크아웃 밖에 있을 때
bin/qgate results robocasa
bin/qgate tradeoff robocasa --fast baseline_compress_K2 --slow baseline_full_v2_with_action_steps
```

**성공률만으로는 게이트 순위를 매길 수 없다.** 압축은 성공률을 속도와 맞바꾸므로,
아무것도 압축하지 않는 게이트가 1등이 되고 그건 쓸모가 없다. 무압축과 무지성 압축을
잇는 직선이 공짜로 얻는 거래이고, 그 위로 얼마나 올라갔는지만 근거가 된다.
자세한 내용과 함정은 [docs/TOOLKIT.md](docs/TOOLKIT.md).

체크포인트 위치와 HuggingFace 주소는 [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md).

## 담지 않은 것

- **평가 결과·라벨 파케이·학습된 가중치.** 가중치는 HuggingFace 에 있고
  (`docs/CHECKPOINTS.md`), 결과는 워크스페이스에 남는다.
- **일회성 프로브와 폐기된 변형.** 잡 스크립트가 부르는 코드와 그 임포트만 남겼다
  (137개 중 62개).
- **가이던스 진화 이력.** 현재 쓰는 판만 담았다.
- `gr00t_finetune.py`, `serve_policy.py`, `serve_policy_dexjoco.py` 는 Isaac-GR00T
  쪽 파일이라 여기 없다. 잡 스크립트가 그 경로를 참조한다.
