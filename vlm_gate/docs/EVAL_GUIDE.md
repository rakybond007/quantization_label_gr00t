# 평가 구성법

게이트가 붙은 폐루프 평가를 세우고 돌리는 절차. RoboCasa 와 dexjoco 둘 다 같은
형태로 굴러간다 — **정책 서버 · 판정기 · 클라이언트 세 프로세스**가 한 잡 안에서
각자 다른 환경으로 돈다.

```
GPU0   정책 서버      quant_gate            정책이 16스텝 청크를 내놓음
GPU1   판정기         backend 에 따라 다름   청크마다 P(quantize) 를 답함
       클라이언트     quant_gate_eval       시뮬레이터를 돌리고 압축을 적용
```

프로세스를 나눈 이유는 환경이 충돌하기 때문이다. 클라이언트는 robocasa 때문에
numpy 1.23.5 / transformers 4.51.3 에 묶여 있고, 판정기는 백엔드에 따라 더 최신
transformers 가 필요하다. **한쪽을 올려 맞추려 하지 말 것** — 그렇게 해서 평가가
열흘 죽은 적이 있다. 필요한 라이브러리는 그 프로세스의 `PYTHONPATH` 에만 얹는다.

## 판정기 백엔드

| `JUDGE_BACKEND` | 무엇 | 언제 |
|---|---|---|
| `module` | 증류된 학생 게이트 (기본) | 대부분의 평가 |
| `cosmos` | Cosmos3-Nano VLM | 교사 자체를 폐루프에 넣어볼 때 |
| `gemma` | Gemma VLM | 구 비교군 |
| `none` | 판정기 없음 (고정 K) | 무압축·무지성 K 기준선 |

`module` 백엔드는 `MODULE_CKPT` 로 체크포인트를 받는다. **어느 체크포인트를 쓸지는
호출자가 정한다** — 숨은 기본값이 없다. 학습이 저장하는 것은 둘이다.

```
gate_module_best.pt    val AUC 가 가장 좋았던 시점    ← 평가에는 이것
gate_module.pt         마지막 에폭
```

과적합이 빠르게 오므로 최종본을 쓰면 손해다 (한 학습에서 5에폭 0.674 → 30에폭 0.554).

DINOv3 학생을 서빙하려면 `GATE_ENCODER=dinov3s` 를 준다. 그러면 런처가 판정기
프로세스에만 transformers 오버레이를 얹는다. 체크포인트에 `encoder` 필드가 저장돼
있어 서버가 어느 아키텍처인지 스스로 안다.

## RoboCasa

```bash
export MODEL_OUTPUT_DIR=/rlwrld-unified-checkpoints/hojin2/quant_gate_modules

# 스모크 — 1 어레이(3태스크) × 2에피소드
MODULE_CKPT=$WS/assets/modules_A/<학생>/gate_module_best.pt \
JUDGE_BACKEND=module TAU=0.5 N_EPISODES=2 \
OUTPUT_BASE=$WS/vlm_gate/output/robocasa/smoke_x \
srun --gpus=1 --job-name=smoke_eval_robocasa_<설명> \
     --wckey=project-short-name:sub_fast \
     --exclude=worker-node100,worker-node1,worker-node104,worker-node3 \
  bash run_scripts/eval/eval_robocasa_gated.sh

# 본평가 — 24태스크 × 50에피소드, 어레이 0-7
sbatch --export=ALL,MODULE_CKPT=...,JUDGE_BACKEND=module,TAU=0.5,N_EPISODES=50,\
OUTPUT_BASE=...,MODEL_OUTPUT_DIR=$MODEL_OUTPUT_DIR \
  --job-name=eval_robocasa_gated_<설명> run_scripts/eval/eval_robocasa_gated.sh
```

`module` 백엔드는 학생이 0.3M CNN 이라 GPU0 을 정책과 공유한다. `--gpus=1` 로 충분하다.
`cosmos`/`gemma` 는 판정기가 GPU1 을 쓰므로 `--gpus=2` 가 필요하다.

## dexjoco

같은 구조에 런처만 다르다. 태스크 6개(single-arm)라 어레이는 0-5 다.

```bash
JUDGE_BACKEND=module MODULE_CKPT=... N_EPISODES=2 SMOKE_TASK=water_plant \
srun --gpus=2 --job-name=smoke_eval_dexjoco_<설명> ... \
  bash run_scripts/eval/eval_dexjoco_gated.sh
```

압축 방식이 다르다는 점만 유의한다. **dexjoco 액션은 절대 관절 타겟이라 블록의
마지막 타겟만 남기고 건너뛴다.** RoboCasa 처럼 더하면 좌표를 두 배로 명령하게 된다.

## N1.7 정책

정책 서버가 msgpack/ZeroMQ 라 클라이언트가 그대로 못 붙는다. 어댑터가 있다.

```bash
RUN=gate|baseline N_EPISODES=2 SMOKE_TASK=OpenDrawer \
srun --gpus=2 ... bash run_scripts/eval/eval_robocasa_n17_gated.sh
```

서버 프로세스만 온라인이어야 한다 — transformers 4.57 은 캐시가 있어도 HF API 를
조회한다. 판정기는 게이트드 리포 때문에 오프라인을 유지해야 하므로 둘을 분리한다.
런처가 그렇게 해 놓았다.

## 판정 — 반드시 산출물로

**잡 상태로 성공을 판단하지 말 것.** 이 프로젝트에서 정상 종료하면서 아무 일도
하지 않은 실행이 여러 번 있었다. 어레이 인덱스가 범위를 벗어나 태스크가 하나도
선택되지 않은 채 서버만 뜨고 끝난 적이 있고 (지금은 런처가 거부한다), 게이트 손실이
어떤 파라미터에도 닿지 않은 학습이 있었다.

```bash
# 에피소드가 실제로 쌓였는가
for t in $OUTPUT_BASE/*/; do
  echo "$(basename $t): $(grep -c '^episode' $t/prediction.txt)"
done

# 게이트가 붕괴하지 않았는가 — 압축률이 태스크마다 갈려야 한다
grep -ohE "gate_quantize_rate[^ ]*" $OUTPUT_BASE/*/eval-*.log

# 판정기가 의도한 체크포인트를 실제로 로드했는가
grep "JUDGE READY" $OUTPUT_BASE/judge-*.log
```

압축률이 전부 0.00 이거나 전부 1.00 이면 게이트가 판별을 못 하고 있는 것이다.

## 결과 읽기

`prediction.txt` 를 **에피소드 줄만, 에피소드 번호로 중복 제거해서** 읽는다.
끝에 붙는 요약 줄(`is_success: 0.7600`)을 에피소드로 세는 실수가 반복됐다.

```python
m = re.match(r"episode\s+(\d+)\s+is_success:\s*\[\s*(True|False)\s*\]"
             r"\s*action_steps:\s*(\d+)", line)
```

numpy 가 `[ True]` 를 앞 공백과 함께 쓰므로 대괄호 안의 공백을 허용해야 한다.
공백을 빼먹으면 **성공한 에피소드만 조용히 사라진다.**

성공한 에피소드의 평균 스텝과 전체 평균 스텝을 **섞어 비교하지 말 것.** 한 번 틀렸다.

### 성공률만 보지 말 것

압축은 성공률과 스텝을 맞바꾼다. 그래서 "무압축보다 낮다"는 것만으로는 판단이 안
되고, **무지성 압축과 무압축을 잇는 직선 대비 얼마나 위인지**를 봐야 한다.

```
무압축      0.657 / 327 스텝
무지성 K2   0.598 / 214 스텝     기울기 0.00052 성공률/스텝

phase5·이진        0.637 / 273.7   →  직선 예측 0.629,  초과 +0.0078
phase5·사건보존형   0.627 / 248.2   →  직선 예측 0.616,  초과 +0.0111
```

성공률만 보면 사건보존형이 더 낮지만, 같은 스텝 수 기준으로는 더 위에 있다.

## 태스크가 50 에피소드를 다 돌았는지 확인할 것

`background` 는 선점 파티션이라 어레이 일부가 미완료로 끝날 수 있다. 태스크별
에피소드 수를 세지 않고 평균을 내면 5 에피소드짜리와 50 에피소드짜리를 비교하게
된다 — 실제로 그렇게 잘못된 회귀를 보고한 적이 있다.
