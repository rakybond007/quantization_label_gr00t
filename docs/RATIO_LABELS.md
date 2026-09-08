# 배속 라벨 — 받아서 쓰는 법

시점마다 **몇 배속으로 갈지**가 적힌 라벨이다. 이 저장소를 클론하고 데이터셋만
있으면 어느 기계에서나 쓸 수 있다. 우리 서버 경로도, GPU 도, VLM 도 필요 없다.

```bash
git clone git@github.com:rakybond007/quantization_label_gr00t.git
cd quantization_label_gr00t

# 붙는지 먼저 본다 (아무것도 쓰지 않는다)
python vlm_gate/scripts/apply_ratio_labels.py \
    --labels assets/labels/libero/libero_v2_ratio.parquet \
    --dataset /내/경로/lerobot/libero_gr00t_delta \
    --dry-run

# 사본을 만들면서 붙인다
python vlm_gate/scripts/apply_ratio_labels.py \
    --labels assets/labels/libero/libero_v2_ratio.parquet \
    --dataset /내/경로/lerobot/libero_gr00t_delta \
    --out     /내/경로/libero_with_ratio
```

`pandas` 와 `pyarrow` 만 있으면 된다.

## 무엇이 붙나

에피소드·프레임마다 열 넷이 붙는다.

| 열 | 뜻 |
|---|---|
| `ratio` | 이 시점에 쓸 배속. **1.0 · 1.5 · 2.0 · 2.5 중 하나** |
| `ratio_conf` | 그 배속을 정한 신뢰도 0~1. 확률이 아니라 **순서** 값이다 |
| `ratio_fixed` | 접촉이라 1.0 으로 고정된 시점이면 1 |
| `ratio_valid` | 라벨이 있는 시점이면 1 |

**`ratio_valid` 가 0 인 행은 손실에서 빼야 한다.** 라벨이 없는 시점을 "1배속이
정답" 으로 학습시키면 안 된다. `ratio` 는 그 자리에 1.0 이 들어가 있는데 그것은
기본값이지 판정이 아니다.

## 배속이 왜 이산인가

디코더가 이산이다. 연속값을 라벨로 적으면 결국 반올림해서 쓰게 되고, **그러면
결정이 반올림에서 일어난다.** 처음부터 디코더가 낼 수 있는 값으로 적는다.

그 눈금에서 배속이 정확히 떨어지는지는 확인했다. 16스텝 청크를 고정 패턴으로
묶고 재예측 창을 장부로 맞추면 이렇게 나온다.

| K | 16스텝 블록 | 실제 배속 |
|---:|---|---:|
| 1 | `[1]*16` | 1.000 |
| 1.5 | `[1,2,1,2,1,2,1,2,1,2,1]` | 1.500 |
| 2 | `[2]*8` | 2.000 |
| 2.5 | `[2,3,2,3,2,3,1]` | 2.500 |

자세한 것은 `vlm_gate/analysis/eval_results/RATIOS_ARE_NOT_K.md`. **K 는 배속이
아니다** — 세 군데서 이름과 실제가 어긋나 있었고 그 때문에 결론이 뒤집힌 적이 있다.

3배·4배는 안 쓴다. 상한이 눈금 최댓값에서 포화하므로 라벨에 아무것도 더해 주지
않는다.

## 라벨이 어떻게 만들어졌나

두 갈래를 합친 하나의 값이다.

```
접촉이 잡히면      ->  1.0배 고정
접촉이 아니면      ->  band_place( VLM 신뢰도, 그 태스크의 [1.0, 상한] )  ->  눈금으로 스냅
```

**접촉 판정은 계산으로만 한다.** 그리퍼가 열리거나 닫히는 순간(무게가 옮겨 가는
때)과 쥔 채 느리게 가는 구간(놓을 자리를 맞추는 중)이다. VLM 에게 묻지 않는다.

**신뢰도는 VLM 이 문항 다섯에 매긴 1~5 등급에서 나온다.**

```
등급 -> 값     (g − 1) / 4
확신           ( 1 + Σw·가점 − Σw·감점 ) / 2
```

문항과 가중치는 **측정된 손상**에서 나왔지 장면 인상으로 지은 것이 아니다.
전량은 `vlm_gate/prompts/`, 뽑은 과정은 `vlm_gate/analysis/libero_prompt_derivation.md`.

**상한은 문항이 정하지 않는다. 평가가 준다.** 태스크마다 배속을 올려 가며 성공률과
스텝을 재서, 성공률이 유지되고 스텝이 실제로 줄어드는 가장 높은 배속이 상한이다
(`vlm_gate/scripts/derive_libero_ceilings.py`). 문항이 정하는 것은 **그 상한을
얼마나 쓸 것인가** 뿐이다.

**신뢰도는 태스크 안에서 분위수로 앉힌다.** 태스크를 섞으면 어려운 태스크의 쉬운
순간이 쉬운 태스크의 어려운 순간보다 높은 배속을 받는다.

## 다시 만들려면

라벨을 새로 뽑을 일은 보통 없다. 문항을 바꿨을 때만이다.

```bash
# 1. 평가 사다리에서 태스크별 상한표
python vlm_gate/scripts/derive_libero_ceilings.py
#    -> vlm_gate/analysis/libero_task_ceilings.json

# 2. VLM 라벨(등급) + 상한표 -> 배속
python vlm_gate/scripts/libero_ratio_label.py <등급 labels.jsonl>
#    -> <...>_ratio.jsonl
```

상한표는 **명령 클리핑만 푼** 사다리로 만든다. 구동부 한계까지 3배로 연 실행은
쓰지 않는다 — libero 에서 1.429배가 이미 0.616 으로 무너져 압축이 아니라 제어기
손상을 재고 있었다(2026-09-08, 각 2000에피).

## 남이 낸 결과를 처음 가져다 쓸 때

먼저 감사를 돌린다.

```bash
python -m qgate.audit <실행 디렉터리> --expect 50
```

중복 에피소드 · 에피소드 수 · 헤더 복수 · **헤더와 재계산 대조** 넷을 본다.
넷째가 이 도구가 있는 이유다. 선점·재개될 때마다 `Total success rate:` 헤더가
덧붙는데 마지막 헤더가 전체를 반영한다는 보장이 없다. robocasa 에서 헤더를 읽고
계산했다가 "구동 한계를 풀면 압축 손상이 회복된다(t +2.37)" 라는 없는 결론이
나왔고, 에피소드 줄에서 다시 세니 사라졌다.

## 학습 쪽에서 알아야 할 것

- F_level 의 `level_ks` 를 **`(1, 1.5, 2, 2.5)`** 로 맞춘다. 코드 기본값은
  `(1, 2, 3, 4)` 이고 그 눈금으로 학습된 체크포인트는 이 라벨과 안 맞는다.
- 디코더의 블록 경계는 `flare/action_head_flevel.py:block_sizes` 이고, 우리가
  평가·라벨에 쓰는 것과 같은 규칙이다(`vlm_gate/scripts/fractional_blocks.py`).
- `ratio_valid == 0` 인 행은 손실에서 뺀다.
