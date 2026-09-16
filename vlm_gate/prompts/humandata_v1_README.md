# humandata v1 — 리얼 Franka 두 데이터셋 전용 판

`pnp_task` · `long_horizon_task` (`/sjw_alinlab2/home/taekwan/Data/human_data/`).
**libero v3c 에서 출발했지만 같은 것이 아니다.** 섞어 쓰면 안 되므로 파일을 나눈다.

## libero v3c 와 다른 곳

| | libero v3c | humandata v1 |
|---|---|---|
| 이미지 | 2장 (front · left_wrist) | 2장 (exterior_1_left · wrist_left) |
| 액션 공간 | 델타 EEF | **joint_pos_abs** (절대 관절 7 + 연속 그리퍼 1) |
| 병합 식 | Σ K개 델타 | **max‖a[i+K] − a[i]‖** (블록-라스트) |
| 그리퍼 | 이진 | **연속 0~1**, 문턱을 분포에서 뽑음 |
| 계산 사실 | 압축 요구 문장 **없음** | **`at 2x ... / at 3x ...` 두 문장 있음** |
| 문항 C | "받아주는 것 안에 넣나" | **"입구가 물건에 비해 넉넉한가"** |
| 문항 D | `letting go, or about to` | **`ALREADY RESTING ... taking its own weight`** |
| 문항 F | 없음 | **있음** (쥔 채 내려놓는 중, 부호 −1) |

문항 A·B·E 와 GUIDANCE 는 libero v3c 와 같다.

## 압축 요구 두 문장에 대하여

```
at 2x this stretch would move 0.034 rad in one step, smaller than most
at 3x 0.049 rad, smaller than most
```

allex 가 쓰는 형태를 그대로 가져왔다. **배율(2x·3x)을 문장에 박는다는 점은 약점이다**
-- 라벨링 시점에 몇 배속을 걸지 모르기 때문이다. 두 배율을 다 주므로 한 쪽에 고정된
것은 아니고, 비교 기준도 같은 K 의 분포다(k2 는 k2 분포). 그래도 배율 없는 표현으로
쓸 수 있었다.

**allex 48,086 청크와 humandata 2,331 청크가 이 문장을 달고 만들어졌다.** 둘이 같은
형태라 서로는 일관되고, robocasa·libero 는 이 문장이 아예 없다.

## 문턱은 데이터셋마다 다르다

`humandata_quantiles.py` 가 각 데이터셋 자기 분포에서 낸다. 손으로 적지 않는다 --
처음에 속도 3단을 0.004/0.012, 그리퍼를 0.4 로 지어냈다가 고쳤다. 실제는:

```
              속도 3분위        그리퍼 골   k2 분위(20/40/60/80)
pnp_task      0.0152 / 0.0251    0.70      .044 .062 .078 .100
long_horizon  0.0123 / 0.0200    0.38      .033 .046 .061 .077
```

## 라벨

`humandata_label_full.py` · Gemini 3.8 Flash(low) · stride 16 · 2,331 청크 · $6.61.
결과는 `analysis/human_data/` 에 있다.
