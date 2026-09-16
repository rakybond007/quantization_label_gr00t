# C 문항의 컷을 바꾼 결과 — 예측대로 갈린다

예측은 `PREDICTION_C.md` 에 **돌리기 전에** 적었다. 같은 72청크, 같은 판정기·온도·
계산 사실. **C 한 문항만** 바꿨다(A·B·D·E·GUIDANCE 는 글자까지 동일, R2).

```
기존  C) Is an object being put into something that would catch it -- a basket, a bin,
         a bowl -- so that landing off-centre changes nothing?

변형  C) Is the place this object has to end up ROOMY FOR IT -- a wide bowl, an open
         basket, a broad plate with slack all round -- rather than a cup, a jar or a
         slot barely wider than the thing itself?
```

## 결과

| | pnp (넓은 접시) | long (좁은 컵) | 차이 |
|---|---|---|---|
| C 기존 | 2.78 | 2.72 | 0.06 |
| **C 변형** | **4.47** | **1.69** | **2.78** |

등급 분포가 겹치는 구간 없이 갈린다.

```
기존   pnp  1:16 2:2 4:10 5:8      long  1:16 2:3 4:9 5:8     <- 거의 동일
변형   pnp  4:19 5:17              long  1:11 2:25            <- 겹침 없음
```

**한 문항만 바꾼 것이 그 문항에만 작용했다.** 같은 등급 비율: A 89% · B 85% ·
D 81% · E 92% (C 는 24%). 사실 문장을 건드렸을 때 네 문항이 다 흔들렸던 것과 다르다.

conf 차이도 방향이 맞게 벌어졌다.

```
기존   pnp 0.561  long 0.550   차이 0.012
변형   pnp 0.671  long 0.484   차이 0.186     실측 유지율 차 0.366 (pnp 0.937 > long 0.571)
```

## 한계 — 이 결과로 말할 수 있는 것과 없는 것

**말할 수 있다:** 문항이 설계대로 켜진다. "받아주느냐" 컷에서 넓은 접시와 좁은 컵이
같은 편에 떨어지던 것이, "입구가 물건에 비해 넉넉하냐" 컷에서는 갈린다.

**말할 수 없다:**

1. **유지율을 예측한다고 말할 수 없다.** 데이터셋이 둘뿐이라 점이 두 개다. 순위상관을
   낼 수도, 오차를 붙일 수도 없다.
2. **C 가 포화했다.** pnp 는 전부 4~5, long 은 전부 1~2 다. 지금의 C 는 "어느
   데이터셋이냐" 를 말하는 지표에 가깝고 한 데이터셋 **안에서** 순간을 가르지 못한다.
   다만 이 두 데이터셋은 목적지가 에피소드 내내 한 종류뿐이라(접시만 / 컵만) C 가
   상수인 것이 사실과 맞다 -- 빠져 있던 **데이터셋 사이 순서**를 이 문항이 채웠다.
   순간 분해력은 여전히 B·D 몫이다.
3. 목적지가 여럿인 벤치마크(libero 는 접시·바구니·선반이 섞인다)에서 이 컷이 같은
   값을 하는지는 따로 재야 한다. 여기서 확인한 것은 이 두 데이터셋에 대해서다.

## A 는 어떻게 되었나

여전히 안 켜진다(1.06 / 1.00, 5등급 0개). A 와 C 는 같은 축의 양변인데 C 가 폭으로
컷을 옮겼으므로 A 의 "작고 고정된 자리" 컷은 이제 중복이면서 안 걸린다. A 를 폭
기준의 반대편으로 다시 쓸지는 다음 문제다 -- 이번 판에서는 손대지 않았다(R2).

## 재현

```
HD_PER_DS=36 python scripts/humandata_gemini_probe.py out.jsonl              # 기존
HD_C_WIDTH=1 HD_PER_DS=36 python scripts/humandata_gemini_probe.py out.jsonl # 변형
```
