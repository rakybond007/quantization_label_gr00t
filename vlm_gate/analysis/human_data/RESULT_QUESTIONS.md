# human_data 두 데이터셋용 문항 — 세 번 고쳐서 도달한 판

libero 정본에서 출발해 **한 번에 하나씩** 고쳤다(R2). 표본 100청크, Gemini
gemini-3.8-flash(low). 국면은 그리퍼 전이에서 찍는다.

**유지율 숫자는 쓰지 않았다.** 실기 eval 이라 압축이 잘 되는 것처럼 보일 수 있고,
거기 맞춰 깎으면 그 숫자에 과적합된다. 판정은 물리적으로 자명한 순서로만 한다:

```
집으러 내려감 · 파지 · 내려놓으러   <  운반 · 놓고 물러남
```

## 바꾼 것 셋

### 1. C 의 컷을 "받아주냐" 에서 "입구가 넉넉하냐" 로

```
전  C) ... put into something that would catch it -- a basket, a bin, a bowl --
       so that landing off-centre changes nothing?
후  C) Is the place this object has to end up ROOMY FOR IT -- a wide bowl, an open
       basket, a broad plate with slack all round -- rather than a cup, a jar or a
       slot barely wider than the thing itself?
```

기존 컷은 **좁은 컵을 넓은 사발보다 너그럽다고 말했다** (내려놓기 국면에서 long
4.90 vs pnp 4.70). 그 자체로 오답이다. 폭 컷은 둘 다 맞춘다.

```
C 평균   전  pnp 2.78  long 2.72  (차 0.06)
        후  pnp 4.34  long 1.82  (차 2.52)   분포 겹침 없음
```

C 는 한 데이터셋 안에서는 상수다(사발만 / 컵만). 그게 사실과 맞다 -- C 의 일은
**데이터셋 사이 순서**이지 국면 순서가 아니다.

### 2. D 에서 `or about to` 를 뺀다

```
전  D) Is the hand letting go, or about to -- the object already resting ...
후  D) Is the object ALREADY RESTING where it is meant to end up, taking its own
       weight, with nothing left but to open the hand and draw back?
```

`or about to` 때문에 아직 쥔 채 내려가는 중에도 켜졌다(내려놓으러 3.55). 좁히니
2.00 으로 내려오고 운반->내려놓으러 간격이 절반이 됐다(long 0.100 -> 0.050).

### 3. F 를 더한다 — B 의 거울

```
F) Is the robot LOWERING what it still holds onto its target right now -- the object
   still in the hand, coming down onto the spot it must end up, not yet let go?
```
부호 **−1**.

**왜 더해야 하는지는 분해로 나왔다.** D 를 좁혀도 내려놓으러가 운반보다 높았고,
그 상승분이 거의 전부 D 였다(long: D +0.073 · C +0.027 · 나머지 0). **D 는 부호가
+1 이라 "안 올리는 것" 까지가 한계이고 내리지는 못한다.**

부호를 물리에 고정하고(B·F 감점 · C·D 가산) 실제로 켜지는 문항만으로 선형
실현가능성을 풀면:

```
B·C·D 만      **불가능**
B·C·D·F       가능   v = B −0.06  C +0.00  D +0.03  F −0.08
```

F 없이는 어떤 말이 되는 가중으로도 이 순서를 못 만든다. F 가 가장 큰 몫을 가져간다.

F 는 설계대로 켜진다: 내려놓으러 4.00 · 운반 1.90 · 물러남 2.85 · 집으러 1.15.

## A·E 는 그대로 둔다

이 두 데이터셋에서 안 켜진다(A 1.0~1.4 · E 1.0~1.4). 그러나 **버릴 이유가 아니다** --
전 청크가 같은 등급이면 conf 에 상수가 더해질 뿐 순위를 안 흔든다(빼고 계산하면
순위상관 +0.998). 문항은 원래 그 벤치마크의 측정된 피해에서 뽑는 것이고, 밀기·
손잡이·서랍이 없는 임바디먼트에서 E 가 안 켜지는 것은 정상이다.

**다만 LP 에 넣으면 안 된다.** 상수에 가까운 문항의 잡음을 긁어 "가능" 을 만들어
낸다 -- 실제로 F 에 0.00 을 주고 E 에 +0.85 를 몬 해가 나왔다.

## 아직 안 정한 것

- **가중.** 라벨링 뒤에 이 임바디먼트에서 잡는다(R5). 위의 v 는 실현가능성을 보인
  것이지 쓸 값이 아니다.
- **pnp 의 내려놓으러**는 잠정 가중에서 운반과 동점이었다(0.661 vs 0.661). 사발이
  정말 너그러우니 그게 틀렸다고 단정하기 어렵다. 가중을 잡을 때 결정된다.
- **libero 에도 같은 수정이 필요한지.** 이 프롬프트는 두 데이터셋 전용이다. libero 는
  목적지가 섞여 있어(바구니 13 · 사발/접시 8 · 머그/통 3) 따로 재야 한다.

## 재현

```
HD_C_WIDTH=1 HD_D_TIGHT=1 HD_F_MIRROR=1 HD_PER_DS=50 \
  python scripts/humandata_gemini_probe.py out.jsonl
```
예측은 PREDICTION_C.md · PREDICTION_D.md · PREDICTION_F.md 에 매번 돌리기 전에 적었다.
