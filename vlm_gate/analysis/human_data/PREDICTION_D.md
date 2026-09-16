# D 를 좁히면 내려놓기 구간이 내려오나 — **돌리기 전 예측**

C 는 폭 컷으로 확정했다. 이번엔 **D 하나만** 바꾼다 (A·B·C·E·GUIDANCE 그대로, R2).

```
지금  D) Is the hand letting go, or about to -- the object already resting where it is
         meant to end up, so that what remains is only to open and withdraw?

새로  D) Is the object ALREADY RESTING where it is meant to end up, taking its own
         weight, with nothing left but to open the hand and draw back?
```

`or about to` 를 뺀다. 그 두 단어 때문에 아직 쥔 채 내려가는 중인데도 켜진다.

## 지금 상태 (폭 컷 C 에서)

```
국면          conf(pnp)  conf(long)   D
1집으러          0.541      0.352     1.0
2파지           0.565      0.365     1.0
3운반           0.745      0.560     1.1
4내려놓으러       0.769      0.660     3.3   <- 아직 안 놓였는데 켜진다
5물러남          0.829      0.670     4.3
```

C 는 이제 상수다(pnp 4.2~4.7, long 1.5~2.0). 그러므로 운반 -> 내려놓으러로 conf 를
올리는 것은 **D 뿐이다.**

## 예측

- **D 가 4내려놓으러에서 3.3 -> 1.5 아래로 떨어진다.** 물건이 아직 그리퍼에 매달려
  있으므로 "이미 놓여 제 무게를 받고 있다" 는 거짓이다.
- D 는 5물러남에서 4.0 이상을 유지한다. 거기서는 참이다.
- 그 결과 **4내려놓으러 conf 가 3운반 아래로 내려온다.** 두 데이터셋 모두.
- A·B·C·E 는 85% 이상 그대로 (C 변경 때 다른 문항이 81~92% 유지된 전례).

## 실패로 볼 조건

- 4내려놓으러가 여전히 3운반보다 높으면 D 문구만으로는 안 되는 것이고, 그때는
  쥔 채 내려놓는 순간을 받는 문항이 따로 필요하다.
- 또는 D 가 5물러남에서도 죽어버리면(2 미만) 문항 하나를 통째로 잃는 것이다.

표본은 같은 100청크. 판정기·온도·계산 사실 동일.
