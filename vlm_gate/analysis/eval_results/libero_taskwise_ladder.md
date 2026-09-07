# libero 태스크별 사다리 — 문항 뽑기 재료

`libero_ladder_noclip.md` 의 다섯 칸을 태스크 40개로 편 것. 태스크당 50에피.
1.667e = `[2,2,1]`(고르게), 1.667f = `[3,1,1]`(앞으로 몰림).

**2.5배가 가장 잘 갈린다** — 0.02 부터 1.00 까지 벌어진다. 1.667·2.0 은 안전 쪽이
천장에 붙어 있어 순서가 안 나온다. 그래서 무리는 2.5배로 가른다.

## 위험 (2.5배 ≤ 0.30)

| 2.5배 | 태스크 | 지시문 |
|---:|---|---|
| 0.02 | libero_10/1 | put both the cream cheese box and **the butter** in the basket |
| 0.10 | spatial/3 | black bowl on the cookie box -> **접시 위** |
| 0.16 | goal/3 | open the top drawer and put the bowl **inside** |
| 0.16 | libero_10/8 | put both moka pots **on the stove** |
| 0.18 | goal/1 | put the bowl **on the stove** |
| 0.20 | spatial/0 | black bowl between the plate and the ramekin -> **접시 위** |
| 0.22 | goal/9 | put the wine bottle **on the rack** |
| 0.22 | spatial/5 | black bowl on the ramekin -> **접시 위** |
| 0.28 | spatial/8 | black bowl next to the plate -> **접시 위** |

## 안전 (2.5배 ≥ 0.86)

| 2.5배 | 태스크 | 지시문 |
|---:|---|---|
| 1.00 | object/0 | alphabet soup -> **바구니** |
| 1.00 | object/7 | milk -> **바구니** |
| 1.00 | goal/5 | **push** the plate to the front of the stove |
| 0.94 | object/9 | orange juice -> **바구니** |
| 0.92 | goal/4 | put the bowl **on top of the cabinet** |
| 0.92 | object/5 | tomato sauce -> **바구니** |
| 0.92 | libero_10/3 | black bowl **in the bottom drawer** and close it |
| 0.90 | object/3 | bbq sauce -> **바구니** |
| 0.90 | goal/2 | wine bottle **on top of the cabinet** |
| 0.88 | goal/7 | **turn on** the stove |
| 0.86 | libero_10/5 | book in the **back compartment of the caddy** |
| 0.86 | libero_10/7 | alphabet soup + cream cheese box -> **바구니** |

## 무엇이 가르는가

### 1. 물체가 아니라 **놓을 자리**가 정한다

같은 검은 그릇을 옮기는데 목표에 따라 갈린다.

```
그릇 -> 스토브 버너 위      0.18   위험
그릇 -> 접시 위      0.10~0.68   위험
그릇 -> 연 서랍 안          0.16   위험

그릇 -> 캐비닛 윗면         0.92   안전
그릇 -> 아래 서랍에 넣고 닫기 0.92   안전
```

와인병도 같다 — 걸이(rack) 0.22, 캐비닛 윗면 0.90.

**좁고 정해진 한 점에 반듯이 올려야 하는가, 아니면 받아 주는 통이거나 넓은 면인가.**
이것이 지금 가장 강한 신호이고, 기존 v1 문항 B 가 정확히 이것을 묻고 있다.

`object` 열 개는 전부 바구니에 떨어뜨리는 것이고 2.5배에서도 0.58~1.00 으로 버틴다.

### 2. 반례가 두 번째 문항을 준다 — 쥘 자리

"물건 둘을 바구니에" 세 개가 0.02 부터 0.86 까지 벌어진다.

```
10/7   알파벳수프 + 크림치즈 상자 -> 바구니    0.86
10/0   알파벳수프 + 토마토소스   -> 바구니    0.48
10/1   크림치즈 상자 + 버터     -> 바구니    0.02
```

크림치즈 상자는 /1 과 /7 양쪽에 있으므로 가르는 것은 **버터**다 — 작고 납작해서
쥘 자리가 좁다. 목표는 똑같이 바구니인데 집는 데서 깨진다.

### 3. 옮기지 않는 일은 안 깨진다

`goal/5` 밀기 1.00, `goal/7` 손잡이 돌리기 0.88. 물체를 들어 옮기지 않는 일은
2.5배에서도 멀쩡하다. 표본이 둘뿐이라 약하지만 부호는 깨끗하다.

## 쓰면 안 되는 갈래

**"서랍·캐비닛 안으로 들어간다"** 는 갈리지 않는다. `goal/3`(위 서랍에 넣기) 0.16
이 위험인데 `libero_10/3`(아래 서랍에 넣고 닫기) 0.92, `libero_10/5`(캐디 칸) 0.86
이 안전이다. 입구 통과 자체가 아니라 그 안의 자리가 얼마나 좁은지가 정한다.
v1 문항 C 가 이것을 묻고 있으므로 손봐야 한다.

**"물체가 그릇이다 / 작다"** 도 갈리지 않는다. 같은 그릇이 목표에 따라 0.18 과
0.92 로 갈린다. 물체 성질만 묻는 문항은 목표에 얽혀 있어 혼자 서지 못한다.

## 블록 모양도 같이 봐야 한다

1.667e(`[2,2,1]`)와 1.667f(`[3,1,1]`)는 배속이 같은데 `spatial/3` 이 0.94 대 0.44,
`spatial/8` 이 0.98 대 0.54 로 갈린다. **한 명령에 3스텝을 몰아 넣으면 평균 배속이
같아도 깨진다.** 상한은 배속 하나로 적을 수 없고 블록 모양을 같이 적어야 한다.
