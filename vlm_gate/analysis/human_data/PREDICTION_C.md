# C 문항의 컷을 바꾸면 두 데이터셋이 갈리나 — **돌리기 전 예측**

## 무엇을 한 가지만 바꾸나

`C` 하나만 바꾼다. A·B·D·E 와 GUIDANCE·계산 사실은 글자까지 그대로 (R2).

```
지금  C) Is an object being put into something that would catch it -- a basket, a bin,
         a bowl -- so that landing off-centre changes nothing?
         => 컷이 "받아주느냐" 다.

새로  C) Is the place this object has to end up ROOMY FOR IT -- a wide bowl, an open
         basket, a broad plate with slack all round -- rather than a cup, a jar or a
         slot barely wider than the thing itself?
         => 컷이 "입구가 물건에 비해 넉넉하냐" 다.
```

## 왜 바꾸나 (지금 판의 측정)

72청크에서 지금 C 는 두 데이터셋을 전혀 안 가른다.

```
              A      C     A>C 인 비율
pnp  (접시)   1.07   2.43      0%
long (컵)     1.11   2.37      0%

C 등급분포  pnp  1:16 2:2 4:10 5:8
            long 1:16 2:3 4:9  5:8    <- 거의 동일
```

A 는 5등급이 72개 중 0개다. pnp 의 목적지가 **말 그대로 접시**이고 A 가 예시로
`a plate` 를 드는데도 1.07 이다 -- 뒷절 `rather than into something that would
catch it` 이 앞 예시를 무효로 만든 것으로 보인다. 그래서 접시도 컵도 전부
"받아주는 쪽" 으로 읽혔다.

## 예측

- **pnp 의 C 가 long 보다 확실히 높게 나온다.** 흰 접시는 넓고, 빨간 컵은 물건만 하다.
  등급 평균 차이 **1.0 이상**을 기대한다(지금은 0.06).
- 놓는 국면(`쥐거나놓는중`)에서 차이가 가장 크다.
- B·D·E 는 거의 안 바뀐다. 다만 사실 문장이 아니라 문항을 바꾸는 것이므로 A 가
  같이 흔들릴 수 있다(A 와 C 는 같은 축의 양변이다).
- conf 의 두 데이터셋 순서가 실측 유지율(pnp 0.937 > long 0.571)과 같아진다.

## 실패로 볼 조건

- C 의 두 데이터셋 평균 차이가 0.5 미만이면 이 컷도 안 듣는 것이다.
- 또는 C 가 한쪽으로 쏠려(전부 1 또는 전부 5) 분산이 죽으면 역시 실패다.

표본은 같은 72청크(probe3 와 동일한 ep·f)를 쓴다. 판정기·온도·사실 모두 동일.
