# robocasa v10 후보 — A 를 순간 문항으로 좁힌다 (**돌리기 전 예측**)

v9 에서 **A 한 문항만** 바꾼다. B·C·D·E·F·GUIDANCE·척도는 글자까지 그대로 (R2).

```
v9   A) Does this motion need the gripper to CLOSE on one exact spot -- a handle,
        a rim, the body of an object it must pick up -- rather than only pushing or
        turning something that stays where it is?

v10  A) Is the gripper CLOSING on its target right now, or about to -- the fingers
        already around the handle, rim or object and only the squeeze left, so that
        a slip here loses it?
```

## 왜 바꾸나 (측정)

`approach` 의 실측 상한이 **4.00 으로 전체 최고**인데 conf 가 가장 낮은 축이다.

```
국면        실측 상한   v9 A 등급   v9 conf
접근          4.00        3.01       0.507   <- 가장 압축 가능한데 가장 낮다
운반          3.73        1.12       0.668
해제(놓음)     3.13~3.20   1.19       0.572
파지          1.67~2.00   3.60       0.438
해제(쥔채)     1.25        1.33       0.533
```

A 가 접근에서 3.01 로 켜진다. 문항이 **"이 동작이 ~를 필요로 하는가"** 라고 일의
목적을 묻기 때문이다 -- 파지하러 가는 동안에도 참이다. `prompts/FORMAT.md` §4 의
"일의 이름이 아니라 순간을 묻는다" 를 어긴 자리다. libero v2->v3c 에서 같은 수정으로
국면몫이 0.489 -> 0.908 로 올랐던 전례가 있다.

## 예측

- **A 가 접근에서 3.01 -> 2.0 아래로 떨어진다.** 손가락이 아직 물체를 두르지 않았다.
- **A 가 파지에서 3.5 이상을 유지한다.** 거기서는 참이다.
- 그 결과 **접근 conf 가 파지 위로, 운반에 가깝게 올라간다.**
- 부호 고정 최적합 rho 가 v9 의 **+0.900 보다 오른다** (접근이 제자리를 찾으면 5국면
  중 가장 큰 오차가 사라진다).
- B·C·D·E·F 는 80% 이상 그대로. 단 B 가 A 를 되받는 문항("that closing spot")이라
  같이 움직일 수 있다.

## 실패로 볼 조건

- A 가 파지에서도 같이 죽으면(3.0 미만) 문항을 하나 잃는 것이다.
- 접근 conf 가 여전히 운반보다 낮으면 A 가 원인이 아니었던 것이다.
- rho 가 +0.900 아래로 떨어지면 되돌린다.

표본은 같은 380타일. 판정기·계산 사실 동일.
