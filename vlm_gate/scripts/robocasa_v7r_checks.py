"""robocasa v7r — **문항은 v7 그대로, 부호·가중만 실측에 재적합한 판.**

라벨을 다시 만들지 않는다. 09-05 에 만든 204만 청크의 등급을 그대로 쓰고 conf 만
다시 계산한다. 등급이 비싸고 가중은 공짜다.

## 왜 다시 잡았나

v7 의 가중은 **태스크 덮개 수**로 정한 것이고 실측 결과에 맞춰 본 적이 없다
(A 4/6, B 2/6, C 4/15, D 5/15, E 6/15). allex 는 반대로 라벨링을 마친 뒤 실측 칸의
순서를 재현하도록 재적합했다. 그 차이가 역전의 상당 부분이었다.

1x 성공률 0.35 이상인 20태스크(문서 규칙), 청크 1,719,257개에서:

| 정답 | v7 가중 | v7r 가중(LOO 홀드아웃) |
|---|---|---|
| 사다리 | −0.693 | **+0.620** |
| 천장 hi | −0.641 | **+0.606** |
| K3 유지율 | −0.190 | **+0.466** |

1x 성공률을 뺀 순위 부분상관에서도 +0.528 / +0.412 / +0.437 로 남는다. 사다리로만
적합한 가중이 **적합에 쓰지 않은 천장 표**에서도 +0.376 이므로 외운 것이 아니다.

## 부호가 바뀐 문항이 셋이다

```
      v7            v7r
A     -1  0.667     -1  0.266
B     -1  0.333     +1  0.768     <- 뒤집힘
C     +1  0.267     -1  0.102     <- 뒤집힘
D     +1  0.333     +1  0.232
E     +1  0.400     -1  0.631     <- 뒤집힘
```

E 가 가장 크게 바뀌었다. `phase9_checks` 주석에 **"E 는 도출이 아니라 안전하다고
못 박았다(pinned, not derived)"** 고 적혀 있는데, 그 못 박음이 틀렸다. 폐루프에서
빈 공간 운반은 생각만큼 너그럽지 않다.

B 와 C 도 뒤집힌다. 사람 직관("파지 순간은 위험" · "기구가 잡아주면 안전")과 반대
방향인데, 이것은 [[vlm-gate-polarity-inverted]] 에서 이미 본 현상이다 -- 정밀 조작이
오히려 압축에 강하다.

## 주의

**이 가중은 이 등급판에만 맞는다.** 판정기나 문항이 바뀌면 다시 잡아야 한다(R5).
새로 라벨링하면 그 새 등급 위에서 다시 적합한다.

재현: `scripts/robocasa_refit_weights.py`
"""
import json
import os

NGRADE = 5

_P = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                  "..", "analysis", "robocasa_ladder", "refit_weights.json")
_R = json.load(open(_P))
SIGN = {k: int(v) for k, v in _R["sign"].items()}
WEIGHT = {k: float(v) for k, v in _R["weight"].items()}

# 문항 글은 v7 과 같다. 여기서 다시 적지 않는다 -- 두 곳에 적으면 어긋난다.
from phase9_checks_v7 import ASK, GUIDANCE, SCALE, _AXES  # noqa: E402,F401


def conf(grades):
    """등급 dict -> conf. 캐리어에 실리는 값이 이것이다."""
    return 0.5 * (1 + sum(SIGN[q] * WEIGHT[q] * ((grades[q] - 1) / (NGRADE - 1))
                          for q in SIGN))
