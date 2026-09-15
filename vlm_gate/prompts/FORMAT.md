# 판정 프롬프트 규격 (모든 벤치마크 공통)

새 벤치마크(dexjoco 등)의 문항을 만들 때 이 규격을 따른다. 기준 구현은
`prompts/allex_v4c_FULL.txt` 이고, 그것을 만드는 코드는
`scripts/allex_litellm_run.py` 다.

---

## 1. 메시지 구조 -- **`role: user` 하나뿐이다**

```
[USER]
<이미지 N장>
{GUIDANCE}

{시야 설명 한 줄}

The robot was told: {에피소드 지시문}

{계산 사실}

{ASK}
```

**system 메시지를 쓰지 않는다.** 과제 틀은 GUIDANCE 가 담당한다.

왜: `vlm_gate.SYSTEM` 은 이진 게이트(YES/NO)용이고 마지막 줄이
"Answer with exactly one word: YES or NO." 다. 등급 문항과 같이 나가면 한 프롬프트가
서로 다른 답 형식을 요구한다. 그리고 그 SYSTEM 은 "reaching/carrying = YES,
closing the gripper = NO" 라는 **답 매핑을 미리 준다.**

그 한 줄이 실제로 라벨을 망친 것이 측정됐다. robocasa 에서 그 줄만 빼도 답의 35~40%
가 바뀌고, 물리 계산 천장 표와의 상관이 **-0.383 -> -0.173** 으로 역전이 절반 이하가
된다. 그 줄이 "정밀 조작이니 압축 불가" 라는 사람 직관을 주입했는데, 폐루프에서는
그것이 반대다 (`CoffeePressButton` 은 압축하면 유지율 1.22 로 오히려 오른다).

## 2. 등급 척도 -- **값 척도만 쓴다**

```
Answer each check on its own line as "A) 3", in order, nothing else -- one digit
from 1 to 5 per check, rating how far that check describes this moment:
  5 = clearly true of this moment
  4 = mostly true
  3 = partly true
  2 = barely true
  1 = not true at all
A grade refers only to the check on that line.
```

**거리·시점 척도를 쓰지 않는다.** robocasa·libero 가 쓰는
`4 = 아직 아니지만 1cm 나 한 동작 거리` 같은 척도는 **사건 문항에만 맞는다.**
"이 물건이 경첩에 붙어 있나" 같은 **속성 문항**에는 "1cm 앞" 이 대응되지 않는다.
그 자리에서 문항이 척도를 무시한 흔적이 남는다 -- robocasa C 는 SYSTEM 을 바꿔도
답이 94% 그대로였다(척도와 무관하게 답이 정해진다는 뜻).

## 3. 계산 사실

- **액션에서 계산한 사실만.** 추정·형용사가 아니라 측정값이다.
- **줄 수를 고정한다.** 조건부로 문장이 붙었다 빠지면 프롬프트 길이가 프레임마다
  달라지고, 배치 패딩이 `mm_token_type_ids` 를 어긋나게 해서 배치 8 에서 32개 중
  17개가 빈 답으로 돌아온 적이 있다.
- **한계 위반이 아니라 녹화 분포 대비로 말한다.** "이 값은 한계를 넘는다" 가 아니라
  "이 데이터셋 순간들의 70% 보다 빠르다". 데이터셋에 없는 값이 로봇의 한계를 뜻하지
  않는다.
- ASK 첫 줄에 `The measurements above are stated as fact -- do not re-estimate or
  repeat them.` 을 둔다.

## 4. 문항

- **한 문항은 한 가지만 묻는다.** 그래야 등급이 의미를 갖는다.
- **정도가 있는 것을 묻는다.** 범주를 등급 옷 입혀 묻지 않는다.
- **정답을 미리 말하지 않는다.** GUIDANCE 는 무엇이 왜 위험한지(물리)를 설명하고,
  어느 국면이 YES 인지 말하지 않는다.
- **일의 이름이 아니라 순간을 묻는다.** 문항이 태스크 정체성으로 답해지면 한 태스크
  안에서 프레임마다 갈리지 않는다. libero v2 의 B·D 가 그랬고, v3c 에서 접촉 국면을
  묻도록 바꾸자 국면몫이 0.489 -> 0.908 로 올랐다.
- 문항 수는 적어도 된다. **늘리면 기존 문항의 답이 바뀐다** -- allex 에 5번째 문항을
  넣었을 때 AUC 가 0.842 -> 0.718 로 떨어졌다.

### 문항을 어떻게 고르나 (방법론 -- 이걸 건너뛰지 말 것)

문항은 장면을 보고 지어내지 않는다. **측정된 피해에서 도출한다.**

1. 균등 압축 사다리로 태스크마다 피해를 잰다 (`analysis/eval_results/LADDERS.json`).
2. 피해가 큰 태스크를 RISK 풀, 견디는 태스크를 SAFE 풀로 가른다.
3. 각 풀의 **지시문에서** 공통 성질을 뽑는다 (장면이 아니라 지시문이다).
4. 그 풀의 태스크를 몇 개나 덮는지, 그 태스크들이 얼마나 피해를 받는지로 순위를 매긴다.
5. **가중은 덮는 태스크 수를 각 변의 합이 1 이 되게 정규화한 것**이다.
   재현: `scripts/derive_libero_questions.py`, 기록: `docs/PROMPT_METHOD.md`.

RISK 풀에서 나온 문항은 감점(SIGN −), SAFE 풀에서 나온 문항은 가점(SIGN +).

## 5. 부호 · 가중치 · conf

```
SIGN    감점(-1) 과 가점(+1)
WEIGHT  각 변(감점끼리, 가점끼리)의 합이 1.0
NGRADE  5

g(등급) = (등급 - 1) / (NGRADE - 1)
conf    = (1 + Σ_가점 w·g - Σ_감점 w·g) / 2      0~1 로 자른다
```

로짓에서 등급 분포를 받을 수 있으면 **기댓값 등급**(`Σ (i+1)·p(i)`)을 쓴다. 정수 등급만
쓰면 conf 가 계단이 되어 역치를 촘촘히 못 돌린다 -- Gemini 는 logprob 을 안 주므로
allex 는 43단 계단이다.

**가중은 라벨링이 끝난 뒤에 정한다.** 라벨에 등급을 그대로 저장해 두면 어떤 가중으로도
conf 를 다시 낼 수 있다.

## 6. 벤치마크 고유 내용

공통 글에 특정 벤치마크·임바디먼트의 것을 넣지 않는다.

```
넣지 않는다   "kitchen robot" · "table-top robot" · "HALF the control rate"
              "the gripper"(양손 로봇에), 손목 카메라(없는 데이터셋에)
넣는다        각자의 GUIDANCE 와 문항, 그리고 시야 설명 한 줄
```

시야 설명은 그 데이터셋이 실제로 보내는 카메라를 말한다.

```
allex     2장   "The two images are the left and right camera views of this one moment."
robocasa  3장   agentview-left · agentview-right · wrist
libero    3장   left · right · wrist
dexjoco   2장   wrist · ego|front     (label_chunks.py 의 views 설정)
```

## 7. 만들고 나서 반드시 할 것

```bash
python scripts/sync_prompts_folder.py              # prompts/ 를 원본에서 다시 쓴다
python scripts/verify_prompts_against_code.py      # 전문이 실제 코드와 같은지
python scripts/verify_prompts_against_code.py --selftest   # 검증기 자체를 검증
```

**전문을 손으로 적지 않는다.** 실제로 틀린 적이 있다 -- robocasa 를 이미지 6장으로
잘못 적어 라벨링에 나가지 않는 문구가 전문에 실렸고, 그것을 근거로 존재하지 않는
결함을 보고했다.

**조각이 아니라 조립된 전체를 모델이 받는 순서대로 읽는다.** GUIDANCE 만, ASK 만
읽으면 조각끼리의 모순을 못 본다. SYSTEM 의 YES/NO 모순이 그렇게 몇 달을 살아남았다.
