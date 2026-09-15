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

- **액션에서 계산한 사실만.** 화면을 보고 짐작한 것이 아니라 계획된 액션에서 나온
  값이다. 다만 판정기에게 건네는 형태는 아래 `문장으로 바꿀 때` 를 따른다 --
  상태는 말로, 압축 요구만 숫자로.
- **줄 수를 고정한다.** 조건부로 문장이 붙었다 빠지면 프롬프트 길이가 프레임마다
  달라지고, 배치 패딩이 `mm_token_type_ids` 를 어긋나게 해서 배치 8 에서 32개 중
  17개가 빈 답으로 돌아온 적이 있다.
- **한계 위반이 아니라 녹화 분포 대비로 말한다.** "이 값은 한계를 넘는다" 가 아니라
  "이 녹화본 대부분보다 크다". 데이터셋에 없는 값이 로봇의 한계를 뜻하지 않고, 시연
  범위를 넘는 것 자체가 감점 사유가 아니다.
- ASK 첫 줄에 `The measurements above are stated as fact -- do not re-estimate or
  repeat them.` 을 둔다.

### 무엇을 계산하나 (네 벤치마크가 쓰는 뼈대)

계산 사실이 있는 이유: **압축 요구는 픽셀에 안 보인다.** 두 프레임은 궤적이 어디로
갈 계획인지 말해주지 않는데, K 배로 합칠 때 컨트롤러가 받을 점프는 계획된 액션에만
있다. 속도·방향도 정지 프레임에서 "운반 중" 과 "내려놓는 중" 을 못 가른다.

**단, 계산 사실 칸에 지시문을 넣지 말 것.** allex 에서 넣어봤더니 모델이 화면 대신
지시문을 보고 답해서 한 태스크 안 청크들이 한 값으로 뭉쳤다 (Rotate Box 77청크가
같은 값, 칸 내 분산 0.112 -> 0.049, rho +0.525 -> +0.491). 그래서 `facts_v3(x)` 는
`task` 를 넘기지 않는다 -- 버그가 아니라 측정으로 정한 것이다. 지시문은 `ASK` 앞의
`The robot was told:` 줄로만 한 번 들어간다.

구현: `robocasa_descriptors.py` · `allex_v2_common.py` · `libero_descriptors.py` ·
`dexjoco_descriptors.py` 의 `descriptors()`. 모두 **계획된 액션**(실행 결과가 아니라
컨트롤러에 나갈 값)에서 청크 창 하나를 보고 낸다.

| 무리 | 무엇 | 어디서 쓰나 |
|---|---|---|
| 잡음/놓음 | `grip_change` `grip_close` `grip_open` `grip_at` `gripper_closed` | robocasa · libero |
| (다지손) | `hand_change` `hand_active_frac` `hand_speed_mean/max` `hand_trend` | dexjoco · allex |
| 속도 | `speed_mean` `speed_max` `speed_pct`(분포 내 분위) `rot_speed_mean/max` | 전부 |
| 방향 | `reversal`(방향 급반전) `decel`(멈춤으로 감속) `up`/`down`/`turn` | 전부 |
| 하중 | `closed_slow`(무언가 든 채 기어감) `held` | 전부 |
| **압축 요구** | `merge_demand_kK` / `merged_pos_max` `merged_rot_max` `merged_hand_max` / `skip_excess` | 전부 |

**압축 요구가 핵심 양이다.** 배율 K 로 합칠 때 컨트롤러가 **한 제어 틱에** 요구받는
변위다. 청크마다 다르고 계획된 액션에서 바로 나온다 -- 국면 구성으로 역산하는 길은
막혀 있다(태스크마다 국면 구성이 거의 같아 설계행렬 조건수 1158).

**식은 액션 공간이 정한다. 재는 물리량은 같다.**

| 액션 공간 | 식 | 구현 |
|---|---|---|
| 절대 타깃 | `max ‖a[i+K] − a[i]‖` (블록의 마지막만 남음) | allex `md(K)`, dexjoco `_merged_*` |
| 상대/델타 | `Σ` K 개 델타 | robocasa `w[0:-1:2] + w[1::2]` |

allex·dexjoco 는 절대 관절 타깃이라 블록-라스트가 곧 병합이다(로봇은 같은 자세에
도달하되 K 틱이 아니라 1 틱에). robocasa·libero 는 델타라 더한다. dexjoco 가 이름을
`skip_excess` 라 붙였지만 재는 것은 **병합 점프**다 -- 이름에 속지 말 것.
`dexjoco_descriptors.py` 머리주석이 그 이유를 적어 두었다.

함정 둘:
- **회전은 측지거리로.** rotvec 을 그냥 빼면 안 된다. `(r[:-1].inv() * r[1:]).as_rotvec()`
  (`_rotvec_step_angles`). dexjoco 는 손목 6D 회전이 있어 반드시 걸린다.
- **다지손엔 이진 그리퍼 차원이 없다.** robocasa 의 `grip_change` 류가 성립하지 않는다.
  16-DoF 손은 연속이라 파지/해제를 **손관절 운동량**으로 잡는다
  (`hand_change` `hand_active_frac` `hand_trend`).

**한계 위반으로 쓰지 말 것.** v2 에서 "시연에 없던 크기" 라고 적었다가 **뺐다.**
우리 목적은 느린 시연보다 빨리 움직이는 것이고, 시연 범위를 넘는 것 자체는 감점
사유가 아니다 -- 시뮬에서 clip 을 푸는 것과 같은 이치다. `MERGE_LIMIT_V2`(0.385 rad)
· `CLIP`(1.0) · `JUMP_LIMIT_*` 는 **디스크립터 내부 신호**로만 쓰고, 판정기에게는
좋다/나쁘다 대신 **다른 순간들에 비해 큰가 작은가**만 말한다.

### 문장으로 바꿀 때 (`allex_facts.py` 가 기준이다)

**상태는 말로, 압축 요구만 숫자와 함께.**

```
상태      문턱값으로 3단 나눠 말로만 준다. 날숫자를 안 준다.
          "the arms are moving fast" / "the fingers are shifting a little"
          "the palms are close in, and drawing together"
          -- 날숫자를 넘기면 해석이 판정기 몫이 된다. 문턱값이 이미 분포에서
             나온 것이라 말 자체가 분포를 담고 있다.

압축 요구  숫자 + 분위대. 이것만 숫자를 준다.
          "at 2x this stretch would move 0.128 rad in one step, larger than most"
          "at 3x 0.171 rad, much larger than most moments in this recording"
```

분위대 5단(`much smaller than most` / `smaller than most` / `about average` /
`larger than most` / `much larger than most moments in this recording`)의 경계는
**그 데이터셋 전체에서 뽑는다.** allex 는 9,179 청크에서 k2 p20 .059 · p40 .084 ·
p60 .116 · p80 .173, k3 는 .079/.114/.156/.234. 새 벤치마크는 자기 분포로 다시 낸다.

**사실 문장을 늘리면 모든 문항의 답이 바뀐다.** allex 에서 돌림/유지를 가르려고
`speed_ratio`·`rot_asym` 두 문장을 넣어봤다. 신호는 있었는데(순간 정답지 84청크에서
AUC 0.320·0.343) 판정기가 쓰지 못했다:

| 판정기 | A 물고vs접근 | A 돌림vs유지 | 답이 바뀐 비율 |
|---|---|---|---|
| sonnet | 0.774 -> 0.762 | 0.500 -> 0.500 | A 1% B 1% C 24% D 5% |
| gemini | 0.834 -> **0.691** | 0.575 -> 0.527 | A 46% B 36% C 51% D 46% |

노린 것은 안 오르고 본래 표적이 떨어졌다. **사실 두 문장이 네 문항을 다 흔든다** --
R2 간섭은 문항끼리만의 문제가 아니다. 그래서 두 문장은 뺐다.

**신호가 있는데 판정기가 못 쓰면 문장으로 주지 말고 계산으로 직접 쓴다** (conf 에
곱하거나 배속을 직접 조정). 문항을 늘리는 것도 같은 이유로 권하지 않는다 -- 화면에서
못 읽는 것을 물으면 답이 안 나온다.

숫자는 고정 폭(`%.3f`)으로, 문장 수는 조건과 무관하게 항상 같게.

### 기간을 글로 못 박지 말 것

창은 16스텝으로 같아도 초로 환산하면 데이터셋마다 다르다.

```
robocasa  20 fps  ->  0.80 초
allex     30 fps  ->  0.53 초
libero    10 fps  ->  1.60 초
dexjoco   30 fps  ->  0.53 초
```

robocasa 전문은 "You are judging one **second** of a kitchen robot's motion" 이라고
적고 있었다 -- 실제의 1.25배다. 같은 문구를 libero 로 옮기면 1.6초를 1초라 부르게
된다. allex 처럼 **"the next stretch of its motion"**, libero 처럼 **"one moment"**,
계산 사실 머리글은 **"over the chunk ahead"** 로 쓴다.

**알려진 예외:** `allex_v4c_FULL.txt` 의 마지막 GUIDANCE 줄에 "from one second to the
next" 가 남아 있다(실제 0.53초). 고치면 48,086 청크 라벨이 무효가 되고, 그 줄은
측정값 주장이 아니라 "한 구간 안에서 국면이 계속 바뀐다" 는 취지라서 그대로 둔다.
새로 만드는 프롬프트는 따라 하지 말 것.

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
