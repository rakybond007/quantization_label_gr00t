"""robocasa 문항 v22 후보 = **v10 + GUIDANCE 다시 씀 + C 좁힘.** 라벨 아직 없다.

## 왜 v10 을 고치나

v10 을 지킨 근거는 "네 지표 모두에서 제일 좋다" 였는데, 그 지표의 정답표가
`analysis/subaction/robocasa_limits.json` 이었다. 그 파일의 값 대부분이 **데이터로
식별되지 않는다.** 병목 모형(태스크 천장 = 지나는 국면 한계의 최솟값)에서 국면 값을
하나씩 고정해 놓고 나머지를 다시 적합하면 (`robocasa_limit_bands.json`):

    place_precise   1.25 ~ 1.25    식별
    place_open      1.25 ~ 1.667   식별
    pull_swing      1.5  ~ 1.83    식별
    ---
    carry           1.83 ~ 무한    미식별
    grasp_open      1.83 ~ 무한    미식별
    grasp_confined  1.667 ~ 무한   미식별
    place_catcher   1.83 ~ 무한    미식별
    approach        3.17 ~ 무한    미식별
    contact_coarse  3.17 ~ 무한    미식별
    contact_knob    3.17 ~ 무한    미식별

미식별 국면은 **어떤 태스크에서도 병목일 필요가 없어서** 하한 이상 어떤 값을 줘도
적합이 똑같다. json 의 `grasp_open` 2.00 과 RESULTS.md 의 4.00 은 같은 적합 품질이고
(RMSE 0.6583 로 동일), 4.00 은 적합기가 탐색 상단까지 밀어붙인 값일 뿐이다. `carry`
3.73 은 오히려 식별 구간(1.83~) 안에서 더 나쁜 쪽이다.

**실측이 말하는 것은 하나다: 놓기(정밀·넓은 면)와 문·서랍 당겨 열기가 빡빡하다.**
파지가 위험한지, 운반이 안전한지는 데이터에 없다.

## v21 에서 고친 것: 조립된 전문을 처음부터 끝까지 읽고 나온 것 전부

v18~v21 은 GUIDANCE 와 ASK 만 보고 돌렸다. 실제로 나가는 것은
`GUIDANCE + VIEW + 지시문 + 계산 사실 + ASK` 이고, VIEW 와 계산 사실은 한 번도 읽지
않았다. 읽고 나온 것을 한 판에 다 넣는다.

**GUIDANCE (셋 중 둘은 내가 v21 에서 새로 만든 것이다)**

* `this stretch of the motion` -> `the stretch of motion ahead of it`. 어느 구간인지
  안 썼다. 계산 사실 머리글이 `over the chunk **ahead**` 이고 allex 도
  `the **next** stretch of its motion` 이라고 명시한다. (v10 부터 있던 문제)
* `the same path, covered in fewer and longer steps` -> `reaching the same places in
  fewer and longer steps`. **거짓이었다** -- 웨이포인트를 줄이면 팔이 코너를 자르므로
  실제 경로는 같지 않다. 같은 것은 도달해야 하는 지점들이다. (내가 v21 에서 넣음)
* hemmed-in 증폭을 닫힘에만 건다. v21 은 "arrive somewhere exactly" 로 내려놓기와
  닫힘을 묶은 뒤 `higher still when the place it has to reach is hemmed in` 이라 했는데,
  **B 는 `that closing spot` 만 묻는다.** 좁은 곳에 내려놓기(디스펜서 아래 머그 =
  place_precise 1.25, 실측 최빡빡)를 받는 문항이 없었다. v10 의 2문단은 닫힘만 말해
  B 와 일치했고, 내가 v21 에서 깨뜨렸다.
* `a fitting that is anchored in place` -> 명사 삭제. fitting 은 수도꼭지·노브·스위치
  같은 부품을 뜻하고 **문과 서랍은 아니다.** 바로 뒤 예시가 `a door pushed shut` 이라
  명사가 예시를 못 덮는다. v10 은 명사 없이
  `something that is already held in place by its own hinge, rail or mount` 였다.

**VIEW -- 판본으로 옮기고 libero 문장을 뺐다**

v21 까지 VIEW 는 라벨러·프로브에 하드코딩돼 있었고 `libero_gemini_probe.py` 와 글자까지
같았다. 그래서 (1) robocasa 에 없는 `fine-insertion instant` 를 찾으라 하고, (2) 손목
뷰를 `only ... grasp-closure` 로 묶어 v21 이 위험의 중심으로 삼은 **내려놓는 순간(F)**
을 배제했다. (3) em dash 도 이 문장만 썼다. 셋 다 고쳤고, 판본에 넣어 옛 판 프로브가
그대로 재현되게 했다.

**계산 사실 -- `robocasa_v22_descriptors.py`**

`(mean step 0.41, peak 0.52)` 를 뺐다. 단위가 없어 0.41 이 무엇인지 알 수 없고 앞의
백분위 문구가 같은 뜻을 이미 전달한다. 겸사겸사 v8 의 머리글 replace 가 조용히
no-op 되던 위험을 assert 로 바꿨다.

**C -- 명사만 바꾸려다 동사를 앞으로 뺐다**

`a fitting` 문제가 C 에도 있다. 명사를 `something` 으로 바꾸면 뒤의 `and not
something` 과 겹쳐 더 나빠지므로 동사를 앞으로 뺀다("just pressing, turning or
pushing something that is anchored in place"). 가르는 절과 열거는 v21 그대로다.
**C 분리 AUC 0.927 / KNOB 0.885 가 v21 의 성과이므로, v22 프로브가 각각 0.89 / 0.85
미만이면 C 는 v21 문구로 되돌린다.**

**F**

`STILL LOWERING ... still coming down` 의 still 중복 제거.

## v20 에서 고친 것: C 를 **읽히게** 다시 씀

v20 은 가르는 축은 맞췄지만 문구가 추상 두 겹이었다 -- "going a little too far or a
little off line is **taken up by** the mechanism, and not **paid for by** losing hold".
정지 화면을 보는 모델이 확인할 수 있는 성질이 아니다. v19 의 실패도 같은 병이었다:
"rather than **gripping** something and having to **carry that grip along**" 에서 앞쪽
"쥔다" 만 읽히고 뒤쪽 "끌고 간다" 가 흘렀고, 그래서 꼭지 돌리기(쥐지만 제자리)까지
0.960 -> 0.700 으로 깎였다.

**통과해야 하는 셋을 이름으로 나열하고, 실패하는 하나만 접속으로 뺀다.**

    꼭지 돌리기   쥔다 · 안 움직인다     -> 통과
    밀어 닫기     안 쥔다 · 움직인다     -> 통과
    버튼 누르기   안 쥔다 · 안 움직인다  -> 통과
    쥐고 당겨 열기 쥔다 · 움직인다        -> 유일하게 둘 다  -> 탈락

처음에 조건을 "does not have to keep hold of **AND** travel along with" 로 썼는데,
이것은 `not(A and B)` 를 의도하지만 영어로는 `not A and not B` 로도 읽힌다 -- v19 가
정확히 그렇게 반쪽만 읽혀 실패했으므로 같은 위험을 다시 둘 수 없다. 그래서 탈락하는
경우를 **긍정문으로 이름 붙여** 못 박는다: "not something it has to keep hold of while
it travels, the way a handle is held on to and pulled to swing a door or drawer open".
경계 사례를 이름으로 대는 것이지 답을 지정하는 것이 아니다.

## v19 에서 고친 것: C 의 가르는 축

v19(=v18) 의 C 는 "쥐고 끌고 가는 것이 아니라" 로 pull_swing 을 갈랐는데,
**가르는 축을 잘못 잡았다.** `contact_knob` 은 어휘 정의가 "손잡이·꼭지를 **잡고**
돌림" 이고 실측으로 가장 관대한 쪽(>=3.17)인데, "쥔다" 를 벌점으로 쓰면 같이 깎인다.
380타일 프로브에서 그대로 나왔다:

    부류                        v10     v19
    PULL 쥐고 당겨 열기 1.5~1.83  0.768   0.162   <- 의도대로
    KNOB 잡고 돌림 >=3.17        0.960   0.700   <- 같이 깎였다
    PUSH 밀기·버튼 >=3.17        0.850   0.715   <- 같이 깎였다

v20 은 축을 **"빗나가면 기구가 흡수하나, 쥔 것을 놓치나"** 로 바꾼다. 꼭지를 돌릴
때는 쥐고 있어도 마운트가 여유를 먹고 꼭지를 놓치지 않는다. 문을 쥐고 당겨 열 때는
경로가 호에서 벗어나는 순간 손잡이가 손에서 뜯긴다. 밀어 닫을 때는 애초에 쥔 것이
없다. 셋이 이 축에서 제대로 갈린다.

## v18 에서 고친 것: F 하나

v18 의 F 는 v10 의 구문 모호성("LOWERING what it still holds **onto** its target")
을 풀려고 동사를 `SETTING DOWN` 으로 바꿨는데, **범위가 넓어졌다.** 380타일 프로브
(`analysis/ratio_prompt/rc_v18.jsonl`)에서 **이미 놓고 물러나는 타일의 F 등급이
+0.72 오르고 같은 타일의 D 가 -0.39 떨어졌다** -- 모델이 답을 D 에서 F 로 옮겼다.
`SETTING DOWN` 이 내려놓기 완료까지 포함해 읽히고, 뒤에 붙인 `not yet let go` 가
막지 못했다. 그 결과 "이미 놓은 뒤"(안전)가 "쥔 채 내려놓는 중"(위험)보다 낮아져
순서가 뒤집혔고 검사1 AUC 가 0.597 -> 0.552 로 내려갔다.

v19 는 `LOWERING` 을 되살리고 `holds onto` 를 문장에서 빼서 모호성만 없앤다. 그리고
**D 의 "taking its own weight" 의 정확한 여집합**("the object not yet taking its own
weight")을 F 에 넣어 두 문항을 관측 가능한 성질로 배타적으로 만든다.

## v10 에서 바꾼 것 셋

1. **GUIDANCE 1문단.** v10 은 "thinned out -- how many of its commanded poses could be
   dropped" 라고 물었다. 우리가 정하는 것은 몇 개를 버리는지가 아니라 **이 구간을 빨리
   지나가도 되는지**다. 배속도 우리가 정하지 않는다.

2. **GUIDANCE 2문단이 F 를 짚는다.** v10 은 위험을 "그리퍼가 한 지점에 닫히는 곳"
   (A) 과 "그 자리가 좁을 때"(B) 로만 걸었다. 그런데 실측으로 값이 묶인 위험은
   place_precise 1.25 / place_open 1.25~1.667 = **F(쥔 채 내려놓는 중)** 이고, A/B 가
   가리키는 grasp_* 는 하한만 있다. 유일하게 뒷받침되는 위험을 가이던스가 한 번도
   가리키지 않았다. 도착 정확성으로 묶어 **놓기를 먼저, 파지를 뒤에** 쓴다.

3. **C 를 좁힌다.** v10 C 는 "a door or drawer on its hinge or rail" 이라 두 국면을
   한 문항에 넣었다 -- 밀어 닫기/버튼/꼭지(contact_coarse·knob >= 3.17, 가장 관대)와
   **손잡이를 쥐고 당겨 열기(pull_swing 1.5~1.83, 식별된 빡빡한 셋 중 하나)**. 둘 다
   "경첩 위의 문" 이어서 문구로 갈리지 않고, `Open*` 3태스크가 통째로 최대 가점
   (+0.50) 쪽에 들어갔다. **쥐고 당겨 여는 경우를 문항에서 명시적으로 뺀다.**

## R2 를 왜 어기나

R2(한 판에 한 군데)는 **판을 실측으로 비교할 수 있을 때** 지키는 규율이다. 지금
정답표가 무너져서 비교할 지표 자체가 없다. 그래서 기준은 "측정값이 좋은가" 가 아니라
**"문구가 사실인가"** 하나이고, 사실이 아닌 세 자리를 같이 고친다. 실측 비교는 새
정답표(식별된 3국면만 쓰는 판)가 생긴 뒤에 다시 한다.

## 가중

**여전히 잠정이다.** v9 에서 물려받은 값이고 실측에 맞춘 적이 없다. 특히 C 의 0.50 은
위 3번 결함 위에서 정해진 값이라 다시 잡아야 한다. 라벨링 뒤에 적합한다(R5).

A/B 의 위험 부호는 건드리지 않았다 -- 하한만으로는 뒷받침도 반박도 안 되므로
실측 적합(R5)이 정할 자리다.
"""



NGRADE = 5

# 가중은 실측된 분리 폭에서 읽었다. 두 쪽 합은 v1 처럼 각각 1 이다.
#
# A = 0.6 -- "닫아야 하는가" 는 단계 0(1.032) 과 1~3(평균 0.79) 을 가른다. 폭 0.24.
# B = 0.4 -- "그 자리가 좁은가" 는 단계 1(0.969) 과 2~3(평균 0.75) 을 가른다. 폭 0.22
#            로 A 에 거의 맞먹고, K3 에서는 오히려 더 크다(0.828 대 0.502 = 0.33 對
#            A 의 0.32). v1 에서 B 를 0.333 으로 두고도 아무 일이 없었던 것은 무게가
#            아니라 겨눈 곳이 틀렸기 때문이므로, 다시 겨눈 B 에는 무게를 준다.
#
# C = 0.50 -- 기구가 경로를 정한다는 것은 실측된 안정 쪽 논거이고, 다섯 문항 중 모델이
#            유일하게 확신하는 문항이다(뽑힌 등급 확률 0.773, 엔트로피 0.407,
#            0.5 미만 10%). 가점 쪽 최대.
# D = 0.15 -- "넓은 면에 내려놓는다" 에 가점을 주는데, 그 넓은 면은 조리대이고
#            *ToCounter 가 오히려 약한 쪽이다(0.766). D 가 가장 잘 맞는 태스크들에서
#            증거가 반대를 가리키므로 최소로 둔다. 문구는 그대로 두었다 -- 지우는
#            것보다 무게를 내리는 쪽이 변경이 작다.
# E = 0.35 -- "열린 공간을 건너 옮긴다" 는 PnP 두 부류에 모두 걸려 중립 증거이고,
#            모델도 잘 못 답한다(확률 0.407). 가운데.
SIGN = {"A": -1, "B": -1, "C": +1, "D": +1, "E": +1, "F": -1}

# 문항 이름. **문구·부호·가중은 한 글자도 안 바뀐다** -- 부록 생성기와 진단표가
# 읽을 이름만 붙인 것이고, 라벨에 영향을 주지 않는다(libero_v3c_checks 와 같은 모양).
NAME = {"A": "CLOSE_EXACT", "B": "HEMMED_IN", "C": "GUIDED_IN_PLACE",
        "D": "LETTING_GO", "E": "CARRY_OPEN", "F": "SETTING_DOWN"}
# F 가중은 **잠정**이다 -- 위험 쪽 셋을 균등으로 두고, 라벨링 뒤에 실측에
# 맞춰 다시 잡는다(R5). 지금 값으로 성능을 판단하면 안 된다.
WEIGHT = {"A": 0.40, "B": 0.27, "F": 0.33,
          "C": 0.50, "D": 0.15, "E": 0.35}

# 기간을 글로 못 박지 않는다. 창은 16스텝인데 robocasa 는 20fps 라 0.80초,
# libero 는 10fps 라 1.60초다. "one second" 라고 쓰면 robocasa 에서 25% 과장이고
# 같은 문구를 libero 에 옮기면 배로 틀린다. allex 도 기간 대신 "the next stretch
# of its motion" 이라고만 한다.
GUIDANCE = (
    "You are judging one moment of a kitchen robot's motion, to decide whether the arm "
    "could get through the stretch of motion ahead of it faster -- reaching the same "
    "places in fewer and longer steps -- without changing the outcome.\n\n"
    "Going faster leaves the arm fewer chances to correct itself along the way. That "
    "costs most where the arm still has to arrive somewhere exactly: setting down what it "
    "is carrying onto the spot it must end up on, or closing the gripper on one exact "
    "point -- and, when it is the gripper closing, more still if that spot is hemmed in. "
    "It costs least where the arm only acts on something that is anchored in place -- a "
    "button or switch pressed, a knob or tap turned, a door pushed shut. There a longer "
    "step just presses or turns a little harder and nothing is lost.\n\n"
    "Judge the moment in front of you, not the task as a whole. One task passes through "
    "both kinds from one moment to the next."
)

# **VIEW 를 판본에 넣는다.** v21 까지는 라벨러·프로브 스크립트에 하드코딩돼 있었고,
# 그래서 libero 의 문장이 그대로 넘어와 있었다 -- robocasa 에는 삽입 태스크가 없는데
# "fine-insertion instant" 를 찾으라고 했고, 손목 뷰를 "파지 닫힘에만" 쓰라고 못 박아
# v21 이 위험의 중심으로 삼은 **내려놓는 순간**(F)을 배제했다. em dash 도 이 문장만
# 썼다. 판본에 두면 옛 판의 프로브 결과가 그대로 재현된다.
VIEW = (
    "You are shown 3 camera views: agentview-left, agentview-right, and a wrist "
    "(eye-in-hand) close-up. The wrist camera is mounted on the gripper, so objects "
    "normally look close in it -- general closeness is normal. Use the wrist view to "
    "judge the close moments: the gripper actually closing on something, and something "
    "being set down onto the spot it must end up on."
)

# 척도는 **allex(Gemini) 와 같은 값 척도**다(`prompts/FORMAT.md` §2).
SCALE = (
    "  5 = clearly true of this moment\n"
    "  4 = mostly true\n"
    "  3 = partly true\n"
    "  2 = barely true\n"
    "  1 = not true at all\n"
)

# B: v10 의 "or lined up with a slot **or a handle**" 에서 손잡이를 뺐다. 손잡이는 좁은
#    자리가 아니고, Open* 의 손잡이 파지는 실측 분해에서 grasp_open("열린 곳에 놓인
#    것을 집음")이다. 그대로 두면 B 가 가장 트인 파지를 confined 로 몬다.
#
# C: v10 은 "a door or drawer on its hinge or rail" 이라 **쥐고 당겨 여는 것**
#    (pull_swing 1.5~1.83, 식별된 빡빡한 셋 중 하나)과 **밀어 닫기·버튼·꼭지**
#    (contact_coarse/knob >= 3.17, 가장 관대)를 한 문항에 넣었다. 가르는 성질은
#    "쥔 손을 데리고 움직여야 하는가" 다 -- 꼭지를 돌리거나 문을 밀어 닫을 때는
#    손이 제자리에서 일하고, 당겨 열 때는 손잡이를 쥔 채 호를 따라 끌려간다.
#    **답을 지정하지 않는다.** v18 초안에 "Answer 1 if ..." 를 넣었더니 본절이 참인
#    상황에서 지시문이 답을 꺾는 자기모순 문항이 됐다.
#
# F: v10 은 "LOWERING what it still holds onto its target" 로 holds onto 가 붙어 읽혀
#    구문이 두 갈래였고 "the spot it must end up" 에 on 이 빠져 있었다. v18 은 동사를
#    SETTING DOWN 으로 바꿔 그걸 풀었으나 **범위가 넓어져** 이미 놓은 타일까지 먹었다
#    (프로브에서 F +0.72 / D -0.39). v19 는 LOWERING 을 되살려 하강 중으로 못 박고,
#    D 의 "taking its own weight" 의 여집합을 명시해 D 와 배타적으로 만든다.
_AXES = (
    "A) Is the gripper CLOSING on its target right now, or about to -- the fingers\n"
    "   already around the handle, rim or object and only the squeeze left, so that a\n"
    "   slip here loses it?\n"
    "B) Is that closing spot hemmed in -- reached into a cabinet, a microwave, an oven or\n"
    "   a sink, or lined up with a slot -- so the gripper has little room around it as\n"
    "   it closes?\n"
    "C) Is the arm just pressing, turning or pushing something that is anchored in\n"
    "   place -- a button or switch, a knob or tap, a door or drawer being shoved shut\n"
    "   -- and not something it has to keep hold of while it travels, the way a handle\n"
    "   is held on to and pulled to swing a door or drawer open?\n"
    "D) Is the object ALREADY RESTING where it is meant to end up, taking its own\n"
    "   weight, with nothing left but to open the hand and draw back?\n"
    "E) Is the gripper carrying something through open space -- already holding it,\n"
    "   still well away from wherever it is going?\n"
    "F) Is the arm STILL LOWERING the object onto its target right now -- the gripper\n"
    "   not yet opened, the object not yet taking its own weight, coming down onto\n"
    "   the spot it must end up on?\n")

ASK = ("The measurements above are stated as fact -- do not re-estimate or repeat "
       "them. Answer each check from what the cameras show about the MOMENT in "
       "front of you, read together with those measurements.\n"
       "Answer each check on its own line as \"A) 3\", in order, nothing else "
       "-- one digit from 1 to 5 per check, rating how far that check describes this "
       "moment:\n" + SCALE + "A grade refers only to the check on that line.\n"
       + _AXES + "Answer:")
