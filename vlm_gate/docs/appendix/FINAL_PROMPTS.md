# 부록: 최종 VLM 프롬프트 전문

`scripts/dump_final_prompts.py` 가 문항 모듈에서 직접 읽어 생성한다. 손으로 옮겨 적지 않는다.

판정기는 `nvidia/Cosmos3-Nano` 이고 그리디 디코딩으로 텍스트를 답한다. 등급 토큰 자리의 로짓은 그 답이 얼마나 확실했는지를 덧붙이는 데만 쓰며, 강제 슬롯 argmax 로 답을 짓지 않는다.

**문구가 어디 있는지가 벤치마크마다 다르다.** 한 번 틀리면 이 부록이 엉뚱한
프롬프트를 최종이라고 가리키게 되므로(실제로 libero 에서 그랬다) 먼저 적는다.

| 벤치마크 | 부호·가중치 | 문항 본문 |
|---|---|---|
| robocasa | `scripts/phase9_checks_v7.py` | 같은 모듈의 `GUIDANCE` / `ASK` 상수 |
| libero | `scripts/libero_v3c_checks.py` | **`analysis/_evolver/_libero/libero_{guidance,questions}_v3c.txt`** |
| allex | `scripts/allex_v4c_checks.py` | 같은 모듈의 `GUIDANCE` / `ASK` 상수 |

libero 의 `libero_v2_checks.py` ~ `libero_v3c_checks.py` 는 바이트 단위로 같다.
그 모듈이 부호·가중치만 담기 때문이고, **문항이 안 바뀌었다는 뜻이 아니다.**
`prompts/libero_v2.txt` 는 v2 판이며 최신이 아니다.

**이 부록은 GUIDANCE·QUESTION·계산 사실만 담는다. SYSTEM 과 view_note 는 빠져 있다.**
둘 다 실제로 프롬프트에 들어가며, SYSTEM 은 과제 자체를 규정하고(YES/NO 압축 가능성)
view_note 는 이미지 배치를 설명한다. 조립된 전문은 `prompts/<벤치>_<판>_FULL.txt` 에
있고 `scripts/sync_prompts_folder.py` 가 생성한다 -- **프롬프트를 통째로 인용해야 할 때는
그 파일을 쓸 것.** 조립 순서:

```
[SYSTEM]  SYSTEM 상수 + "Additional learned guidance (...):" + GUIDANCE
[USER]    이미지 N장 + "Task: " + 지시문 + 계산 사실 + view_note + QUESTION
```

프롬프트는 세 덩이로 조립된다 -- **계산 사실**(지시문 + 액션에서 계산한 사실 문장), **GUIDANCE**(무엇을 판단하는 일인지), **QUESTION**(등급 척도와 문항). 계산 사실은 프롬프트의 일부다. 빼고 물으면 다른 질문이 된다.

---

## robocasa

24 태스크 · 압축 여부 게이트. v1→v7, 네 판.

- 모듈 `scripts/phase9_checks_v7.py`
- 계산 사실 `scripts/robocasa_descriptors.py: facts_text`
  (`phase9_two_sided.py:132` 이 `ins = f"{instr}\n{facts_text(x)}"` 로 조립한다)
- 등급 수 5 · 문항 5개
- 감점 AB (가중 합 1.00) · 가점 CDE (가중 합 1.00)

| 문항 | 이름 | 부호 | 가중 |
|---|---|---|---|
| A | CLOSE_EXACT | 감점 | 0.600 |
| B | HEMMED_IN | 감점 | 0.400 |
| C | FIXTURE_HELD | 가점 | 0.500 |
| D | LETTING_GO | 가점 | 0.150 |
| E | CARRY_OPEN | 가점 | 0.350 |

### 계산 사실 (실제로 나간 형태, 두 청크 예시)

libero 절은 프롬프트가 조립된 텍스트 파일 하나라 예시가 파일 안에 이미 있다.
robocasa 는 문구가 모듈 상수에 있고 사실은 함수가 청크마다 만들기 때문에, 같은
것을 볼 수 있도록 여기에 실제 출력을 붙인다. **아래 QUESTION 첫 줄이 "The
measurements above are stated as fact" 인 것이 이 블록이 실재했다는 증거다** --
사실 블록 없이 그 문장만 나가면 가리킬 대상이 없다.

```
MEASURED FROM THE PLANNED MOTION over the next ~1 second (these are computed facts, not estimates): the gripper stays open throughout; the end-effector keeps a consistent direction; it is among the slowest motions in this dataset -- faster than 3% of all moments here (mean step 0.17, peak 0.50); it is not creeping along with something held; it is not decelerating to a stop.

MEASURED FROM THE PLANNED MOTION over the next ~1 second (these are computed facts, not estimates): the gripper stays open throughout; the end-effector keeps a consistent direction; it is on the fast side for this dataset -- faster than 70% of all moments here (mean step 0.74, peak 0.95); it is not creeping along with something held; it is not decelerating to a stop.
```

다섯 문장이 조건과 무관하게 **항상** 나온다. 조건부로 줄이 붙었다 빠지면 프롬프트
길이가 프레임마다 달라지고, 배치 패딩이 `mm_token_type_ids` 를 어긋나게 해서
배치 8 에서 32개 중 17개가 빈 답으로 돌아왔다 (`robocasa_descriptors.facts_text`
주석에 기록됨).

### GUIDANCE  (sha1 `c3ba33024565`)

```
You are judging one second of a kitchen robot's motion, to decide how much of it could be thinned out -- how many of its commanded poses could be dropped, letting the arm travel further between the ones that remain, without changing the outcome.

What thinning takes away is the arm's chance to correct itself on the way. That matters most where the gripper has to close on one exact spot, and it matters most of all when that spot is hemmed in. It matters least where the arm only pushes or turns something that is already held in place by its own hinge, rail or mount.

Judge the moment in front of you, not the task as a whole. One task passes through both kinds from one second to the next.
```

### QUESTION  (sha1 `9b446e769f82`)

```
The measurements above are stated as fact -- do not re-estimate or repeat them. Answer each check from what the cameras show about the MOMENT in front of you, read together with those measurements.
Answer each check on its own line as "A) 3", in order, nothing else -- one digit from 1 to 5 per check, rating how far that check describes this moment:
  5 = it is happening right now -- the picture shows the contact, the grip or
      the position the check describes
  4 = not yet, but the hand is right up against it, a centimetre or a single
      motion away
  3 = the hand is heading for it and still some way off
  2 = the thing the check is about is there in the picture, but the arm is busy
      with something else
  1 = there is nothing in this picture the check could be about
A grade refers only to the check on that line.
A) Does this motion need the gripper to CLOSE on one exact spot -- a handle, a rim,
   the body of an object it must pick up -- rather than only pushing or turning
   something that stays where it is?
B) Is that closing spot hemmed in -- reached into a cabinet, a microwave, an oven or
   a sink, or lined up with a slot or a handle -- so the gripper has little room
   around it as it closes?
C) Is the thing being acted on held in place by the furniture or appliance itself
   -- a door or drawer on its hinge or rail, a knob, switch, button or lever on its
   mount -- so the mechanism, not the arm, decides where it ends up?
D) Is the hand letting go, or about to -- the object already resting where it is
   meant to end up, so that what remains is only to open and withdraw?
E) Is the gripper carrying something through open space -- already holding it,
   still well away from wherever it is going?
Answer:
```

### conf 계산

```
g(등급) = (기댓값등급 - 1) / (NGRADE - 1)
conf    = (1 + Σ_가점 w·g - Σ_감점 w·g) / 2
```
기댓값 등급은 등급 토큰 자리의 softmax 분포에서 Σ (i+1)·p(i) 로 낸다.
정수 등급으로 계산하면 conf 가 계단이 되어 역치가 안 먹는다(하네스 R4).

---

## libero

40 태스크 · 압축 여부 게이트. v1→v3c.

**문구가 checks 모듈에 없다.** 모듈(`scripts/libero_v3c_checks.py`)은 부호·가중치만
담고, 문항 본문은 아래 두 텍스트 파일에 있다. `libero_v2_checks.py` ~
`libero_v3c_checks.py` 가 바이트 단위로 같은 것은 그 때문이며, **문항이 안 바뀌었다는
뜻이 아니다.**

- 모듈(부호·가중치) `scripts/libero_v3c_checks.py`
- GUIDANCE `analysis/_evolver/_libero/libero_guidance_v3c.txt` (sha1 `053844846ac9`)
- QUESTION `analysis/_evolver/_libero/libero_questions_v3c.txt` (sha1 `697bf64ffaca`)
- 계산 사실 `scripts/libero_checks.py: facts`
- 등급 수 5 · 문항 5개
- 감점 AB (가중 합 1.00) · 가점 CDE (가중 합 1.00)

`prompts/libero_v2.txt` 는 **v2 판이고 최신이 아니다.** 이 부록이 한때 그 파일을
최종으로 적었다 -- 그것이 오해의 원인이었다.

| 문항 | 이름 | 부호 | 가중 |
|---|---|---|---|
| A | SMALL_TARGET | 감점 | 0.636 |
| B | CLOSING_GRIP | 감점 | 0.364 |
| C | CATCHER | 가점 | 0.533 |
| D | LETTING_GO | 가점 | 0.267 |
| E | NO_HOLD | 가점 | 0.200 |

이름이 모듈의 `NAME` 과 어긋난다 -- 모듈은 v2 이름(`CROWDED_PICK`, `WIDE_SURFACE`)을
그대로 들고 있다. 부호·가중치가 자리(A~E)로 붙으므로 계산은 맞지만, 이름만 보고
문항을 짐작하면 틀린다.

### v2 에서 무엇이 바뀌었나 -- B 와 D 를 재조준했다

v2 의 B·D 는 **태스크 속성**을 물어서 한 태스크 안에서 프레임마다 갈리지 않았다.
v3a 가 D 를, v3b 가 B 를 각각 그 순간의 **접촉 국면**으로 돌렸고(한 판에 한 문항, R2),
v3c 가 둘을 합쳤다.

```
B) v2   집을 것이 얹혀 있거나 사이에 끼어 있는가        (CROWDED_PICK, 태스크 속성)
   v3c  지금 그리퍼가 닫히고 있거나 곧 닫는가 -- 손가락이 이미 둘러싸고 조이는 것만
        남아서, 여기서 미끄러지면 놓치는 상황인가       (CLOSING_GRIP, 국면)

D) v2   물체보다 훨씬 큰 면에 놓는가                  (WIDE_SURFACE, 태스크 속성)
   v3c  손이 놓고 있거나 곧 놓는가 -- 물체가 이미 있어야 할 자리에 놓여서 남은 건
        열고 물러나는 것뿐인가                       (LETTING_GO, 국면)
```

측정된 효과 (`analysis/libero_retention/V2_DIAGNOSIS.md`, `score_v3a.txt`, `score_v3b.txt`):

| 판 | 국면몫 | 태스크몫 | 국면 conf 폭 | 1.80x 유지율 상관 |
|---|---|---|---|---|
| v2 | 0.489 | 0.949 | 0.05 | +0.454 |
| v3a (D 재조준) | 0.768 | 0.903 | 0.104 | +0.448 |
| v3b (B 재조준) | **0.908** | 0.617 | **0.222** | +0.374 |

```
D (v3a)  접근 1.33 · 파지 1.55 · 해제 4.61 · 운반 1.33
B (v3b)  접근 1.13 · 파지 4.56 · 해제 1.49 · 운반 1.69
```

국면몫이 0.489 -> 0.908 로 올라간 대가로 태스크몫이 0.949 -> 0.617 로, 유지율 상관이
+0.454 -> +0.374 로 내려갔다. **프레임별로 갈리게 만든 것과 태스크를 맞히는 것은 서로
당긴다.**

### GUIDANCE  (sha1 `053844846ac9`)

```
You are judging one moment of a table-top robot's motion, to decide how much of it could be thinned out -- how many of its commanded poses could be dropped, letting the arm travel further between the ones that remain, without changing the outcome.

Two things decide that, and both are about where the object has to end up or where it has to come from, not about how fast the arm happens to be going.

A destination can be forgiving or unforgiving. Dropping something into a basket or a bowl is forgiving: the container catches it, and landing a centimetre off changes nothing. Setting something down on a plate, on a burner, or into a rack is not: it has to come to rest squarely on a small support, and the last approach is where a longer step puts it down askew.

An approach can be open or threaded. Reaching for something that stands clear on the table is open. Reaching for something perched on top of another object, or wedged between two things standing close by, leaves one line in: a longer step is a wider swing, and what should have been cleared gets struck.

Most of the time neither is true -- the gripper is crossing open space, or is already holding something with room around it, and a longer step changes nothing. That is the common case.

Judge the moment in front of you, not the task as a whole. One task passes through both kinds from one moment to the next: the same episode that threads a bowl out from between two dishes then carries it through open air.
```

### QUESTION  (sha1 `697bf64ffaca`)

```
The measurements above are stated as fact -- do not re-estimate or repeat them. Answer each check from what the cameras show about the MOMENT in front of you, read together with those measurements.
Answer each check on its own line as "A) 3", in order, nothing else -- one digit from 1 to 5 per check, rating how far that check describes this moment:
  5 = it is happening right now -- the picture shows the contact, the grip or
      the position the check describes
  4 = not yet, but the hand is right up against it, a centimetre or a single
      motion away
  3 = the hand is heading for it and still some way off
  2 = the thing the check is about is there in the picture, but the arm is busy
      with something else
  1 = there is nothing in this picture the check could be about
A grade refers only to the check on that line.
A) Is an object being set down on a small fixed spot it has to sit squarely on
   -- a plate, a burner, a rack -- rather than into something that would catch
   it?
B) Is the gripper closing on the object right now, or about to -- the fingers
   already around it and only the squeeze left, so that a slip here loses it?
C) Is an object being put into something that would catch it -- a basket, a bin,
   a bowl -- so that landing off-centre changes nothing?
D) Is the hand letting go, or about to -- the object already resting where it is
   meant to end up, so that what remains is only to open and withdraw?
E) Is this a job that needs no hold kept on anything -- a push, a knob turned, a
   drawer pressed shut -- so that once it is sent it carries on without the hand?
Answer:
```

### 계산 사실 (실제로 나간 형태, 한 청크 예시)

```
pick up the black bowl on the cookie box and place it on the plate
MEASURED FROM THE PLANNED MOTION over the chunk ahead (these are computed facts, not estimates): the gripper stays closed throughout; the end-effector keeps a consistent direction; it is moving at a normal pace (mean step 0.42, peak 0.61); it is holding something while creeping along; it is decelerating to a near stop.
```

### conf 계산

```
g(등급) = (기댓값등급 - 1) / (NGRADE - 1)
conf    = (1 + Σ_가점 w·g - Σ_감점 w·g) / 2
```
기댓값 등급은 등급 토큰 자리의 softmax 분포에서 Σ (i+1)·p(i) 로 낸다.
정수 등급으로 계산하면 conf 가 계단이 되어 역치가 안 먹는다(하네스 R4).

---

## allex

단일 배속 게이트. 서브태스크 라벨 없이 지시문만. v4→v4c, 세 판.

- 모듈 `scripts/allex_v4c_checks.py`
- 계산 사실 `scripts/allex_v3_checks.py: facts_v3`
- 등급 수 5 · 문항 4개
- 감점 AB (가중 합 1.00) · 가점 CD (가중 합 1.00)

| 문항 | 이름 | 부호 | 가중 |
|---|---|---|---|
| A | PINCH_RIGID | 감점 | 0.800 |
| B | POUCH | 감점 | 0.200 |
| C | FREE_END | 가점 | 0.300 |
| D | IN_TRANSIT | 가점 | 0.700 |

### GUIDANCE  (sha1 `733d71476881`)

```
You are judging one instant of a two-armed robot with hands, to decide how much of the next stretch of its motion could be thinned out -- how many of its commanded poses could be dropped, letting the arms travel further between the ones that remain, without changing the outcome.

What thinning takes away is the arms' chance to correct on the way, and there are two ways that loses the object. One is a hold that exists only as a balance: the thing is trapped between two hands and nothing is closed around it, so the hold IS the push of one hand against the other. If that thing is stiff it cannot give, and a coarser swing changes the balance and drops it. Something soft squashes and stays pinched.

The other is a load that does not hold its own shape. A bag or a sagging thing hangs, swings and shifts while it is carried, and the further the arms travel between commanded poses the further it swings.

Most moments are neither. Sliding something across to the other side, pushing something that is still resting on a surface, or moving with nothing in the hands -- a longer step changes nothing.

Judge the moment in front of you, not the name of the job. One segment passes through several of these from one second to the next.
```

### QUESTION  (sha1 `6fc571516e1d`)

```
The measurements above are stated as fact -- do not re-estimate or repeat them. Answer each check from what the cameras show about the MOMENT in front of you, read together with those measurements.
Answer each check on its own line as "A) 3", in order, nothing else -- one digit from 1 to 5 per check, rating how far that check describes this moment:
  5 = clearly true of this moment
  4 = mostly true
  3 = partly true
  2 = barely true
  1 = not true at all
A grade refers only to the check on that line.
A) Are the two hands SQUEEZING something between them to hold it, and is that
   thing HARD -- a box, a carton, something that will not squash?
B) Is the item a POUCH OR PACKET -- a soft mailer, a padded envelope, a
   plastic parcel -- rather than a rigid carton with square sides?
C) Is the thing only being SENT ACROSS -- slid or passed over to the other side,
   or shoved on its way -- with no particular spot it has to come to rest on?
D) Are the hands ON THEIR WAY -- reaching out towards something, or drawing back
   from what they have finished, carrying nothing between them?
Answer:
```

### conf 계산

```
g(등급) = (기댓값등급 - 1) / (NGRADE - 1)
conf    = (1 + Σ_가점 w·g - Σ_감점 w·g) / 2
```
기댓값 등급은 등급 토큰 자리의 softmax 분포에서 Σ (i+1)·p(i) 로 낸다.
정수 등급으로 계산하면 conf 가 계단이 되어 역치가 안 먹는다(하네스 R4).

---

