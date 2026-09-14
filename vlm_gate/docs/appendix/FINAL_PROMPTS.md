# 부록: 최종 VLM 프롬프트 전문

`scripts/dump_final_prompts.py` 가 문항 모듈에서 직접 읽어 생성한다. 손으로 옮겨 적지 않는다.

판정기는 `nvidia/Cosmos3-Nano` 이고 그리디 디코딩으로 텍스트를 답한다. 등급 토큰 자리의 로짓은 그 답이 얼마나 확실했는지를 덧붙이는 데만 쓰며, 강제 슬롯 argmax 로 답을 짓지 않는다.

프롬프트는 세 덩이로 조립된다 -- **계산 사실**(지시문 + 액션에서 계산한 사실 문장), **GUIDANCE**(무엇을 판단하는 일인지), **QUESTION**(등급 척도와 문항). 계산 사실은 프롬프트의 일부다. 빼고 물으면 다른 질문이 된다.

---

## robocasa

24 태스크 · 압축 여부 게이트. v1→v7, 네 판.

- 모듈 `scripts/phase9_checks_v7.py`
- 계산 사실 `scripts/judge_ab.py: facts_text`
- 등급 수 5 · 문항 5개
- 감점 AB (가중 합 1.00) · 가점 CDE (가중 합 1.00)

| 문항 | 이름 | 부호 | 가중 |
|---|---|---|---|
| A | CLOSE_EXACT | 감점 | 0.600 |
| B | HEMMED_IN | 감점 | 0.400 |
| C | FIXTURE_HELD | 가점 | 0.500 |
| D | LETTING_GO | 가점 | 0.150 |
| E | CARRY_OPEN | 가점 | 0.350 |

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

40 태스크 · 압축 여부 게이트. 문구가 모듈이 아니라 텍스트 파일에 있다 -- 파일 이름이 v2 인데 v3c 가 쓴다. 다음에 고칠 사람이 헷갈리지 않게 적어 둔다.

- 모듈 `scripts/libero_v3c_checks.py`
- 계산 사실 `scripts/libero_checks.py: facts`
- 등급 수 5 · 문항 5개
- 감점 AB (가중 합 1.00) · 가점 CDE (가중 합 1.00)

| 문항 | 이름 | 부호 | 가중 |
|---|---|---|---|
| A | SMALL_TARGET | 감점 | 0.636 |
| B | CROWDED_PICK | 감점 | 0.364 |
| C | CATCHER | 가점 | 0.533 |
| D | WIDE_SURFACE | 가점 | 0.267 |
| E | NO_HOLD | 가점 | 0.200 |

### 프롬프트 전문  (`prompts/libero_v2.txt`, sha1 `8d73fef1528c`)

문구가 모듈이 아니라 이 파일에 있다. 실제로 판정기에 나가는 형태 그대로이며, 맨 위 INSTRUCTION 블록이 계산 사실이다.

```
### INSTRUCTION (에피소드 지시문 + 계산 사실)
pick up the black bowl on the cookie box and place it on the plate
MEASURED FROM THE PLANNED MOTION over the chunk ahead (these are computed facts, not estimates): the gripper stays closed throughout; the end-effector keeps a consistent direction; it is moving at a normal pace (mean step 0.42, peak 0.61); it is holding something while creeping along; it is decelerating to a near stop.

### GUIDANCE
You are judging one moment of a table-top robot's motion, to decide how much of it could be thinned out -- how many of its commanded poses could be dropped, letting the arm travel further between the ones that remain, without changing the outcome.

Two things decide that, and both are about where the object has to end up or where it has to come from, not about how fast the arm happens to be going.

A destination can be forgiving or unforgiving. Dropping something into a basket or a bowl is forgiving: the container catches it, and landing a centimetre off changes nothing. Setting something down on a plate, on a burner, or into a rack is not: it has to come to rest squarely on a small support, and the last approach is where a longer step puts it down askew.

An approach can be open or threaded. Reaching for something that stands clear on the table is open. Reaching for something perched on top of another object, or wedged between two things standing close by, leaves one line in: a longer step is a wider swing, and what should have been cleared gets struck.

Most of the time neither is true -- the gripper is crossing open space, or is already holding something with room around it, and a longer step changes nothing. That is the common case.

Judge the moment in front of you, not the task as a whole. One task passes through both kinds from one moment to the next: the same episode that threads a bowl out from between two dishes then carries it through open air.

### QUESTION
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
B) Is the thing being taken perched on top of another object, or wedged in among
   things standing close by, so that there is only one line in for the hand?
C) Is an object being put into something that would catch it -- a basket, a bin,
   a bowl -- so that landing off-centre changes nothing?
D) Is an object being set down on a surface so much larger than itself that
   where on it the object lands makes no difference -- a table top, a counter,
   the top of a cabinet -- and not onto a dish or a spot marked out for it?
E) Is this a job that needs no hold kept on anything -- a push, a knob turned, a
   drawer pressed shut -- so that once it is sent it carries on without the hand?
Answer:

### SIGN / WEIGHT
  A  감점  0.636
  B  감점  0.364
  C  가점  0.533
  D  가점  0.267
  E  가점  0.200
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
| A | PINCH_RIGID | 감점 | 0.600 |
| B | SWINGING_LOAD | 감점 | 0.400 |
| C | FREE_END | 가점 | 0.550 |
| D | IN_TRANSIT | 가점 | 0.450 |

### GUIDANCE  (sha1 `733d71476881`)

```
You are judging one instant of a two-armed robot with hands, to decide how much of the next stretch of its motion could be thinned out -- how many of its commanded poses could be dropped, letting the arms travel further between the ones that remain, without changing the outcome.

What thinning takes away is the arms' chance to correct on the way, and there are two ways that loses the object. One is a hold that exists only as a balance: the thing is trapped between two hands and nothing is closed around it, so the hold IS the push of one hand against the other. If that thing is stiff it cannot give, and a coarser swing changes the balance and drops it. Something soft squashes and stays pinched.

The other is a load that does not hold its own shape. A bag or a sagging thing hangs, swings and shifts while it is carried, and the further the arms travel between commanded poses the further it swings.

Most moments are neither. Sliding something across to the other side, pushing something that is still resting on a surface, or moving with nothing in the hands -- a longer step changes nothing.

Judge the moment in front of you, not the name of the job. One segment passes through several of these from one second to the next.
```

### QUESTION  (sha1 `4e631aec122b`)

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
B) Is a LIMP load being carried along -- something that hangs, sags or sways
   under the hands as the arms travel?
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

