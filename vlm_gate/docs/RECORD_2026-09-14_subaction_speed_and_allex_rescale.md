# 기록 2026-09-14 · 서브액션 배속 결정 + allex 상한 재조정

키워드: 배속 결정 · speed ratio · 서브액션 · sub-action · 국면 · phase ·
quantizability gate · robocasa 폐루프 · clip-scale · allex ceiling rescale ·
hojin_quantization_confidence.json · 추종비 · tracking ratio

**이 문서는 무엇인가.** 2026-09-14 하루에 나온 분석·구현·적용을 전부 적는다.
배속 라벨링은 이 날 이후 **보류**로 결정됐다. 나중에 재개하는 사람이 처음부터
다시 하지 않도록, 되는 것과 안 되는 것과 그 이유를 남긴다.

---

## 0. 한 장 요약

| | 결론 |
|---|---|
| 서브액션 배속 (VLM 없음) | 균일 배속보다 **낫다**. 0.714 vs 균일 2.5x 0.650 · 2.0x 0.657. 단 n=77, ±0.052 |
| clip 해제 | **필수**. 켠 채로는 0.650 으로 동률, 풀면 0.714. 사다리가 클립 푼 조건이다 |
| VLM 이 후보 중 고르기 | **나빠진다.** 0.714 -> 0.575. 다만 판정 호출이 0~10% 뿐이라 VLM 탓이 아니다 |
| 진짜 병목 | **그리퍼 기반 국면 정의.** 문 태스크는 100% 가 "접근" 이 된다 |
| 고칠 신호 | **추종비**(실제 EE 이동 / 명령 크기). 접촉 구간에서 3~30배 떨어진다 |
| 문항 아홉 판이 다 실패한 이유 | 하네스 **R1 위반** -- 태스크당 지표로 채점했다 |
| allex 상한 재조정 | **된다.** VLM 재실행 없이. 도구와 인계문 완성 |

---

## 1. 파일 (전부 절대경로)

### 배속 결정 파이프라인

```
/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/
  scripts/subaction_rate.py                     국면->후보->배속. rate_for / candidates / pick_from
  scripts/robocasa_service_compress.py          --subaction-rate · --subaction-vlm
  run_scripts/eval/eval_subaction_rate.sh       VLM 없는 판 (GPU1)
  run_scripts/eval/eval_subaction_vlm.sh        VLM 판 (GPU2: 정책+판정)
  analysis/subaction/robocasa_subactions.json         좁힌 후보 목록 (태스크당 국면마다 1개)
  analysis/subaction/robocasa_subactions_wide.json    넓힌 후보 목록 (에피소드마다 갈릴 짝 포함)
  analysis/subaction/robocasa_limits_pinned.json      서브액션별 배속 한계
  analysis/subaction/RESULTS.md                       측정 기록 (이 문서와 중복되나 더 날것)
```

### 문항

```
  scripts/ratio_checks_sa1.py            속성 5문항 (전 태스크 동일). VLM 판이 쓴 것
  scripts/ratio_checks_scene.py          장면 문항 v1 (에피소드별)
  scripts/ratio_checks_scene2.py         장면 문항 v2 -- A 문구만 교체
  scripts/ratio_checks_scene2_FROZEN.py  v2 동결본 (홀드아웃용)
  scripts/ratio_ep_scene_label.py        에피소드 시작 프레임 라벨러
  analysis/ratio_prompt/ep_scene.jsonl   v1 라벨 72행
  analysis/ratio_prompt/ep_scene2.jsonl  v2 라벨 72행
```

### allex 상한 재조정

```
  scripts/allex_rescale_ceiling.py                  도구
  docs/HANDOFF_allex_ceiling_rescale.md             다른 서버 에이전트용 인계문
  output/allex_rescale/spec_20260914_fine.json      최종 산출물
```

### 폐루프 산출물

```
  output/robocasa/subaction_rate_pin4/            접근·운반 4.0 고정 (버린 판)
  output/robocasa/subaction_rate_pin25_clipon/    2.5 고정, clip 켬
  output/robocasa/subaction_rate_noclip/          2.5 고정, clip 해제  ← 기준선
  output/robocasa/subaction_vlm/                  VLM 이 후보 중 선택
```

---

## 2. 방법 -- 서브액션 분해로 배속을 정한다

교수님 요구: VLM 라벨이 "압축할지 말지" 가 아니라 **몇 배속까지 되는지**를 정해야
한다. 사용자가 준 방법: 태스크를 원자 동작으로 쪼개고, 태스크별 실측 상한을
교차 비교해 **서브액션별 한계를 역산**하고, 그 규칙을 VLM 에게 넘긴다.

### 2-1. 병목 모형

태스크의 상한 = 그 태스크가 지나는 서브액션들의 **최솟값**. 예를 들어
A(잡기->운반->조심히 놓기)가 2배까지, B(잡기->운반->대충 놓기)가 3배까지면
잡기·운반은 3배까지 되고 조심히 놓기가 2배에서 막는 것이다.

역산 결과 (`analysis/subaction/robocasa_limits_pinned.json`):

```
place_precise 1.25 · place_open 1.50 · pull_swing 1.67 · place_catcher 2.00
grasp_open 2.09 · grasp_confined 2.41 · approach 2.5 · carry 2.5
contact_coarse 2.54 · contact_knob 3.41
```

LOO RMSE 0.833 · 순위상관 **+0.804**.

### 2-2. 식별 불가능한 항 -- 접근·운반

모든 태스크에 다 들어 있는 서브액션(접근·운반)은 **교차 비교로 값이 안 나온다.**
어느 태스크에서도 병목이 아니면 상한을 아무리 크게 잡아도 모형이 맞는다.

4.0 으로 두고 폐루프를 돌렸더니 실현 배속이 3.7~4.0 까지 올라가 3개 중 2개를
잃었다(`output/robocasa/subaction_rate_pin4/`). 사용자 지시로 **2.5 고정**.
그때 순위상관이 +0.632 -> **+0.804** 로 올랐다.

### 2-3. 국면 판정 (지금의 결함)

`scripts/subaction_rate.py: phase_of_chunk()` 는 그리퍼로 정한다.

```python
dg = np.diff(grip)
if np.any(dg > 0.5): return "파지"
if np.any(dg < -0.5): return "해제"
return "운반" if g.mean() > 0.5 else "접근"
```

지시문이 서브액션 목록을 주고 액션이 국면을 주면 교집합이 유일해진다 -- 좁힌
목록에서 **24 태스크 96 칸 전부 후보가 1 개**였다. 그래서 VLM 이 고를 것이 없다.

**이 정의가 무너지는 곳이 있다. 3절을 보라.**

---

## 3. 가장 중요한 발견 -- 국면 정의가 병목이다

### 3-1. 증상

폐루프에서 `subaction_picks` 의 approach 비율:

```
CloseDoubleDoor   99.9%      OpenDoubleDoor  92.0%
CoffeeServeMug    83.5%      PnPCounterToSink 57.0%
```

판정 호출률이 0.0% / 2.4% / 4.4% / 9.8% 였다. **VLM 에게 물어볼 자리가 없었다.**

### 3-2. 원인 (학습 데이터에서 GPU 없이 재현됨)

데이터셋: `/sjw_alinlab/home/hojin2/quantization_agent_workspace/assets/datasets/robocasa_n17_mirror`
(LeRobot 형식. parquet 컬럼은 `observation.state` 와 `action`.
chunk 당 300 에피이므로 `episode_index//1000` 로 경로를 만들면 안 된다.)

액션 인덱스: `end_effector_position` = 5:8 (델타) · 마지막 채널 = 그리퍼.
상태 인덱스: `end_effector_position_absolute` = 7:10 · `gripper_qpos` = 21:23.

| task | 그리퍼 전이 있는 에피 | 평균 전이수 | 그리퍼 채널 값 |
|---|---|---|---|
| **CloseDoubleDoor** | **0%** | 0.00 | `[0.0]` 항상 열림 |
| CoffeeServeMug | 100% | 2.13 | `[0.0, 1.0]` |
| OpenDoubleDoor | 100% | 4.00 | `[0.0, 1.0]` |
| PnPCounterToSink | 100% | 2.07 | `[0.0, 1.0]` |

**CloseDoubleDoor 는 300 에피 전부에서 그리퍼를 한 번도 안 쓴다.** 문을 손등으로
밀어 닫기 때문이다. 그래서 에피소드 전체가 "접근" 이 되고, 실제로 문을 미는
접촉 구간까지 접근 상한 2.5 로 돈다.

같은 규칙을 학습 데모에 적용하면 폐루프와 거의 같은 분포가 나온다 -- 즉
**GPU 없이 고칠 수 있다**:

| task | 학습데모 접근 비율 | 폐루프 approach |
|---|---|---|
| CloseDoubleDoor | 100% | 99.9% |
| OpenDoubleDoor | 80% | 92.0% |
| CoffeeServeMug | 63% | 83.5% |
| PnPCounterToSink | 57% | 57.0% |

### 3-3. 고칠 신호 -- 추종비

```
추종비 = |실제 EE 이동| / |명령 크기|
       = norm(diff(state[:, 7:10])) / norm(action[:, 5:8])
```

문이 버티면 명령은 계속 나가는데 손끝이 안 간다. CloseDoubleDoor 한 에피의
청크별 추종비: 기준 0.0122 인데 접촉 구간만 **0.0004 ~ 0.007** 로 3~30 배 낮다.

에피소드 중앙값의 0.5 배를 문턱으로 잡으면:

| task | 접촉으로 잡히는 칸 | 에피당 | 그리퍼로는 |
|---|---|---|---|
| CloseDoubleDoor | 6.6% | 1.7칸 | **0%** |
| CoffeeServeMug | 9.2% | 1.9칸 | 37.0% |
| OpenDoubleDoor | 10.1% | 4.5칸 | 19.8% |
| PnPCounterToSink | 11.4% | 3.2칸 | 43.1% |

그리퍼가 아예 못 보던 CloseDoubleDoor 에서 에피당 1.7 칸을 잡는다. 두 신호는
**배타가 아니라 보완**이다 -- 그리퍼는 파지·해제를, 추종비는 밀기·버티기를 본다.

**주의 두 가지.**
- 추종비의 **절대값은 네 태스크가 0.0122~0.0125 로 같다.** 절대 문턱으로는 안
  갈린다. 반드시 에피소드 상대 문턱을 쓴다. (등급 절대 문턱 3.5 로 잘랐다가
  72.7% 가 기본값으로 떨어지고 상관이 -0.430 이 된 전례가 있다.)
- 폐루프에서는 에피소드 중앙값을 미리 모른다. 지금까지 본 청크의 누적 중앙값을
  써야 하고 초반 몇 청크는 기준이 불안정하다. **미구현.**

---

## 4. 폐루프 측정

4 태스크 x 20 에피. 사다리(`analysis/eval_results/LADDERS.json` 의
`robocasa/uniform`)와 비교한다. **사다리는 클립 푼 조건이다.**

### 4-1. clip 이 결론을 바꾼다

`--clip-scale` 기본값이 1.0(클립 켬)이라 처음에 조건이 안 맞았다. 사다리와 같은
`CLIP_SCALE=10` 으로 다시 돌렸다.

| task | 1x | 균일 2.0x | 균일 2.5x | 서브액션 clip켬 | 서브액션 clip해제 |
|---|---|---|---|---|---|
| CloseDoubleDoor | 0.86 | 0.70 | 0.70 | 0.70 | 0.65 |
| CoffeeServeMug | 0.78 | 0.60 | 0.64 | 0.65 | 0.82 |
| OpenDoubleDoor | 0.92 | 0.80 | 0.64 | 0.70 | 0.65 |
| PnPCounterToSink | 0.70 | 0.52 | 0.62 | 0.55 | 0.75 |
| **평균** | 0.815 | 0.657 | 0.650 | **0.650** | **0.714** |

클립을 켜면 균일 배속과 동률, 풀면 +0.064 앞선다. n=77 이라 표준오차 ±0.052 --
**추세이지 확정이 아니다.**

### 4-2. VLM 이 후보 중에서 고르면 나빠진다

문항은 전 태스크 동일(`ratio_checks_sa1`), 태스크마다 다른 것은 넓힌 후보 목록뿐.

| task | 기준선(무VLM) | VLM | 판정 호출률 |
|---|---|---|---|
| CloseDoubleDoor | 13/20 | 11/20 | **0.0%** |
| CoffeeServeMug | 14/17 | 13/20 | 4.4% |
| OpenDoubleDoor | 13/20 | 10/20 | 2.4% |
| PnPCounterToSink | 15/20 | 12/20 | 9.8% |
| 합계 | **0.714** | **0.575** | |

차이 0.139, 표준오차 0.076 (p~0.07).

**잡음 바닥을 먼저 보라.** CloseDoubleDoor 는 판정 호출이 0 회이고 rate_mean
2.5000 으로 기준선과 같은 설정인데도 13/20 -> 11/20 이 나왔다. **동일 설정에서
2 에피 차이가 난다.**

기전: 후보를 넓히면 VLM 이 **더 빠른 쪽**을 고를 수 있다. 파지에서 기준선은
grasp_open 2.093 하나인데 넓힌 목록은 grasp_confined 2.412 도 후보다.

---

## 5. 왜 배속 문항 아홉 판이 다 실패했나 -- 하네스 R1 위반

`docs/QUESTION_DESIGN_HARNESS.md` 의 R1:

> 문항은 태스크당 지표로 채점하지 않는다. ... 라벨이 태스크 상수로 수렴한다.
> 주 채점은 국면 지표로 하고 태스크 상관은 부호 확인용으로만 쓴다.

v1a · v1b · v2a · v2b · v3 · v5 · sa1 · sa2 · LIBERO-sa1 을 **전부 태스크당
순위상관으로 채점했다.** R1 이 예고한 대로 수렴했다 -- sa1 은 범위 2.2~2.5 로
사실상 상수, sa2 는 69% 가 한 답으로 붕괴.

### 5-1. 태스크 축이 애초에 작다

`subaction_rate_pin4` 75 에피의 성패 분산을 나누면

```
태스크 안 (장면 차이)  0.2278   93.1%
태스크 간 (배속 차이)  0.0168    6.9%
```

태스크마다 값이 하나인 한계표가 건드릴 수 있는 몫이 **7%** 다. 거기서 순위상관
+0.804 를 달성해도 폐루프가 안 움직이는 것이 당연하다. **방법이 틀린 것이 아니라
축이 작다.**

같은 배속 스케줄을 먹은 20 에피소드가 절반만 성공한다. 나머지를 가르는 것은
장면이고, **장면은 에피소드마다 변하는 유일한 입력**이다 (지시문은 태스크당
하나로 고정, 액션 계산은 그리퍼 상태만 준다).

### 5-2. 장면 문항 (에피소드별 채점, R1 준수)

pin4 72 에피의 **시작 프레임 + 성패**로 채점. 태스크 평균을 뺀 뒤 성공과의 상관.

| 문항 | v1 | v2 (A 문구만 교체) |
|---|---|---|
| A REACH / OFFSCREEN | −0.216 | −0.165 |
| B CONFINED | −0.065 | **−0.216** |
| C SMALL | +0.020 | **−0.221** |
| **합산 conf** | +0.078 (p=0.52) | **+0.238 (p=0.044)** |

첫 에피소드별 유의 신호다. **그러나 A 만 바꿨는데 B·C 가 같이 움직였다**
(R2 가 경고한 간섭) -- 이득을 A 덕이라 말할 수 없다. A 는 argmax 가 72/72 3등급인
죽은 문항이고 **소프트값(기댓값 등급)만 움직인다.** n=72, 두 번째 시도.
홀드아웃 안 함. v2 는 `ratio_checks_scene2_FROZEN.py` 로 동결해 뒀다.

---

## 6. 버린 것들 (다시 시도하지 말 것)

| 시도 | 결과 |
|---|---|
| 접근·운반을 4.0 으로 | 실현 배속 3.7~4.0, 3개 중 2개 잃음 |
| 등급 절대 문턱 3.5 | 72.7% 가 기본값으로, 상관 −0.430 |
| conf 절대값 -> 배속 | 배속이 사실상 상수 (±0.04) |
| 4지선다 국면 분류 | 32% 가 한 답으로 붕괴, 파지·해제 예측 0% |
| sa2 후보 목록 다지선다 | 69.4% 가 carry 로 붕괴 (단 `n_grade=1` 버그가 섞여 있어 결론은 오염됨) |
| 태스크별로 문항 문구를 다르게 | **금지.** 태스크마다 배속을 손으로 정해주는 것과 같아 일반화가 0 |
| 스텝 기반 보간으로 판정 | 성공-스텝 곡선이 단조롭지 않아 가짜 +0.62 가 나왔다. 배속으로 보간해야 한다 |

---

## 7. 반복해서 걸린 함정

- **`--clip-scale` 기본값이 1.0 이다.** 사다리와 비교하려면 `CLIP_SCALE=10`.
- **`MODEL_OUTPUT_DIR` 없으면 sbatch 가 거부한다.** `/rlwrld-unified-checkpoints/$USER/...`
- **게이트 블록이 이미 판정기를 부른다.** `--subaction-vlm` 분기에서 또 부르면
  청크마다 두 번 돈다. 위에서 나온 `_picks` 를 재사용한다.
- **후보가 1 개면 판정기를 부르지 않는다.** 판정이 호출당 ~900ms 로 병목이다.
  다만 그렇게 건너뛰면 `q` 가 None 이 되어 `gate_yes += int(q)` 가 죽는다 --
  중립값 `conf, q = 0.0, False` 를 넣어야 한다.
- **robocasa_n17_mirror 는 chunk 당 300 에피다.** `//1000` 로 경로를 만들면 없다.
- **`/tmp` 는 노드 로컬이다.** 워커에서 읽을 파일은 공유 경로(`_tmp/`)에 둔다.
- **tmux 창이 전경 작업에 잡히면 `alloc.sh run` 이 안 먹는다.** 병렬 클라이언트는
  `setsid nohup ... & disown` 으로 띄워 셸을 비워 둔다.
- **NFS 에 재귀 find/grep 금지.** 표적 조회만.

---

## 8. allex 상한 재조정 (별건, 완료)

`docs/HANDOFF_allex_ceiling_rescale.md` 가 본문이다. 요점만:

- 배달본 `/rlwrld2/home/david/action_quantization/v1/subtask_labeled_data_update_eef_256x256_hojin/meta/hojin_quantization_confidence.json`
  은 청크마다 `p`(1단 신뢰도)와 `K_max`(청크별로 조인 상한)를 담고 있고
  `K = snap(1 + p*(K_max-1))` 로 만들어졌다. **재현율 99.885%** 이므로 VLM 재실행
  없이 상한만 옮겨 다시 구울 수 있다.
- 상한은 **비율로** 옮긴다: `K_max' = 1 + (K_max-1)*(T-1)/(C-1)`. 2단이 청크마다
  다르게 조인 구조를 보존한다.
- 격자를 늘려야 한다. 배달본 격자 `[1,2,2.5,3]` 은 1 과 2 사이가 비어 있어, 상한을
  내리면 대부분이 1.0 으로 떨어진다. `--grid fine` = `[1,1.5,1.75,2,2.5,3]`.
  **1.75 만 넣으면 거의 안 살아난다 -- 1.5 눈금이 있어야 한다.**
- 최종 지시값: 상한 `Bring 2.0 · Pass 2.5 · Rotate Box 1.75 · Rotate PolyBag 3.0`,
  하한 전부 1.5.
- **Rotate PolyBag 의 p 가 거꾸로다.** 운용자는 "봉투는 빨리 뒤집어도 된다" 인데
  p 평균이 0.189 로 네 서브태스크 중 최저다 (Bring 0.689 · Pass 0.630 ·
  Rotate Box 0.376). 상한을 3.0 까지 열어 순서는 바로잡았지만 여전히 절반이
  K=1.0 이다. **p 자체를 고치는 것은 라벨의 판단을 바꾸는 일이라 손대지 않았다.**
- `BRING_SOFT`(봉투면 Bring 을 2.0 으로)는 **소급 적용 불가**. 2단 문항 D 의 답이
  배달본에 없다.

---

## 9. 다음에 재개하면

1. **`phase_of_chunk` 에 추종비를 넣는다.** 3-3 절. 누적 중앙값 문턱을 설계해야
   한다. 오프라인 검증까지는 GPU 없이 된다.
2. 그 위에서 기준선(무VLM)을 다시 재고, 그 다음에 VLM 을 얹는다. **순서를
   지킨다** -- 국면이 안 보이는 상태로 VLM 을 얹으면 물어볼 자리가 없다.
3. 채점은 **에피소드별**로 한다 (R1). 태스크당 순위상관은 부호 확인용으로만.
4. 장면 문항 v2 를 새 폐루프의 에피소드에서 **홀드아웃 검증**한다. 지금 +0.238
   (p=0.044) 은 n=72 두 번째 시도라 믿을 수 없다.

---

## 10. allex 배달본은 두 개다 -- 리스케일 대상을 헷갈리지 말 것

2026-09-15 추가. 8절에서 다룬 JSON 은 **리스케일 대상이 아니었다.**

| | 경로 | 라벨이 어디에 |
|---|---|---|
| (가) | `/rlwrld2/home/david/action_quantization/v1/subtask_labeled_data_update_eef_256x256_hojin/` | `meta/hojin_quantization_confidence.json` · 14,809 청크 · `p`·`K_max`·`K`·`hojin(0~1)` |
| (나) | `/rlwrld2/home/david/action_quantization/merged_v5tempo_hojin/` | **parquet 의 `hojin` 열** · 프레임마다 배속 K (float32, 1.5~3.0) |

**실제 리스케일 대상은 (나)다.** (가)의 도구(`allex_rescale_ceiling.py`)는 못 쓴다 --
(나)에는 `p`·`K_max` 가 없고 최종 K 만 있어서 비례 이동의 기준을 p 에서 얻을 수
없다. 기준을 **그 서브태스크의 현재 최대 K** 로 잡는 별도 도구가 필요하다:
`scripts/allex_rescale_hojin_column.py`.

(나)를 파일명으로 훑으면 라벨이 안 보인다. `meta/` 에는 `subtasks.jsonl` ·
`episodes.jsonl` 뿐이고 라벨은 데이터 열이다. **열 이름으로 찾아야 한다.**

### 10-1. 기준점을 하한에 둔다

    anchor=floor (기본)   K' = F + (K - F) * (T - F) / (C - F)
    anchor=one            K' = 1 + (K - 1) * (T - 1) / (C - 1)

(나)는 **하한 1.5 가 이미 지켜지고 있다.** 1 을 기준으로 줄이면 1.5 가 1.25 로
내려가 하한을 깨고, 다시 1.5 로 자르면 바닥이 뭉개진다. 전체 평균으로는
1.7852(floor) vs 1.7155(one).

### 10-2. 적용 결과 (2026-09-15, 전량 1,280 에피)

상한 `Bring 2.0 · Pass 2.5 · Rotate Box 2.0 · Rotate PolyBag 2.5`, 하한 1.5.

| 서브태스크 | 현재최대 | 새상한 | 배율 | 평균 전 -> 후 |
|---|---|---|---|---|
| Rotate Box | 3.00 | 2.00 | 0.333 | 1.760 -> 1.587 |
| Bring Object | 3.00 | 2.00 | 0.333 | 2.304 -> 1.768 |
| Pass Object | 3.00 | 2.50 | 0.667 | 2.439 -> 2.126 |
| Rotate PolyBag | 3.00 | 2.50 | 0.667 | 2.255 -> 2.003 |

전체 평균 배속 **2.1207 -> 1.7852**.

산출물: `/rlwrld2/home/david/action_quantization/merged_v5tempo_hojin_recal/`
(`data/` 만 새로 쓰고 `meta`·`videos` 는 원본 심볼릭 링크. **원본 불변.**)

산출물을 다시 읽어 검증했다(320 에피): 행수·`frame_index` 보존, 상한 초과 프레임
0, 하한 미만 프레임 0, 값이 바뀐 열은 `hojin` 하나뿐.

**Rotate Box 가 1.589 로 사실상 하한에 붙는다.** 원래 값의 47% 가 이미 1.5 여서
상한을 3.0 -> 2.0 으로 내리면 대부분이 바닥에 깔린다.
