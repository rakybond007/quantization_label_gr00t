# F-level 체크포인트 RoboCasa 평가 (jimin_eval)

이 레포는 **평가 쪽 절반**이다. 학습과 모델 코드는 `F_level` 레포에 있고
(`papers/reproducing/FLARE`, 문서는 그쪽 `jimin_train.md`), 여기에는 RoboCasa
시뮬레이터를 돌리고 정책 서버에 물리는 **롤아웃 클라이언트**와 sbatch 러너가 있다.

```
정책 서버 (F_level 레포)  ←── ZMQ ──→  롤아웃 클라이언트 (이 레포)
serve_flevel_robocasa.py                scripts/robocasa_service_flevel.py
```

브랜치: **`flevel-conf-eval`**

```bash
git clone git@github.com:rakybond007/quantization_label_gr00t.git
git -C quantization_label_gr00t checkout flevel-conf-eval
```

---

## 0. 왜 레포가 둘인가

서버는 F-level 모델 클래스(`GR00T_N1_5_FLEVEL`, 레벨별 디코더 + ratio readout)를
띄워야 해서 `F_level` 안에 있어야 하고, 클라이언트는 RoboCasa 시뮬레이터 스택
(`robosuite` / `robocasa` / MuJoCo)이 깔린 환경에서 돌아야 한다. 두 환경은 의존성이
충돌해서 **서로 다른 conda env** 를 쓴다. 그래서 프로세스가 둘이고, ZMQ 로 붙는다.

> 참고: `F_level` 에도 같은 `gr00t/eval` 스택이 있어서 원리상 한 레포로 합칠 수 있다.
> 현재는 클라이언트가 여기 있다.

---

## 1. 환경

| 쪽 | 인터프리터 | 무엇이 들어 있나 |
|---|---|---|
| 서버 | `<workspace>/environments/flevel_hj/bin/python` | torch, transformers, F_level 의존성 |
| 클라이언트 | `~/miniconda3/envs/robocasa_eval/bin/python` | robosuite, robocasa, MuJoCo(EGL) |

클라이언트는 headless 렌더가 필요하므로 **`MUJOCO_GL=egl`** 을 반드시 준다.
서버는 오프라인 노드에서 HF 조회를 막기 위해 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

```bash
# 서버가 F_level 을 찾을 수 있어야 한다
export PYTHONPATH=<F_level>/papers/reproducing/FLARE:<F_level>
```

---

## 2. 한 태스크 돌려보기 (수동)

```bash
# (1) 서버 — F_level 레포에서
PYTHONPATH=<F_level>/papers/reproducing/FLARE:<F_level> \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
<flevel_env>/bin/python -u <F_level>/papers/reproducing/FLARE/scripts/serve_flevel_robocasa.py \
  --model-path <checkpoint> --port 5555 --mode conf --conf-threshold 0.5

# 뜨면 이 줄이 찍힌다 — 실제 적용된 grid/역치 확인용
#   INFO:root:conf mode: grid=[1, 2] tau=0.5000
#   Server is ready

# (2) 클라이언트 — 이 레포에서
MUJOCO_GL=egl <robocasa_env>/bin/python -u scripts/robocasa_service_flevel.py \
  --port 5555 --host localhost --env_name CloseDrawer \
  --video_dir <out>/CloseDrawer --seed 42 --n_episodes 50 \
  --max_episode_steps 1500 --generative_textures --no_record_video \
  --no-action-clip
```

### `--mode` 는 체크포인트 종류에 맞춰야 한다

| 체크포인트 | `--mode` | readout 이 내는 것 |
|---|---|---|
| 2-디코더 conf (`ratio_target_kind=conf`) | `conf` | conf 스칼라 → `conf >= tau` 로 압축 여부 |
| 4-디코더 ratio, FLARE 없음 | `ratio_current` | 4-way 클래스 |
| 4-디코더 ratio, FLARE 있음 | `ratio_future` | 4-way 클래스 |

**conf 체크포인트를 `ratio_current` 로 띄우면 안 된다.** 그 데이터 설정의 마지막
transform 이 carrier 를 클래스 인덱스로 범위검사하는데, conf 는 연속값이라 죽는다.
서버가 체크포인트의 `ratio_target_kind` 를 읽어 불일치면 바로 멈춘다.

### 주요 인자

| 인자 | 뜻 |
|---|---|
| `--conf-threshold τ` | **평가 인자**. 학습된 값이 아니다. 한 체크포인트로 τ 를 쓸어 성공률·배속 곡선을 그린다 |
| `--force-level 1` | readout 을 무시하고 K=1 강제 — 같은 체크포인트 안의 무압축 기준선 |
| `--force-level 2` | 압축 디코더 강제 (2-디코더면 K=2 또는 2.5) |
| `--no-action-clip` | arm delta saturation 제거 (아래 3절) |
| `--clip-scale N` | 클립 경계를 N 배로 넓힘 (제거가 아니다) |
| `--merge-by-k` | **압축 디코더가 없는 베이스라인 전용.** 서버가 16행 전부를 보내고 클라이언트가 합친다 |

---

## 3. 클립에 대해 — 중요

압축 디코더가 내는 한 행은 fine 액션 여러 개를 더한 값이라, fine 정책이 학습한
±1 범위를 예사로 넘는다. 그 범위는 **정규화 산물이지 하드웨어 한계가 아니다.**
잘라내면 OOD 명령만 만든다.

**`--no-action-clip`** 이 saturation 자체를 없앤다. 유한한 OSC affine scaling 은
유지하고, gripper·mobile base·PD gain·MuJoCo 액추에이터 한계는 건드리지 않는다.
압축이 합치는 축이 정확히 그 arm delta 6축(eef_pos 3 + eef_rot 3)이다.
RoboCasa `base_motion` 은 데이터셋 통계가 min=max=0 이라 항상 0 이고,
`gripper_close` 와 `control_mode` 는 블록의 마지막 값을 쓰지 합산하지 않는다.

**`--clip-scale N` 은 다른 물건이다** — 경계를 N 배로 넓힐 뿐이다. 그리고 RoboCasa
에서는 `--clip-scale 8` 이 사실상 클립을 안 거는 것과 같다. 원시 액션이 ±1 에 갇혀
있어 K=2 병합 행은 구조적으로 ±2, K=2.5 는 ±3 을 못 넘기 때문이다. 에피소드 200개
실측:

| | max\|a\| | >1 | >2 | >8 |
|---|---|---|---|---|
| K=1 | 1.000 | 0.000% | 0.000% | 0.000% |
| K=2 | 2.000 | 9.906% | 0.000% | 0.000% |
| K=2.5 | 3.000 | 12.403% | 2.439% | 0.000% |

즉 기본 경계(±1)에서는 K=2 성분의 9.9% 가 잘리지만 **±8 에서는 한 건도 안 잘린다.**
`--clip-scale 8` 로 돈 예전 결과와 `--no-action-clip` 결과는 같은 물리다.

패치는 **`env.reset()` 마다 다시 걸어야 한다** — robosuite 가 reset 에서 컨트롤러를
다시 만들기 때문에, 한 번만 걸면 두 번째 에피소드부터 조용히 클립이 돌아온다.
클라이언트가 이미 그렇게 하고 있고, 패치 건수가 0 이면 assert 로 죽는다. 로그로 확인:

```
[clip] arm delta saturation removed (1 controllers patched)
[clip]   probe 40.7 -> [2.035, 0.0, ...] finite=True no_action_clip=True
```

---

## 4. 24개 태스크 한 번에 (sbatch array)

```bash
cd <this repo>
VARIANT=vlm DEC=k2 ARM=readout N_EP=50 \
  sbatch run_scripts/eval/gpu26/eval_flevel_rc24_conf.sh
```

8-element array 로 태스크를 `i, i+8, i+16` 씩 나눠 맡는다 (DemoSpeedup·ISR 팔과 같은
묶음이라 태스크별로 숫자가 나란히 붙는다). 한 array 요소가 서버 하나를 띄우고
태스크 3개를 병렬 클라이언트로 물린다.

| 환경변수 | 기본값 | 뜻 |
|---|---|---|
| `VARIANT` | `vlm` | `vlm` / `vlm_contact` / `contact` — 어떤 라벨로 학습했나 |
| `DEC` | `k2` | `k2`(K=2) / `k25`(K=2.5) |
| `ARM` | `readout` | `readout` = readout 이 고름, `fine` = K=1 강제 |
| `TAU` | (없음) | 비우면 체크포인트의 값(보통 0.5) |
| `N_EP` | `50` | 태스크당 에피소드 |
| `NOCLIP` | `1` | `0` 이면 `CLIP_SCALE` 로 경계만 넓힘 |
| `CLIP_SCALE` | `1.0` | `NOCLIP=0` 일 때만 의미 |
| `STEP` | `60000` | 체크포인트 스텝 |
| `TASK_LIST` | (없음) | 공백 구분 태스크 이름 — 스모크용 |
| `RESUME` | `0` | `1` 이면 이미 기록된 에피소드는 건너뛰고 나머지만 |
| `OUT_BASE` | (없음) | 기존 결과 폴더를 지정 |

체크포인트 경로는 `/s3ckpt/$USER/flevel/conf_${DEC}_${VARIANT}_seed42/checkpoint-$STEP`
로 만들어진다. 다른 자리에 두었으면 스크립트의 `CKPT=` 한 줄만 고치면 된다.

### 스모크

같은 스크립트를 그대로 쓴다. 별도 스모크 스크립트는 러너와 어긋나기 마련이다.

```bash
sbatch --array=0 --time=1:30:00 \
  --export=ALL,VARIANT=vlm,DEC=k2,ARM=readout,N_EP=3,TASK_LIST="CloseDrawer" \
  run_scripts/eval/gpu26/eval_flevel_rc24_conf.sh
```

### 이어받기

중간에 끊겼거나 설정을 바꿔 나머지만 채우고 싶을 때:

```bash
VARIANT=vlm DEC=k2 ARM=readout N_EP=50 NOCLIP=1 RESUME=1 \
  OUT_BASE=$HOME/eval_out/flconf24_k2_vlm_readout_clip8_n50 \
  sbatch run_scripts/eval/gpu26/eval_flevel_rc24_conf.sh
```

클라이언트가 `prediction.txt` 의 `episode` 줄을 세서 그만큼 건너뛴다. 건너뛸 때도
`env.reset()` 은 호출하므로 **시드 정렬이 유지된다** — 이어서 도는 에피소드가
처음부터 돌렸을 때와 같은 초기 상태를 받는다. `RESUME=1` 에서는 `client.log` 를
덮어쓰지 않고 붙인다 (완료된 에피소드의 `levels=` 줄이 그 배속의 유일한 기록이다).
태스크마다 `clip_provenance.txt` 에 설정 전환 지점이 남는다.

`RESUME` 없이 돌리면 태스크 폴더를 지우고 처음부터 간다 — requeue 가 반쪽짜리
기록을 이어받아 조용히 짧아지는 걸 막기 위한 기본값이다.

---

## 5. 결과 읽기

태스크마다 `<out>/<TASK>/` 아래에 남는다.

| 파일 | 내용 |
|---|---|
| `prediction.txt` | `episode N is_success: [True] action_steps: 132 n_infer: 15` 줄과 맨 끝 요약 |
| `client.log` | 에피소드마다 `levels={1.0: 2, 2.0: 13}` — 실제 고른 배속 분포 |
| `flevel_chunks.csv` | 청크 단위 기록 (episode, chunk, step, level, k, readout_level, rows, ratio_probs) |

- **성공률** = `is_success: [True]` 비율
- **성공시 스텝** = 성공한 에피소드의 `action_steps` 평균. 압축이 걸릴수록 줄어든다
- **mean_k** = `levels=` 분포의 가중평균. 1.0 이면 압축이 한 번도 안 걸린 것

> `prediction.txt` 끝의 `flevel_levels:` 요약 줄은 **그 실행에서 새로 돈 에피소드만**
> 집계한다. 이어받기를 했다면 배속 분포는 `client.log` 의 `levels=` 줄을 모아 세라.

집계 예:

```python
import re, os, statistics as st
EP = re.compile(r"episode\s+\d+\s+is_success:\s*\[\s*(True|False)\s*\]\s*action_steps:\s*(\d+)")
LV = re.compile(r"levels=(\{[^}]*\})")
eps = [(m.group(1) == "True", int(m.group(2)))
       for line in open(f"{d}/prediction.txt") if (m := EP.search(line))]
succ = sum(ok for ok, _ in eps) / len(eps)
steps = st.mean([s for ok, s in eps if ok])
kn = kd = 0
for line in open(f"{d}/client.log", errors="ignore"):
    for m in LV.finditer(line):
        for k, c in eval(m.group(1)).items():
            kn += float(k) * c; kd += c
mean_k = kn / kd
```

---

## 6. 걸리기 쉬운 곳

| 증상 | 원인 |
|---|---|
| 서버가 `ValueError: level_map identity 이 1..2 밖의 level 을 가리킨다` | 구버전 `F_level`. `hj-two-decoder-conf` 최신으로 |
| 서버가 `--mode conf 인데 체크포인트는 ratio_target_kind='class'` | `--mode` 를 체크포인트 종류에 맞춰라 (2절) |
| 서버 기동이 10분 넘게 걸림 | `/s3ckpt` 는 S3 마운트다. 샤드당 ~230–290초. 정상이다. 러너는 30분까지 기다린다 |
| `zmq.error.ZMQError: Address already in use` | 같은 노드에서 포트가 겹쳤다. 러너는 `SLURM_ARRAY_JOB_ID` 로 포트를 흩는다 |
| 클라이언트가 조용히 죽음 | `MUJOCO_GL=egl` 이 빠졌거나 `--video_dir` 상위 폴더가 없다 |
| 배속이 전부 1.0 | τ 가 예측 conf 분포 위에 있다. 서버 로그의 `conf mode: ... tau=` 로 실제 값 확인 |
| `mean_k` 가 태스크마다 1.0 아니면 2.0 으로만 나옴 | 버그가 아니라 라벨 성질이다. conf 가 에피소드 안에서 거의 상수라 전역 τ 는 태스크 선택기처럼 작동한다 |
| `RESUME` 없이 돌렸더니 기존 결과가 사라짐 | 의도된 기본값이다. 이어받으려면 `RESUME=1 OUT_BASE=...` |

---

## 7. 이 레포에서 F-level 평가용으로 만진 것

| 파일 | 무엇 |
|---|---|
| `scripts/robocasa_service_flevel.py` | 롤아웃 클라이언트 (신규) |
| `run_scripts/eval/gpu26/eval_flevel_rc24_conf.sh` | 2-디코더 conf 체크포인트용 24-태스크 러너 (신규) |
| `run_scripts/eval/gpu26/eval_flevel_rc24.sh` | 4-디코더 ratio 체크포인트용 24-태스크 러너 |
| `vlm_gate/scripts/robocasa_service_compress.py` | `patch_clip_bounds` / `patch_no_action_clip` 제공 (기존) |

클라이언트의 두 가지 설계 결정:

1. **청크를 다시 합치지 않는다.** 디코더가 이미 병합된 행을 내주므로 한 행을 env
   스텝 하나로 그대로 실행한다. 여기서 또 합치면 이중 압축이다. `--merge-by-k` 는
   압축 디코더가 없는 베이스라인에 배속만 test-time 으로 적용하는 별도 팔이다.
2. **배속 그리드를 전선에서 읽는다.** 서버가 `_flevel_ratio_grid` 와 `_flevel_k` 를
   같이 보낸다. 모듈 상수 `RATIO_GRID = (1.0, 1.5, 2.0, 2.5)` 는 4-디코더 기본값이라
   2-디코더 체크포인트에서 level 2 를 K=1.5 로 잘못 읽는다.
