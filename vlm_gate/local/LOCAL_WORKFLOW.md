# 맥북에서 서버 작업하기

월요일부터 서버 안에서 에이전트를 못 돌린다. 에이전트는 맥북으로 내려오고,
**작업은 여전히 서버에서 100% 돈다.** 맥북에서 돌리는 파이썬은 한 줄도 없다.

## 구조

맥북의 Claude Code ──ssh──> 서버의 tmux 세션 `dev` ──srun──> 계산노드

서버에 셸을 하나 띄워놓고 그걸 원격 조종한다. 셸이 계속 살아있으므로
`cd`, `export`, `conda activate`, 그리고 **srun 할당**이 명령 사이에 유지된다.
노트북이 잠들거나 와이파이가 끊겨도 서버 쪽 작업은 계속 돈다.

## 하루 시작

```bash
export DEV_HOST=hojin2@<login-node>
tools/dev up                       # 세션 확보 (이미 있으면 그대로 재사용)
tools/dev alloc --gpus=1           # 대기열 한 번 서고, 세션이 계산노드로 올라감
```

이후 모든 명령은 그 GPU 노드 안에서 돈다. **대기열은 하루에 한 번만 선다.**

## 반복 루프 (지금과 형태가 같다)

```bash
# 맥북에서 파일 수정 (에이전트가 로컬 파일 편집)
tools/dev run 'python vlm_gate/scripts/cosmos_1call_v6.py --limit 4'
# → 코드가 먼저 올라가고, 계산노드에서 돌고, traceback 전문이 돌아옴
```

`dev run`은 수정된 코드를 rsync로 먼저 올린다. 올리는 건 `vlm_gate/{scripts,run_scripts,analysis}`,
`tools`, `docs` 뿐 — **3.4MB / 509파일, 1초.** venv·데이터·체크포인트·타일은 절대 안 건드린다.

| 상황 | 명령 |
|---|---|
| 짧은 스모크, 즉시 결과 | `dev run '<명령>'` |
| 긴 잡 (라벨링·학습) | `dev bg 'sbatch ...'` → `dev tail` |
| pdb·대화형 프롬프트 | `dev keys 'p chunk.shape'` → `dev cap` |
| 할당 반납 | `dev free` |

## 스모크 테스트는 어떻게 하나

**맥북에서 안 한다.** 맥북은 macOS/arm이고 서버는 우분투/CUDA라 로컬 검증은 의미가 없다.
스모크는 지금과 똑같이 `srun` 할당 안에서 돈다 — 다만 그 할당을 매번 새로
잡는 게 아니라 `dev alloc`으로 잡아둔 걸 재사용한다. 지금보다 오히려 빠르다.

판정 기준도 그대로: 잡 상태가 아니라 산출물 개수(`prediction.txt`의 `^episode` 줄 등).

## 달라지는 것 — 파일시스템을 훑을 수 없다

맥북에는 코드만 있고 데이터가 없다. 에이전트가 `assets/labels/*.parquet`이나
프레임 캐시를 로컬에서 열어볼 수 없다. 데이터 확인은 전부 서버에서 한 줄로:

```bash
tools/dev run 'python -c "import pandas as pd; d=pd.read_parquet(\"assets/labels/robocasa/phase5.parquet\"); print(d.shape, d.p_block.describe())"'
```

그래서 **잡이 끝나면 요약 JSON을 남기도록** 스크립트를 고쳐두는 게 중요하다.
`ls`로 훑어서 상황 파악하던 습관은 못 쓴다.

## 인프라팀 정책과의 정합성

정책이 지목한 안티패턴을 이 구조가 오히려 줄인다.

- 로그인 노드 대량 readdir → rsync 대상이 509파일로 고정. 데이터 디렉터리는 제외.
- 상태 폴링 → `dev run`은 서버가 기다렸다가 한 번에 돌려준다. 왕복 폴링 없음.
- 잡 수천 개 → `dev alloc` 재사용으로 오히려 잡 수가 준다.
- 비밀정보 → API 키는 맥북에 두고, 서버에는 필요한 잡에만 주입.

## 확인이 필요한 것 하나

정책의 "AI 툴 트래픽 차단"이 **Gemini/Claude API 호출까지 막는 것인지** 확인해야 한다.
막힌다면 서버에서 API 라벨링이 불가능해진다. 다만 주력 라벨러는 이미
**cosmos(가중치가 서버 GPU에 로컬로 있음)** 라서 파이프라인 본체는 영향 없다.
API는 검증용 보조 경로였다.
