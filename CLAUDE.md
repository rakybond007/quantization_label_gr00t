# 이 저장소에서 일하는 방법 (맥북 로컬 에이전트용)

너는 **사용자의 맥북**에서 돌고 있다. 연산은 전부 **원격 우분투 서버**에서 일어난다.
2026-08-31부터 서버 안에서 코딩 에이전트를 돌리는 것이 금지됐다.

## 절대 규칙

1. **맥북에서 파이썬·학습·평가를 돌리지 않는다.** 맥북은 macOS/arm, 서버는 우분투/CUDA다.
   로컬 재현·로컬 스모크는 의미가 없다. `pip install`, `conda`, `python foo.py` 전부 금지.
   맥북에서 하는 일은 **파일 편집과 `git`뿐이다.**
2. **서버에서 뭘 하려면 `tools/dev`를 쓴다.** 그 외의 방법으로 서버를 건드리지 않는다.
3. **데이터는 로컬에 없다.** `assets/`, `vlm_gate/output/`, 프레임 캐시, 체크포인트는
   서버에만 있다. Read/Grep/ls로 열려고 하지 말 것 — 없다. 확인은 `dev run 'python -c ...'`.
4. **학습·평가는 스모크 먼저.** `srun` 스모크로 산출물이 실제로 나오는 걸 확인한 뒤에만
   `sbatch`. 판정 기준은 잡 상태가 아니라 **산출물 개수**다. 예외 없음.
5. **공유 마운트를 재귀 탐색하지 않는다.** `find`/`du -sh`/`grep -r`를 깊이 제한 없이 걸지 말 것.
   26만 파일 타일 디렉터리는 매니페스트로 접근한다. 인프라팀이 이런 부하를 추적하고 있다.

## 기본 루프

```bash
tools/dev up                  # 하루에 한 번
tools/dev alloc --gpus=1      # 대기열은 이때 한 번만 선다
tools/dev run 'python vlm_gate/scripts/foo.py --limit 4'   # 이후 전부 그 GPU 노드 안에서
```

`dev run`은 바뀐 코드를 먼저 올리고(코드만 ~3MB), 서버에서 돌리고, **출력 전문과 종료코드**를
돌려준다. traceback이 그대로 온다. tmux 셸이 살아있어서 `cd`/`export`/`conda activate`/
**srun 할당**이 명령 사이에 유지된다 — 매번 대기열을 서지 않는다.

| 상황 | 명령 |
|---|---|
| 짧은 스모크 | `dev run '<명령>'` |
| 긴 잡 (라벨링·학습·평가) | `dev bg 'sbatch ...'` → `dev tail` |
| pdb·대화형 | `dev keys 'p x.shape'` → `dev cap` |
| 지금 어느 노드? | `dev where` |

## 상황 파악은 `ls`가 아니라 준비된 한 줄로

파일시스템을 훑을 수 없으므로 `vlm_gate/local/CHEATSHEET.md`의 명령을 쓴다.
새 스크립트를 만들 때는 **끝날 때 요약 JSON을 남기도록** 짠다. 그래야 다음에 확인할 수 있다.

## 참고 문서

- `vlm_gate/local/SETUP.md` — 최초 1회 세팅 (사용자가 직접 할 일 포함)
- `vlm_gate/local/PATHS.md` — 서버 경로·conda 환경·slurm 정책 지도
- `vlm_gate/local/CHEATSHEET.md` — 상태 확인용 한 줄 명령 모음
- `vlm_gate/local/STATE.md` — 현재 연구 진행 상황
- `vlm_gate/local/memory/` — 이전 세션에서 쌓인 교훈 (읽고 시작할 것)

## 커밋

맥북의 클론이 코드의 원본이다. 의미 있는 진전마다 커밋하고 `quant` 리모트에 푸시한다.
`dev`의 rsync는 이터레이션용 임시 전송일 뿐 이력이 아니다.
