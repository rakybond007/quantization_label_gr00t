---
name: local-agent-via-tmux
description: 2026-08-31부터 에이전트는 맥북에서 실행 — 서버의 영속 tmux 셸을 ssh로 원격 조종하며 작업은 100% 서버에서 돈다
metadata:
  type: project
---

인프라팀 정책으로 **2026-08-31(월)부터 서버 내 코딩 에이전트·원격 IDE 사용 금지**. 위반 시 7일 접근 정지(로컬에서의 우회 남용도 동일). 에이전트는 사용자의 **맥북**으로 내려가고, 연산은 전부 서버에 남는다.

**구조:** 맥북 Claude Code ──ssh──> 서버 tmux 세션 `dev` ──srun──> 계산노드. 도구는 `tools/dev` 하나. `dev alloc`으로 srun 할당을 tmux 셸에 잡아두면 이후 `dev run`이 전부 그 계산노드에서 돌아 **대기열을 하루 한 번만** 선다. 문서: `docs/LOCAL_WORKFLOW.md`.

**Why:** 처음에 `srun --jobid=` 재접속이나 로컬 스모크 스크립트를 제안했다가 사용자가 강하게 반려했다 — 전자는 로컬에서 쓸 수 없는 명령이고, 후자는 **맥북(macOS/arm)과 서버(우분투/CUDA)의 환경이 달라 로컬 검증이 무의미**하기 때문이다. tmux 원격 조종은 사용자가 직접 제시한 방향이고, 셸이 영속하므로 `cd`/`export`/할당이 명령 사이에 유지된다는 점이 결정적이다. 2026-08-26 서버에서 send-keys+base64 eval+rc파일 방식으로 상태 유지·따옴표·종료코드 전달을 4개 케이스로 검증했다.

**How to apply:**
① **맥북에서는 파이썬을 돌리지 않는다.** 로컬 스모크·환경 재현 시도 금지. 검증은 `dev run`으로 서버에서.
② 맥북에는 **코드만** 있다(`vlm_gate/{scripts,run_scripts,analysis}`, `tools`, `docs` = 3.4MB/509파일). 데이터·venv·체크포인트·타일은 서버 전용이라 로컬 Read/Grep으로 못 연다 — 데이터 확인은 `dev run 'python -c ...'`.
③ `ls`로 훑어 상황 파악하는 습관을 못 쓰므로 **잡이 끝날 때 요약 JSON을 남기도록** 스크립트를 미리 고쳐둔다.
④ 스모크→sbatch 원칙은 그대로 유지([[smoke-before-submit]]). 할당만 재사용될 뿐 절차는 동일.
⑤ 미확인: 정책의 "AI 트래픽 차단"이 Gemini/Claude **API 호출**까지 포함하는지. 포함되면 API 라벨링 불가 — 다만 주력은 서버 GPU 로컬 가중치인 cosmos라 본체는 무영향.

관련: [[nfs-load-discipline]], [[smoke-before-submit]], [[monitoring-cadence]]
