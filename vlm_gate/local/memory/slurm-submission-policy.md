---
name: slurm-submission-policy
description: "rlwrld 클러스터 sbatch 필수 정책 — wckey, MODEL_OUTPUT_DIR($USER 포함), 파티션 규칙"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1a26f530-5afb-4579-9ceb-a2bdbf3522ac
  modified: 2026-08-03T18:41:36.671Z
---

rlwrld 서버 sbatch 정책 (2026-08 사용자 공지 기준):

1. **wckey 필수**: `#SBATCH --wckey=project-short-name:sub_fast` (또는 `:others`). 우리는 `sub_fast` 사용.
2. **MODEL_OUTPUT_DIR 필수**: 제출 시 `--export=ALL,MODEL_OUTPUT_DIR=...` 명시. 경로는 반드시 `$USER` 포함 — Kakao: `/rlwrld-unified-checkpoints/$USER/...`, AWS(SKT): `/fsx/rlwrld-unified-checkpoints/$USER/...`. 스크립트 내부에서 `: "${MODEL_OUTPUT_DIR:?...}"` 가드 후 `--output-dir "$MODEL_OUTPUT_DIR"`.
3. **파티션**: 학습 잡은 `sjw_alinlab` 파티션(사용자 지시), eval류는 background. 불량 노드 exclude: worker-node100,1,104,3 (수시 갱신).
4. job-name은 길고 설명적으로(≈50자+).

**Why:** 정책 위반 시 제출 거부/사일런트 실패. 실제로 MODEL_OUTPUT_DIR 미지정 시 sbatch가 에러로 거부됨.
**How to apply:** 모든 sbatch 제출 전 이 세 요소(wckey/MODEL_OUTPUT_DIR/파티션) 확인. 서브에이전트에 제출 지시할 때도 이 정책을 프롬프트에 명시. 관련: [[ops-download-and-bg-task-lessons]]

**srun 옵션 제한 (사용자 지시)**: srun 할당 시 `--time`, `-p`(파티션 지정) 등 시스템 config성 옵션 금지 — 시간 한도는 시스템 기본(debug 3h)에 맡긴다. 허용: `--gpus`, `--job-name`, `--wckey`, (불량노드 `--exclude`).
