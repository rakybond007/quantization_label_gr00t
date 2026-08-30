---
name: smoke-before-submit
description: 모든 학습·eval은 srun 스모크로 산출물이 실제로 나오는지 확인한 뒤에만 sbatch 본실행
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1a26f530-5afb-4579-9ceb-a2bdbf3522ac
  modified: 2026-08-26T03:50:06.013Z
---

학습이든 eval이든 **sbatch 본실행 전에 srun으로 스모크를 먼저 돌리고, 산출물이 실제로 생성되는 것을 눈으로 확인한 뒤에만** 본실행을 제출한다. 예외 없음.

**Why:** 잡이 `COMPLETED`로 끝나도 산출물이 0인 경우가 반복됐다 — eval이 8/17~8/26 사이 여러 번, 매번 다른 원인으로(numpy 1.26 vs robocasa 1.23 요구, transformers 5.15 업그레이드로 gr00t 임포트 불가, task embedding npz의 numpy2 pickle, HF 캐시의 paligemma `processor_config.json` 누락) 조용히 죽었다. 스모크 없이 τ 스윕 3개(각 8-array)를 제출했다가 며칠을 날렸다. 사용자 지적: "제출하기 전에 smoke로 돌아가고 롤아웃까지 뽑히는지 확인을 해야 맞을거 같은데?"

**How to apply:**
① **스모크 판정 기준은 잡 상태가 아니라 산출물 개수**다. eval은 `prediction.txt`에 `^episode` 줄이 실제로 쌓이는지, 학습은 `[train] epoch 1/N` 로그가 나오는지 확인한다.
② 스모크는 **srun**으로 — 즉시 할당받아 짧게. 옵션은 `--gpus`/`--job-name`/`--wckey`/`--exclude`만 (`--time`, `-p`는 시스템 기본값이 붙으므로 지정 금지). 규모는 eval이면 1태스크 × 1~2에피소드, 학습이면 소수 에피소드 3스텝.
③ 통과하면 그때 sbatch. 학습은 `sjw_alinlab`, eval·라벨링은 `background` (파티션 용도 구분: [[slurm-submission-policy]]).
④ 장시간 잡에는 감시를 붙이되 **PENDING을 정지로 오판하지 말 것** — RUNNING 상태에서만 정지 카운터를 올린다.

관련: [[training-launch-protocol]], [[monitoring-cadence]]
