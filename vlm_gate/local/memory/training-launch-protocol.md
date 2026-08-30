---
name: training-launch-protocol
description: 학습 잡 제출 절차 — srun 할당으로 의도대로 도는지 확인한 뒤에만 본학습 sbatch
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1a26f530-5afb-4579-9ceb-a2bdbf3522ac
  modified: 2026-08-07T07:43:05.184Z
---

학습(train) 잡은 반드시: **① srun으로 GPU 할당받아 짧게(스모크) 돌려 의도대로 동작하는지 직접 확인 → ② 통과 시에만 본학습을 sbatch 제출 → ③ 제출 후 초반 헬스체크(2분 생존 + 30분 시점 loss 정상) 모니터링.**

**Why:** 사용자 명시 규칙. sbatch 직행으로 66894(torchrun PATH 버그, 51초 즉사)류 사고 발생. 스모크는 환경 문제(PATH·로드·shape)와 학습 신호(loss 하강, 출력 비상수)를 모두 잡는다.
**How to apply:** 내가 직접 제출하든 서브에이전트에 위임하든 이 3단계를 프롬프트/절차에 명시. eval 잡은 예외적으로 sbatch 직행 가능하나 새 배선이면 1태스크 srun 스모크 권장. 관련: [[slurm-submission-policy]]
