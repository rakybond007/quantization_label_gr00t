---
name: monitoring-cadence
description: "백그라운드 작업 모니터링 주기 규칙 — 작업량 예측해서 긴 작업은 긴 주기로, standby 표시 최소화"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 1a26f530-5afb-4579-9ceb-a2bdbf3522ac
  modified: 2026-08-13T04:18:43.505Z
---

백그라운드 에이전트/작업은 써도 되지만, 사용자 터미널에 "Standby for monitor events" 같은 대기 표시가 반복 노출되는 것을 싫어함.

**Why:** 짧은 주기 폴링·빈번한 중간 확인이 터미널을 어지럽히고 답답하게 만듦.
**How to apply:** ① 작업 시작 시 소요시간을 예측하고, 모니터링 주기를 그 길이에 비례해 설정(수 시간짜리는 완료 예상 시점 근처에서만 확인). ② 세션이 붙잡는 대기 루프/짧은 sleep 폴링 금지 — 긴 작업은 sbatch나 setsid 분리 실행. ③ 중간 상태 확인은 사용자가 물을 때 또는 예측 완료 시점에만. 관련: [[training-launch-protocol]]
