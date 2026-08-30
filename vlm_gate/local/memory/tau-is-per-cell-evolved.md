---
name: tau-is-per-cell-evolved
description: "게이트 임계값 τ는 (티처×아키텍처)마다 다른 탐색 대상 — evolve 루프의 일부로 관리, 홀드아웃 분리 필수"
metadata: 
  node_type: memory
  type: project
  originSessionId: 1a26f530-5afb-4579-9ceb-a2bdbf3522ac
  modified: 2026-08-16T15:54:36.115Z
---

게이트 임계값 τ=0.5는 중립값이 아니라 임의 선택이다. judge마다 YES/NO 토큰 확률 캘리브레이션이 달라 같은 0.5가 완전히 다른 동작점을 의미한다 — cosmos 학생은 conf 최대 0.726(qrate@0.5=0.19), gemma 학생은 최대 0.999(qrate@0.5=0.53).

**Why:** 이 차이를 모르고 τ₃=0.8을 이식했다가 K3 사다리 실험 전체가 무효가 됐다(K3가 한 번도 발동 안 함). 또 cosmos가 gemma보다 스텝이 40 느린 원인이 판별력이 아니라 동작점일 가능성이 큼.

**How to apply:** ① τ는 (도메인×티처×아키텍처) 셀마다 별도 탐색·기록 — analysis/EXPERIMENT_MATRIX.md 표에 유지. ② τ 탐색을 evolve 루프의 일부로 보되(사용자 승인), 튜닝 태스크와 홀드아웃 태스크를 분리해 테스트셋 튜닝을 피할 것. ③ 임계값을 프롬프트 문장으로 옮기는 것은 동등 변환이 아님 — 순위 자체가 재형성되므로 별도 검증 필요. 관련: [[evolver-composite-gating]]
