# Checklist: AVA-VLA
**arXiv**: 2606.15099 · **Verification**: abs+body(674줄) 대조
## ② Full title
"Think Less, Act Early: Reinforced Latent Reasoning with Early Exit in Vision-Language-Action Models"
## 핵심 주장 검증
- ✅ AVA-VLA = "Adaptive Variable Alignment VLA" (본문 80행) — 스카우트 약칭 정확
- ✅ state confidence 기반 Early-Exit로 latent reasoning 조기종료 (§3.6), 6× 추론 가속, LIBERO 98.3%
- ⚠️ 중요: RL 학습 기반 프레임워크 (training-free 아님); early-exit는 자체 latent reasoning 깊이를 게이트 — 액션 압축 게이트가 아님. 직접 이식 불가, '정책 내부 신호' 계열 아이디어로만 차용 가능
