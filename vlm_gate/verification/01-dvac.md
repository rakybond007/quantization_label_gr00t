# Checklist: DVAC (Denoising-Variance Adaptive Chunking)
**arXiv**: 2606.03847 · **Verification**: abs+body(909줄) 대조, 핵심수치 본문확인
## ② Full title
"Denoising Tells When to Replan: Denoising-Variance Adaptive Chunking for Flow-Based Robot Policies"
(스카우트 표기는 부제 생략 ⚠️ — 내용 일치)
## 핵심 주장 검증
- ✅ "LIBERO 94.75%→98.00%, replanning −43.0%" — 본문 79행·107행 문자 그대로 일치 (π0.5 기반 정책)
- ✅ training-free, 최종 denoising 추정치들의 분산으로 chunk의 '안정 prefix' 식별
- ✅ 임계값은 rolling local variance 분포 상대값 (태스크별 튜닝 불필요)
- ⚠️ 프레이밍 주의: DVAC은 '리플랜 시점/실행 길이' 적응이지 압축(K) 게이트가 아님 — 우리에겐 "무학습 내부신호 게이트"라는 경쟁 베이스라인으로 이식하는 것 (스카우트 제안 그대로)
## 7-pattern: 수치복사✅ 차원날조✅ 베이스라인명✅(LIBERO/RoboTwin/CALVIN 본문 존재) 제목⚠️(부제생략) 프레이밍⚠️(위) 한정어✅(43.0% 정확) 용례✅
