# Checklist: RTC (Real-Time Chunking)
**arXiv**: 2506.07339 · **Verification**: abs+body(819줄) 대조
## ② Full title
"Real-Time Execution of Action Chunking Flow Policies" (스카우트 통칭 'Real-Time Chunking' = 본문 자칭 RTC ✅)
## 핵심 주장 검증
- ✅ 청크 경계를 inpainting 문제로: 실행 확정분 freeze + 나머지 inpaint (§3.1), soft masking으로 cross-chunk 연속성 (§3.2)
- ✅ 재학습 불필요, 임의 diffusion/flow VLA 적용 가능 (본문 31행)
- ✅ 동기: 청크 전환 불연속/모드 점프가 분포이탈 유발 (본문 43행)
## 판정: 진위 ✅. 단 우리 적용성은 별개 (본문 판단 참조)
