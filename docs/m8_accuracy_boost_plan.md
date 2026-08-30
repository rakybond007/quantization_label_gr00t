# m8 head 정확도 증대 실험 계획

작성일: 2026-04-25
배경: mh_m8 + ens→8 = 0.622, mh_m8+econsist + ens→8 = 0.617. m8 head 단독 (293603 부분결과) = **0.657** (이미 64%↑). 여기서 더 끌어올려 0.70+ 목표.

---

## A. 추가 학습 없이 (now)

### A1. m8 head 단독 (이미 진행 중) ★
- Job: 293603 (mh_m8), 293604 (mh_m8+econsist)
- 진행: 79%
- 현재까지: mh_m8 m8-only = **0.657**, mh_m8+econsist m8-only = 0.622

### A2. Denoising step sweep (이미 제출)
- Jobs: 293658 (mh_m8 ens→8 d8), 293659 (mh_m8 ens→8 d16), 293660 (mh_m8 m8 d8), 293661 (mh_m8 m8 d16)
- 가설: 4-step denoising은 너무 공격적. 8/16에서 +2~5% 가능
- 비교: 동일 모델/head를 denoising=4 vs 8 vs 16

---

## B. 추가 학습 (호환성 유지)

### B1. Method 1 — Ensemble→m8 distillation
**아이디어**: m8 head가 v_f2 (이미 well-trained aux head) 출력을 frozen target으로 학습. 단방향, stop_grad.

**구현**:
- 코드: [`flow_matching_action_head.py`](../gr00t/model/action_head/flow_matching_action_head.py) — config flag `m8_distill_from_f2_weight`, forward에 새 loss term 추가
- 호환성: 기본값 0 → off. 켜면 추가 loss term만 더해짐. checkpoint 구조 동일.

**Loss 추가**:
```
loss += m8_distill_from_f2_weight * MSE(v_m8, v_f2[:8].detach())
```

**Fine-tune 스크립트**: [`run_scripts/train/robocasa/finetune_mh_m8_distill.sh`](../run_scripts/train/robocasa/finetune_mh_m8_distill.sh)
- Source: `mh_m8_econsist/checkpoint-60000` (better starting point)
- 10k steps, save 2k, max keep 3
- aux_warmup=0, consist_warmup=0 (이미 학습된 모델)
- 출력: `mh_m8_econsist_discfix_distill/checkpoint-10000`

**Eval**: [`run_scripts/eval/eval_mh_m8_distill_head_m8.sh`](../run_scripts/eval/eval_mh_m8_distill_head_m8.sh) — head=m8

### B2. Method 2 — m8 refinement decoder
**아이디어**: m8 출력을 받아 작은 MLP가 residual delta 생성. `final = m8_pred + refine(vl_pooled, m8_pred)`.

**구현**:
- 코드: 동일 파일 — config flag `use_m8_refinement` + 새 모듈 `m8_refinement` (3-layer MLP, hidden=512)
- 추론: head=`m8_refined` 신규 옵션 (head=`m8`은 기존 그대로 유지 → 호환성)
- 학습: refine loss는 perturbed-GT proxy 사용 (cheap surrogate). 정확한 refine을 원하면 inference-time m8 출력으로 교체 필요 (TODO)

**Loss 추가**:
```
a_proxy = clean_m8 + 0.05 * randn_like(clean_m8)  # GT에 작은 noise
delta = m8_refinement([vl_pooled, a_proxy])
loss += m8_refinement_weight * MSE(a_proxy + delta, clean_m8)
```

**모듈 크기**: `Linear(2048+8*32 → 512) + GELU + Linear(512→512) + GELU + Linear(512→256)` ≈ 1.4M 파라미터, 마지막 layer zero-init (refine 시작 시 identity)

**Fine-tune 스크립트**: [`run_scripts/train/robocasa/finetune_mh_m8_refine.sh`](../run_scripts/train/robocasa/finetune_mh_m8_refine.sh)

**Eval**: [`run_scripts/eval/eval_mh_m8_refine_head.sh`](../run_scripts/eval/eval_mh_m8_refine_head.sh) — head=m8_refined

---

## 실행 순서

1. ✅ A1 (293603/4) → 결과 곧 (현재 79%)
2. ✅ A2 (293658-61) → 디노이징 sweep 결과
3. **B1+B2 fine-tune 동시 제출** (각 10k step, 2GPU)
4. fine-tune 끝나는 즉시 해당 eval 스크립트 제출
5. 최종 모든 변형 비교 표 작성

## 비교 변형 매트릭스

| Variant | Model | Head | Denoising | Eval Job |
|---------|-------|------|----------:|----------|
| baseline | base | main | 4 | (기존) |
| baseline_merged | base | main + 16→8 | 4 | (기존) |
| mh_m8 + ens→8 | mh_m8 | ens_fix + 16→8 | 4 | 292742 |
| mh_m8 + ens→8 | mh_m8 | ens_fix + 16→8 | **8** | 293659 |
| mh_m8 + ens→8 | mh_m8 | ens_fix + 16→8 | **16** | 293660 (오타: ens_fix d16은 293660, m8 d8=...) |
| mh_m8 + m8 | mh_m8 | m8 | 4 | 293603 |
| mh_m8 + m8 | mh_m8 | m8 | **8** | 293661 (확인) |
| mh_m8 + m8 | mh_m8 | m8 | **16** | 293658 (확인) |
| mh_m8+econsist + m8 | mh_m8+econsist | m8 | 4 | 293604 |
| mh_m8 + distill + m8 | distilled | m8 | 4 | TBD |
| mh_m8 + refine + m8_refined | refine | m8_refined | 4 | TBD |

(jobid 매핑은 squeue 결과로 재확인 필요)
