# MoE4 Z-score Target Normalization + Critic + Length Cost

목적: MoE4 (main 16 + m8 16→8 + m4 8→4 + n8 raw 8) 라우터가 "샘플별로" 적절한 expert를 고를 수 있도록 — 절대 loss 스케일 차이(systemic bias)를 자동 제거하고, 그 위에 해석 가능한 길이 trade-off를 얹는다.

---

## 배경

### Compaware의 실패 모드
기존 `moe_compression_weight` (compaware):

```
total += -λ_c · Σ p_i · norm(1/horizon_i)
```

- 모든 샘플에 동일한 길이 보너스 → quality 무시.
- 실측 (MoE3, λ_c=0.1): m4 pick = 91~93%, 성공률 ~16%p 하락.

### 단순 length cost의 한계
초기 시도 (γ·h/H_max를 per-expert loss에 가산):

- robocasa smoke 측정 → `loss_main=1.15, loss_m8=0.96, loss_m4=1.21, loss_n8=0.96`
- libero smoke 측정 → `loss_main=2.01, loss_m8=1.90, loss_m4=1.61, loss_n8=1.32`
- 데이터셋에 따라 **expert별 loss 스케일이 완전히 다름**. m4가 robocasa에선 가장 높고 libero에선 가장 낮음.
- 즉 라우터 supervise target = `softmax(-loss/τ)`은 데이터셋마다 다른 systemic 편향을 반환. γ로 누르려면 데이터셋마다 γ를 따로 측정해 정해야 함 — 추측 학습.

### 핵심 통찰
"이 expert가 잘 했나?"는 절대 loss로 답할 수 없음. **이 expert가 본인 평균 대비 잘 했나?**가 신호여야 함.

---

## 방법: Per-Expert Z-score Normalization

### Per-expert running 통계 (EMA)
학습 step마다 raw `losses_b4`에서 per-expert 평균/분산을 EMA로 업데이트:

```python
cur_mean = losses_b4.detach().mean(dim=0)       # (K,)
cur_var  = losses_b4.detach().var(dim=0)
running_mean ← (1-α)·running_mean + α·cur_mean
running_var  ← (1-α)·running_var  + α·cur_var
```

- `α = 0.01` (EMA decay; 대략 100 step의 효과적 window)
- 첫 step은 직접 init (0/1 init bias 회피)

### Z-score target
```
basis = critic_pred  (critic ON) or losses_b4  (critic OFF)
z_i = (basis_i − running_mean_i) / sqrt(running_var_i + ε)
target_signal = z + γ · h / H_max     ← σ 단위 길이 패널티
target_dist = softmax(− target_signal / τ_target)
KL(router || target_dist) ← supervise loss
```

**해석:**
- `z_i < 0` → expert i가 이번 샘플에서 **본인 평균보다 잘함** → 라우터 prob ↑
- `z_i > 0` → 본인 평균보다 못함 → prob ↓
- expert별 절대 스케일 차이는 자동 상쇄 (수학적 사실: `E[z_i] = 0` for all i)
- γ가 z-score 위에 얹히므로 **σ 단위로 해석 가능**:
  - `γ = 0.3` → "0.3σ 품질 손실까지는 짧은 expert 선호"
  - 데이터셋의 절대 loss 범위 추측할 필요 없음

### Soft-mixture는 raw losses 사용
정규화는 **라우터 supervise target에만 적용**. soft-mixture는 raw `losses_b4`로 expert들이 순수 학습 신호를 받음:

```
soft_mixture = (router_probs · losses_b4).sum(-1).mean()
```

이유: 각 expert가 자기 prediction 정확도를 절대 단위로 학습해야 함. 정규화된 신호로 학습시키면 expert가 "본인 평균을 낮추는" 쪽으로 갈 위험.

### Critic은 선택적
Critic head (`moe_critic_weight > 0`):
- per-sample loss 예측: `critic_pred = MLP(router_in)` → (B, K)
- supervised regression: `MSE(critic_pred, losses_b4.detach())`
- target signal로는 `critic_pred`를 z-score (라벨 노이즈 감소 + task-conditional)

critic OFF면 raw `losses_b4`를 직접 z-score. critic ON이 더 안정적이지만 동일한 정규화 효과.

---

## Total loss (단순 single_pick + 정규화 ON 경우)

```
total = soft_mixture                                  # raw expert training signal
      + (warmup · λ_balance) · balance                 # mean prob → uniform
      + (warmup · λ_supervise) · KL(router, target)    # z-score 기반 target
      + λ_critic · critic_loss                         # critic regression
```

- `compaware`는 0 (이번 실험에선 사용 안 함). 정규화 + γ가 그 역할 대체.

---

## Inference 호환성

정규화는 **학습 중 라우터 supervise 신호**에만 작용. 추론:
- `head_router(router_in)` → softmax → expert pick. 기존 코드 그대로.
- critic / running stats는 사용 안 함.
- 기존 eval client / per_quad_mask / single_pick 추론 코드 100% 호환.

---

## 학습 제출 (현재)

| Job | Target | flags |
|-----|--------|-------|
| 315158 | robocasa | `--moe-target-normalize --moe-target-normalize-ema 0.01 --moe-length-cost-weight 0.3 --moe-critic-weight 0.3 --moe-critic-hidden 256` |

설정: MoE4 (main+m8+m4+n8), bs64, 60K steps, per_expert_h, single_pick.

Checkpoint:
- `ckpt/robocasa/groot/groot_n1_5_bs64_moe4_per_expert_body_zscore_critic_lc_0p3/`

---

## Eval 후 비교 포인트

1. **Per-expert pick 분포** (50-ep 평균 router probs):
   - baseline MoE4 (314547 per_quad_mask, compaware 없음) 대비 router 분산 (samples 간 차이) 어떻게 다른지
   - 각 expert의 pick 비율이 데이터셋의 task 특성과 합리적으로 연관되는지
2. **성공률** vs canonical baseline (50-ep cap, succ-only step):
   - mh_m8, moe4_per_expert, compaware(λ_c=0.1, 314685), z-score+critic+lc(315158) 네 모델 비교
3. **압축률 (succ-only step)**:
   - baseline 334.8 step 대비 감소량
   - 이번 모델의 short expert 비율과 step 감소량의 상관

---

## 한계 / 미해결

1. **다양성 보장 X**: z-score 정규화는 systemic bias만 제거. 만약 main이 거의 모든 샘플에서 본인 평균보다 잘하면 여전히 main 위주 분포. 진정한 multi-expert specialization은 데이터/expert 구조가 자체적으로 만들어내야 함.
2. **γ 자체는 여전히 하이퍼파라미터**: 값은 σ 단위로 해석 가능해졌지만, "0.3σ가 적정인가"는 실험으로 검증해야. γ=0.3, 0.5 정도부터 시도.
3. **per_quad_mask 비지원**: 정규화는 single_pick 라우팅 모드에서만 활성화. per_quad_mask로 확장하려면 stats를 (Q, C) 또는 (C,) 단위로 분리 필요.

---

## 코드 변경 위치

- `gr00t/model/action_head/flow_matching_action_head.py`:
  - `FlowMatchingActionHeadConfig`: `moe_target_normalize`, `moe_target_normalize_ema` 필드 추가
  - `__init__`: attribute init + per-expert running stats buffers (`_moe_loss_running_mean`, `_moe_loss_running_var`, `_moe_norm_inited`) — single_pick + normalize ON일 때만
  - `_moe_forward`: training 시 EMA stats 갱신, target_signal에 z-score 적용, γ는 σ 단위 가산
- `scripts/gr00t_finetune.py`: 두 신규 args + config persist + 로그 출력
- `run_scripts/train/robocasa/finetune_gr00t_n1_5_robocasa_moe4_per_expert_body_zscore_critic_lc_0p3.sh`: 학습 sbatch
