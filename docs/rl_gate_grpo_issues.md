# GRPO Gate Training — 설계 문제 및 수정 사항

> 작성일: 2026-04-23  
> 현상: merge_rate가 학습 내내 초기값에서 전혀 변하지 않음 (v1: 12%, v3: 62% 고정)

---

## 실험 결과 요약

| Job | Config | iter | avg success | merge | 상태 |
|-----|--------|------|-------------|-------|------|
| 292131 v1 | K=8 sequential, init_logit=-2.0 | ~118/300 | ~0.40 | 10–15% | loss≈0, 불변 |
| 292145 v2 | K=8, alpha_len=1.0 | ~113/300 | ~0.45 | 8–10% | KL 약간, 불변 |
| 292467 v3 | K=8, init_logit=+0.5 | ~71/300 | ~0.50 | 60–65% | loss≈0, 불변 |
| 292566 PnP-v3-vec | K=32, n_envs=8, init_logit=+0.5 | ~9/300 | 0/32 매번 | 62% | PnP 완전 실패 |

- TurnOnStove v3: **merge 60%에서도 성공률 유지** (iter 59: success=0.88, steps=342) — 검증 가치 있음
- PnPMicrowaveToCounter: merge 62%는 contact-rich task에서 즉각 0% → init_logit 낮춰야 함

---

## 버그 1: `nn.init.zeros_(self.net[-1].weight)` — Hidden Layer Gradient 차단

**파일**: [gr00t/rl/gate.py](../gr00t/rl/gate.py) — `GateMLP.__init__`

```python
# 현재 (문제)
nn.init.zeros_(self.net[-1].weight)
nn.init.constant_(self.net[-1].bias, init_logit)
```

**원인 분석**:

```
forward:  logit = W4 @ h3 + b4
          W4=0이므로  logit = b4  (입력과 무관한 상수)

backward: d(loss)/d(h3) = W4^T @ d(loss)/d(logit) = 0
```

W4=0이면 이전 레이어(h1, h2, h3)로의 gradient가 완전히 0이 됨.  
결과: MLP 전체가 **스칼라 logit(b4) 하나**로만 동작. VL feature는 학습에 기여 없음.

**수정**:
```python
# 수정 후
nn.init.normal_(self.net[-1].weight, std=0.01)   # 거의 0이지만 정확히 0 아님
nn.init.constant_(self.net[-1].bias, init_logit)
```

---

## 버그 2: 에피소드 내 Gradient 상쇄 (Intra-episode cancellation)

**파일**: [gr00t/rl/grpo_trainer.py](../gr00t/rl/grpo_trainer.py) — `grpo_loss()`

**원인 분석**:

한 에피소드의 N개 chunk에 동일한 advantage A가 적용됨.  
각 chunk에서 logit에 대한 gradient:

```
d=1 (merge)    : gradient = -A × (1 - p)   → logit 올리는 방향 (A>0이면)
d=0 (no-merge) : gradient = +A × p         → logit 내리는 방향 (A>0이면)

에피소드 전체 gradient 합 = -A × N × (actual_merge_rate - p)
```

Bernoulli(p) 샘플링의 기댓값이 p이므로 `actual_merge_rate ≈ p` (chunk가 많을수록 더 정확).  
→ **gradient ≈ 0** (수학적으로 소멸)

**구체적 수치** (v3: p=0.62, N≈50 chunks/ep):
- std(actual_rate) = sqrt(0.62×0.38/50) ≈ 0.069
- 실제 gradient 크기 ≈ |A| × 0.069 → 표시 포맷 `{:.3f}`에서 ±0.000으로 반올림

---

## 설계 문제: 같은 Seed K개 + KL Anchor

**현재**: `seeds = [args.seed] * args.group_size` — 모든 K rollout이 동일한 환경에서 시작

- 같은 seed → 같은 VL feature → reward 차이가 gate noise에서만 발생
- 위 버그 2와 합쳐져 advantage-merge_rate 상관관계가 매우 약해짐
- `kl_weight`가 남은 tiny gradient를 prior 방향으로 상쇄 → logit 완전 고정

**참고**: GRPO 원본(LLM)은 같은 프롬프트에서 K개 다른 응답을 생성.  
로봇 RL에서 "같은 seed = 같은 프롬프트"는 성립하지만, 수백 개의 i.i.d. Bernoulli 결정이 있어  
LLM의 "K개 서로 다른 응답" 구조와 다름. 환경 다양성을 통한 reward variance가 필요.

---

## 수정 사항 목록

### 1. GateMLP 초기화 수정 (gate.py)

```python
# AS-IS
nn.init.zeros_(self.net[-1].weight)

# TO-BE
nn.init.normal_(self.net[-1].weight, std=0.01)
```

### 2. K Rollout에 다른 Seed 사용 (train_gate_rl.py)

```python
# AS-IS
seeds = [args.seed] * args.group_size

# TO-BE — iter마다 다른 시드 세트 사용
seeds = [args.seed + it * args.group_size + k for k in range(args.group_size)]
```

GRPO의 "같은 프롬프트" 원칙은 약해지지만, reward variance 확보가 현실적으로 더 중요.  
또는 일부는 같은 seed, 일부는 다른 seed로 절충하는 방식도 가능.

### 3. KL Weight 대폭 축소 또는 제거

```bash
# AS-IS: kl_weight=0.02
# TO-BE: kl_weight=0 or 0.001
--kl-weight 0.0
```

### 4. Learning Rate 상향

```bash
# AS-IS: lr=3e-4
# TO-BE: lr=1e-3
--lr 1e-3
```

### 5. PnPMicrowaveToCounter init_logit 조정

merge 62%는 contact-rich PnP task에서 바로 0% 성공률.  
`init_logit=-1.0` (sigmoid≈0.27, merge 27% 정도)부터 시작.

---

## 검토할 대안 접근법

### 대안 A: 에피소드 수준이 아닌 Chunk 수준 Credit

현재 모든 chunk에 episode reward를 동일하게 부여.  
chunk별로 "이 merge 결정 이후 task progress가 얼마나 됐는가" 형태의 dense reward를 설계하면  
gradient 상쇄 문제 근본 해결 가능. 단, reward shaping 설계가 필요.

### 대안 B: Scalar Gate만 먼저 학습

MLP를 쓰지 않고 single scalar logit만 학습해서 "최적 global merge rate"를 먼저 찾음.  
이후 scalar logit을 초기값으로 하는 context-sensitive MLP를 fine-tune.

### 대안 C: Episode-level Decision (1 decision per episode)

에피소드 시작 시 한 번만 merge 여부를 결정 (항상 merge or 항상 no-merge).  
→ N chunks × advantage cancellation 문제 없음. gradient 크고 명확.  
단, 동적 per-chunk 결정 불가 → 표현력 제한.

---

## 관련 파일

```
gr00t/rl/gate.py              — GateMLP 정의
gr00t/rl/grpo_trainer.py      — grpo_loss(), group_advantages(), episode_reward()
gr00t/rl/episode_runner.py    — 단일 env 롤아웃
gr00t/rl/vec_episode_runner.py — AsyncVectorEnv 병렬 롤아웃
scripts/train_gate_rl.py      — 메인 학습 스크립트
run_scripts/train/rl/         — sbatch 제출 스크립트들
```

## 실험 체크포인트 저장 위치

```
ckpt/rl/gate_baseline_TurnOnStove/         — v1 (iter 10, 20, ..., 110 저장됨)
ckpt/rl/gate_baseline_TurnOnStove_v2/      — v2
ckpt/rl/gate_baseline_TurnOnStove_v3/      — v3 (iter 70 저장됨)
ckpt/rl/gate_baseline_PnPMicrowaveToCounter_v3_vec/ — PnP (iter 10 저장됨)
```
