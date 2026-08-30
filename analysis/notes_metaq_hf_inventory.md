# metaq HF cache inventory (snapshot 2026-05-25)

전체 metaq 관련 HF 캐시 디렉토리 — 삭제 결정 전 기록.

위치: `$HOME/.cache/huggingface/hub/`

## 명시적 v1 (보존 권장)
- `models--prehj--gr00t-n1.5-robocasa-metaq-n16-v1` (17G)
- `models--prehj--gr00t-n1.5-robocasa-metaq-n32-v1` (17G)

## 명시적 v2/v3 (삭제 안전)
- `models--prehj--gr00t-n1.5-robocasa-metaq-n16-v2` (17G)
- `models--prehj--gr00t-n1.5-robocasa-metaq-n32-v2` (17G)
- `models--prehj--gr00t-n1.5-robocasa-metaq-v2-n8-b-only` (17G)
- `models--prehj--GR00T-N1.5-robocasa-metaq-v3a-n32-K4-b-d-lc-0p10-60k` (17G)
- `models--prehj--GR00T-N1.5-robocasa-metaq-v3a-n32-K4-b-only-60k` (17G)
- `models--prehj--GR00T-N1.5-robocasa-metaq-v3b-n32-K6-b-d-lc-0p10-60k` (17G)
- `models--prehj--GR00T-N1.5-robocasa-metaq-v3b-n32-K6-b-only-60k` (17G)

## 버전 모호 (suffix 없음 — v1 변형일 수도, 별개 ablation일 수도)
- `models--prehj--gr00t-n1.5-robocasa-metaq-length_cost_0p20` (17G)
- `models--prehj--gr00t-n1.5-robocasa-metaq-mean_sq_balance` (17G)
- `models--prehj--gr00t-n1.5-robocasa-metaq-no_ema_norm` (17G)
- `models--prehj--gr00t-n1.5-robocasa-metaq-no_shared_t` (17G)

## 합계
- 보존 권장(v1): 2 dirs ≈ 34G
- 명시 비-v1: 7 dirs ≈ 119G
- 버전 모호: 4 dirs ≈ 68G
- 모두 삭제 시: 13 dirs ≈ 221G 회수 (단 v1 2개는 보존이 안전)
- v2/v3만 삭제 시: 7 dirs ≈ 119G

## metaq 안 쓴 ckpt들 (참고 — 이 폴더들은 metaq 아님, no_metaq)
같은 HF 디렉토리에 `no_metaq` 들어간 16개 폴더는 모두 metaq를 명시적으로 사용하지 않은 모델들. 이름이 metaq grep에 걸리지만 삭제 대상 아님.
