# 인계: allex conf 라벨의 서브태스크 상한 재조정

받는 사람: 같은 데이터셋을 가진 **다른 서버의 코딩 에이전트**.
VLM 을 다시 돌릴 필요 없습니다. 배달된 JSON 안의 값만으로 다시 계산됩니다.

## 0. 한 줄 요약

라벨은 `K = snap(1 + p*(K_max - 1))` 로 만들어졌고 `p`·`K_max` 가 청크마다
저장돼 있습니다. 그래서 **서브태스크 상한만 옮기고 다시 구우면** 됩니다.

## 1. 대상 파일

```
<데이터셋루트>/meta/hojin_quantization_confidence.json
```

이쪽 서버에서의 실제 경로 (참고):
```
/rlwrld2/home/david/action_quantization/v1/subtask_labeled_data_update_eef_256x256_hojin/meta/hojin_quantization_confidence.json
```

구조:

| 필드 | 뜻 |
|---|---|
| `column` | 데이터셋에 붙는 열 이름 (`hojin`) |
| `params.grid` | K 가 스냅되는 격자 `[1.0, 2.0, 2.5, 3.0]` |
| `params.method` | `two-stage VLM gate: ... K = snap(1 + p*(K_max-1))` |
| `chunks` | 14,809 개. 각각 `episode_index · start_frame · end_frame · seg · label · K · K_max · p · hojin` |
| `k_stats`, `overall_mean` | 요약 통계 (재계산하면 같이 갱신해야 함) |

`label` 이 서브태스크입니다: `Bring Object` (5692) · `Rotate Box` (5597) ·
`Pass Object` (2548) · `Rotate PolyBag` (972).

## 2. 왜 재계산이 되는가 -- 먼저 이걸 확인하고 시작하세요

`K_max` 는 **청크마다** 다릅니다. 서브태스크 사전상한을 2단 문항이 청크별로
조인 결과이기 때문입니다. `p` 는 1단 신뢰도입니다. 둘 다 저장돼 있으므로

```
K     = grid 중 (1 + p*(K_max-1)) 에 가장 가까운 값
hojin = clip((K-1)/(max(grid)-1), 0, 1)
```

로 K 를 되살릴 수 있습니다. **이쪽에서 재현율 14792/14809 = 99.885% 를
확인했습니다.** 먼저 이 검증부터 돌리고, 99% 미만이면 멈추세요 -- 공식이나
grid 가 다르다는 뜻입니다.

```bash
python allex_rescale_ceiling.py <입력.json> /dev/null --verify-only
```

## 2.5. **배달본은 상한 개정 이전 판입니다** -- 이게 재조정이 필요한 이유

최종 지시값 (2026-09-14 운용자):

```python
SPEC_CEILING = {"Bring Object": 2.0, "Pass Object": 2.5,
                "Rotate Box": 1.75, "Rotate PolyBag": 3.0}
SPEC_FLOOR  = {}            # 전 서브태스크 하한 1.5
DEFAULT_FLOOR = 1.5
GRID_FINE = [1.0, 1.5, 1.75, 2.0, 2.5, 3.0]
```

배달본을 만든 판(`scripts/allex_v2_common.py:198`)의 값과 다릅니다. 배달본은
2026-09-01 생성이고 그 뒤 두 번 갱신됐습니다.

**Rotate PolyBag 만 상한을 올렸습니다(3.0).** 운용자 판단은 "봉투는 빨리
뒤집어도 된다" 인데 1단 신뢰도가 거꾸로 나왔습니다 -- p 평균이 네 서브태스크 중
최저입니다 (Bring 0.689 · Pass 0.630 · Rotate Box 0.376 · **Rotate PolyBag
0.189**). 상한을 열어 순서를 바로잡은 것이고, **p 자체는 건드리지 않았습니다.**

배달본은 **2026-09-01** 생성이라 이 개정 전입니다. 대조하면:

| 서브태스크 | 명세 상한 | 파일 최대 | 초과칸 | 하한 | **하한 미달칸** |
|---|---|---|---|---|---|
| Bring Object | 3.00 | 3.000 | 0.0% | 1.50 | **37.4%** |
| Pass Object | 3.00 | 3.000 | 0.0% | 1.50 | 8.9% |
| Rotate Box | 2.00 | 3.000 | 1.9% | 1.50 | 0.0% |
| Rotate PolyBag | 2.50 | 2.877 | 0.2% | 1.50 | 0.1% |

Rotate 계열은 상한을 넘고, Bring/Pass 는 하한 아래로 내려가 있습니다 -- 2단
문항이 과하게 조인 것입니다. 그래서 이 재조정은 상한만이 아니라 **하한도 같이**
적용합니다.

## 3. 상한을 옮기는 방법 -- 비율로 옮깁니다

서브태스크 안에서 `K_max` 가 청크마다 다른 것은 2단 문항이 그 청크를 조인
결과입니다. **그 조임을 뭉개면 안 됩니다.** 그래서 값을 갈아끼우지 않고
비율로 옮깁니다. 서브태스크 L 의 현재 상한을 C_L (그 서브태스크 `K_max` 의
최댓값), 새 목표를 T_L 이라 하면

```
K_max'(청크) = 1 + (K_max(청크) - 1) * (T_L - 1) / (C_L - 1)
```

이면 그 서브태스크 안의 상대 순서와 조임 비율이 그대로 남고 꼭대기만 T_L 로
갑니다. `T_L = C_L` 이면 항등입니다 (이쪽에서 확인함: `--scale 1.0` 이 원본
분포를 되돌려 줍니다).

## 3.5. 격자를 늘려야 합니다

K 는 격자에 스냅됩니다. 배달본의 격자는 `[1, 2, 2.5, 3]` 이라 **1 과 2 사이가
비어 있습니다.** 상한을 내리면 원값 `1 + p*(K_max-1)` 이 1.5(1 과 2 의 중간점)
를 못 넘는 청크가 많아지고, 그 전부가 1.0 으로 떨어집니다 -- 지시값으로 돌리면
Rotate Box 는 99.7%, Rotate PolyBag 은 100% 가 K=1.0 이 됐습니다.

`--grid fine` 이 `[1, 1.5, 1.75, 2, 2.5, 3]` 을 씁니다. 1.75 만 넣는 것으로는
거의 안 살아납니다(Rotate Box 100% -> 99% 가 여전히 1.0) -- **1.5 눈금이
있어야** 합니다.

**쓰는 쪽이 분수 배속을 실행할 수 있어야 합니다.** `hojin = (K-1)/2` 는
연속값이라 정규화는 그대로 동작하고, `params.grid` 에 새 격자가,
`params.grid_before_rescale` 에 원래 격자가 기록됩니다. 재현 검증은 늘 **파일에
적힌 원래 격자로** 합니다.

## 4. 실행

스크립트는 이 인계문과 같이 갑니다. 원본 절대경로:
`/sjw_alinlab/home/hojin2/quantization_agent_workspace/vlm_gate/scripts/allex_rescale_ceiling.py`
(표준 라이브러리만 씁니다. numpy 도 필요 없습니다.)

```bash
# **권장.** 지시값(상한+하한 1.5)에 늘린 격자까지.
python allex_rescale_ceiling.py <입력.json> <출력.json> --spec --grid fine

# 값을 직접 주려면
python allex_rescale_ceiling.py <입력.json> <출력.json> \
    --ceiling "Rotate Box=2.0,Rotate PolyBag=2.5" --floor "Bring Object=1.5,Pass Object=1.5"

# 전 서브태스크의 (상한-1) 에 일괄로 곱하기
python allex_rescale_ceiling.py <입력.json> <출력.json> --scale 0.8
```

하한은 상한을 옮긴 **뒤에** 걸립니다: `K_max' = min(상한, max(하한, 비율이동값))`.

찍히는 것: 재현율 검증 → 서브태스크별 (현재상한 · 새상한 · 평균K 전 · 후) →
K 분포 전/후 → 평균 hojin.

이쪽에서 `--spec --grid fine` 으로 돌려 본 결과:

```
서브태스크              현상한   새상한   하한   평균K 전    후
  Bring Object         3.000   2.000   1.50    1.533  1.464
  Pass Object          3.000   2.500   1.50    1.989  1.735
  Rotate Box           3.000   1.750   1.50    1.221  1.109
  Rotate PolyBag       2.877   3.000   1.50    1.278  1.294

K 분포 전: {1.0: 8602, 2.0: 4683, 2.5: 1337, 3.0: 187}
K 분포 후: {1.0: 5822, 1.5: 6724, 1.75: 990, 2.0: 1211, 2.5: 62}
평균 K 1.477 -> 1.365 · 평균 hojin 0.2385 -> 0.1826
```

서브태스크별 최종 K 분포:

| 서브태스크 | 평균 K | 1.0 | 1.5 | 1.75 | 2.0 | 2.5 |
|---|---|---|---|---|---|---|
| Pass Object | 1.735 | 7.5% | 31.3% | 18.4% | 40.3% | 2.4% |
| Bring Object | 1.464 | 13.6% | 77.0% | 6.3% | 3.1% | — |
| Rotate PolyBag | 1.294 | 50.0% | 32.9% | 16.6% | 0.5% | — |
| Rotate Box | 1.109 | 78.1% | 21.9% | 0.0% | — | — |

**Rotate PolyBag 은 상한을 3.0 까지 열었는데도 절반이 K=1.0 입니다.** p 가
낮아서(0.189) 원값이 안 올라갑니다. 순서는 Rotate Box 위로 올라왔지만 크기는
운용자 판단보다 여전히 보수적입니다. 고치려면 p 를 손대야 하는데 그것은 라벨의
판단을 바꾸는 일이라 이 도구의 범위 밖입니다.

## 5. 지켜야 할 것

- **입력 파일을 덮어쓰지 마세요.** 스크립트도 안 합니다. 새 경로로 내고,
  갈아끼우는 것은 사람이 확인한 뒤에 하세요.
- `k_stats` · `overall_mean` 은 스크립트가 다시 계산해 넣습니다. 손으로
  고치지 마세요.
- **2단 문항의 물리 한계는 다시 계산되지 않습니다.** `params.limits_source`
  가 "calibrated on THIS recording; constants are not carried between
  recordings" 라고 적어둔 그대로입니다. 이 재조정은 **사전상한만** 옮깁니다.
  녹화본이 다르면 `merge_limit_rad` · `rot_accum_limit_deg` ·
  `gap_rate_limit_m_per_step` 를 그 녹화본에서 다시 보정해야 하고, 그건 이
  스크립트의 일이 아닙니다.
- **`BRING_SOFT` 는 소급 적용이 안 됩니다.** 명세는 Bring Object 가 상자면 3.0,
  봉투면 2.0 인데, 그 판정은 2단 문항 D 가 합니다. 배달본의 청크는
  `episode_index · start_frame · end_frame · seg · label · K · K_max · p · hojin`
  만 담고 있어 **문항 D 의 답이 없습니다.** 봉투 구분까지 반영하려면 2단을 다시
  돌려야 하고, 그건 이 스크립트의 일이 아닙니다. `--spec` 은 Bring Object 를
  상자 기준 3.0 으로 둡니다.
- `modality.json` 에 열을 등록해 둔 상태라면 그쪽은 건드릴 필요가 없습니다
  (열 이름 `hojin` 이 그대로이므로). 이쪽에는
  `meta/modality.json.bak-before-effort` 백업이 남아 있습니다.

## 6. 알아 두실 것 -- 부호

이 라벨 계열의 이전 판(v3)에서 **부호가 뒤집혀 있었습니다.** 균형 잡힌 개발
집합에서 세 가지 집계 모두 −0.75 ~ −0.94 였고, 그 판의 자체 검증기도 이미
부호 게이트 2문항을 떨어뜨린 상태였습니다. 상한 재조정은 부호 문제를 고치지
않습니다 -- 크기만 바꿉니다. 새로 쓰기 전에 유지율과의 상관 부호를 꼭 다시
확인하세요.
