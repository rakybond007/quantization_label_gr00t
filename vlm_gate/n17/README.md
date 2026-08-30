# n1.7 공동 파인튜닝 게이트

GR00T-N1.7 위에 quantization confidence 게이트를 붙이는 조각들.
업스트림 트리는 공개 태그에서 재현되므로 여기엔 **우리가 만든 것만** 둔다.

## 왜 이 구조인가

동결된 정책에서 특징을 빌려 쓰면(기존 B 변형) 게이트가 알 수 있는 것에 천장이 생긴다 —
액션 예측에 맞춰진 표현을 그대로 받기 때문이다. 여기서는 게이트가 백본의 **다른 레이어**를
읽고 액션 헤드와 **함께 학습**된다. unfreeze 된 상위 레이어가 두 목적을 서로 다른 깊이에서
만족하도록 움직인다.

- 액션 헤드: `hidden_states[select_layer]` (n1.7 체크포인트 기준 16)
- 게이트: `hidden_states[gate_layer]` (8 · 10 · 12 중 선택), **이미지 토큰만** attention 풀링
- 백본 forward 는 정책이 어차피 하므로 **추가 비전 연산 0**
- 출력 타겟은 다른 모듈과 동일한 청크당 스칼라 `P(quantize)`

이미지 토큰만 보는 이유: 전체 토큰 평균 풀링은 텍스트까지 섞어 공간·객체 정보를 잃는다.
n1.7 백본이 `image_mask` 를 내보내므로 골라낼 수 있다 (n1.6 Eagle 백본에는 없다).

## 적용

```bash
git worktree add -b n17-quant-gate ../Isaac-GR00T-n17 n1.7-release   # 공개 태그
cp quant_gate.py        ../Isaac-GR00T-n17/gr00t/model/modules/
cp quant_gate_labels.py ../Isaac-GR00T-n17/gr00t/data/dataset/
git -C ../Isaac-GR00T-n17 apply ../vlm_gate/n17/gr00t_n1d7_gate.patch
```

## 학습 배선

라벨은 `get_datapoint` 에서 실린다. 콜레이터가 샘플 딕셔너리의 모든 키를 스택하므로
셔딩된 데이터셋을 다시 굽지 않아도 모델 `forward` 까지 전달된다.
교사 라벨이 없는 스텝은 `gate_valid=0` 으로 손실에서 **제외**한다 (0.5 로 끌어당기지 않는다).

```python
model.attach_quant_gate(gate_layer=10, loss_weight=1.0)
lookup = GateLabelLookup(".../v6b_phase5_1call_full.parquet")
patch_dataset_gate_labels(dataset, lookup)
```

## 환경

n1.7 은 Qwen3-VL 백본이라 transformers 4.57.3 이 필요하다. 평가 스택의 4.51.3 고정을
깨지 않도록 별도 디렉터리 오버레이를 쓴다 (numpy 는 환경 것을 유지 — 섞이면 평가가 죽는다).

```bash
PYTHONPATH=$WS/pylibs/tf4573 python ...
```

확인된 사실: 기존 cu124 드라이버에서 로드된다. 백본 1,524M, `select_layer=16`,
`tune_top_llm_layers=4` 로 레이어 12–15 unfreeze (텐서 44개).

## 주의 — 비교 기준이 달라진다

지금까지 이 프로젝트의 전제는 "정책은 고정, 추론 시점에만 개입"이었다. 공동 파인튜닝은
정책 자체를 바꾸므로 **게이트 없는 n1.7 robocasa 베이스라인을 따로 뽑아야** 하고,
1.5 기반 결과와는 직접 비교되지 않는다.
