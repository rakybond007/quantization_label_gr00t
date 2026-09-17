# DexJoCo v3 — Gemini 라벨링에 실제 사용한 프롬프트

원본: `../scripts/dexjoco_v3_checks.py`. 최신 main `2d5b579` 위에 추가했다.
기존 RoboCasa v22 / Libero v3d / HumanData v3의 공통 ASK 및 1–5 척도를
그대로 사용한다. 별도 SYSTEM 메시지나 YES/NO 지시를 보내지 않는다.

| 파일 | 내용 |
|---|---|
| `dexjoco_v3_FULL.txt` | 이미지 2장 → GUIDANCE → 시야 설명 → 지시문 → 액션 계산 사실 → ASK |
| `dexjoco_v3_FILLED.txt` | 실제 click_mouse 에피소드 0 장면의 조립된 텍스트 |
| `dexjoco_v3_guidance.txt` | 판단 관점 |
| `dexjoco_v3_questions.txt` | 등급 척도와 A–D 문항 |
| `dexjoco_v3_facts_example.txt` | 실제 입력에 포함한 액션 계산 정보 |
| `dexjoco_v3_sign_weight.txt` | 후처리용 부호·잠정 가중치; VLM에 전달하지 않음 |

카메라 순서는 scene(ego_right 또는 front), wrist이다. 두 이미지는 같은
시점의 서로 다른 카메라다. FULL은 첫 예시의 ego_right를 사용하며, 다른
태스크의 실제 카메라 이름은 `build_messages`의 `views`로 전달한다.

A는 잡고 있는 물체의 가동부 조작, B는 국소 표적에 대한 도구 타격 준비/수행,
C는 표면에 지지된 대상 누르기, D는 이미 잡은 물체를 공간에서 통째로
운반/기울이기 정도를 묻는다. 장면의 실제 등급은 VLM이 답한다.

사용자 요청으로 **2x/3x/4x 후보별 이동량 문장은 삭제**했다. 팔·손목·손가락의
데이터셋 통계 대비 빠르기, 방향 반전, 후반 감속 정보는 남긴다. 이는 기록된
절대 타깃으로 계산한 command 정보이며, 접촉·파지 성공을 측정한 값은 아니다.
FORMAT.md의 후보 배속별 수치 예시와 달리 이 삭제 결정을 적용한 버전이다.

## 재생성 및 검증 (API·GPU·데이터셋 불필요)

레포 루트에서:

```bash
python vlm_gate/scripts/sync_prompts_folder.py --benchmark dexjoco
python vlm_gate/scripts/sync_prompts_folder.py --benchmark dexjoco --check
python vlm_gate/scripts/verify_prompts_against_code.py --benchmark dexjoco --selftest
```

표준 전체 생성·검증 경로에도 DexJoCo를 등록했다. `--benchmark dexjoco`는
기존 벤치마크의 외부 `VLM_GATE_SRC`나 모델 의존성 없이 이 추가분만 확인한다.
48장면의 실제 실행 텍스트 SHA256과 세 벤치마크의 공통 ASK를 대조한다.
자기검사는 여섯 생성 파일을 각각 변조했을 때 탐지하는지도 확인한다.
`../tests/fixtures/dexjoco_v3_panel.json`에는 지시문·계산 사실·카메라 이름·해시만
있으며 API 키, 영상 또는 이미지 바이트를 포함하지 않는다.

## 채점 및 현재 라벨링

Gemini `gemini/gemini-3.8-flash`, temperature 0, reasoning low, max tokens 1024.
등급 정수 A–D를 저장한다. 해당 API가 logprobs를 반환하지 않으므로 확률 가중합
등급이 아니다. `g=(grade-1)/4`, 비교용 `conf=(1+C/3+2D/3-2A/3-B/3)/2`이며
식의 A–D는 정규화된 g다. 가중치는 잠정이고 성공 확률이나 배속 정답이 아니다.
등급을 저장하므로 후처리 가중치는 재라벨링 없이 바꿀 수 있다.

2026-09-18 API 검증: 동시 요청 4개 48/48 유효(73초), 8개 48/48 유효(41초).
이는 형식·연결 검증이며 실제 압축 정책의 성공률 검증은 아니다.
6개 single-arm 태스크, 600에피소드의 완전한 16프레임 창을 stride 16으로
샘플링해 13,352건을 라벨링한다. all-zero absolute target 포함 창과 불완전한
꼬리는 제외한다. 원본 키는 `(task, ep_local, frame_index)`이며 task 내부의
packed row index도 저장한다. 보간이나 배속 배정은 이 프롬프트의 기능이 아니다.

실행 원본과 기록 위치:
`/mnt/lustre/slurm/users/prehj/atq_workspace/experiments/dexjoco_format_v3/`.
이 커밋은 프롬프트와 재현 검증을 추가하며, 진행 중인 라벨링의 프롬프트를
바꾸거나 실행 프로세스를 재시작하지 않는다.
