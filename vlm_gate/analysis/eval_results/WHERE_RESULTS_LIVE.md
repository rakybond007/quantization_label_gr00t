# 결과가 어디 있나

**이 저장소의 `output/` 이 전부가 아니다.** 동료들이 낸 결과가 서버 곳곳에 있고,
그것을 못 찾아 같은 실행을 다시 뜨는 일이 실제로 있었다(2026-09-07, libero K=1).
새 eval 을 던지기 전에 여기부터 본다.

| 무엇 | 어디 | 비고 |
|---|---|---|
| 이 저장소 실행 | `vlm_gate/output/{robocasa,libero,dexjoco}/` | `qgate.evalscan` 이 읽는다 |
| LIBERO K=1 40태스크 정본 | `/sjw_alinlab2/home/taekwan/Data/libero_all_Fine` | b32 · seed 7 · 0.9565 · README 있음 |
| LIBERO 옛 실행 (multigpu) | `~/multigpu_workspace/Isaac-GR00T/output/libero/` | `baseline_bs32_hf` 등. 24태스크짜리도 섞여 있다 |
| robocasa 체크포인트 | `~/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000` | 압축·기준선 계열 전부 이것 |

## 실행을 던지기 전에 확인할 것

1. **같은 값이 이미 있는가.** 위 표의 자리를 다 본다. 없으면 동료 세션에 묻는다 --
   나 혼자 작업자가 아니다.
2. **체크포인트가 같은가.** 폴더 시각으로 추정하지 않는다. `server-*.log` 에서
   `models--...` 를 뽑는다. 날짜로 갈랐다가 틀린 적이 있다(libero K3·K4 를 b64 로
   잘못 봤는데 실제로는 b32 였다).
3. **경로가 다른 것과 체크포인트가 다른 것은 다르다.** `/sjw_alinlab2/` 와
   `/sjw_alinlab/` 은 2026-08 홈 이사 때문이고 같은 체크포인트다.
4. **클립·구동 축이 같은가.** `clip_scale` 은 1층(명령 클립), `dyn_scale` 은
   2층(토크캡·forcerange·OSC kp/kd)이다. 하나만 풀면 절반만 푼 것이다.

새로 뜬 실행에는 `policy_ckpt` 헤더가 `prediction.txt` 에 들어간다(동료 세션
`65ad2f4`). 그 전 실행은 로그에서 뽑아야 한다.
