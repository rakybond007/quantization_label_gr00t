# 서버 지도 (맥북에는 없는 것들)

서버 홈: `/sjw_alinlab/home/hojin2`  ·  작업 루트: `~/quantization_agent_workspace` (= `$WS`)

## 맥 클론 ↔ 서버 대응

`dev`가 아래 매핑으로 코드를 올린다. 배치가 서로 다르니 주의.

| 맥북 클론 | 서버 |
|---|---|
| `vlm_gate/scripts` | `$WS/vlm_gate/scripts` |
| `vlm_gate/run_scripts` | `$WS/vlm_gate/run_scripts` |
| `vlm_gate/analysis` | `$WS/vlm_gate/analysis` |
| `gr00t` | `$WS/Isaac-GR00T/gr00t` |
| `tools` | `$WS/tools` |

주의: 서버의 `$WS/Isaac-GR00T/vlm_gate`는 **git 씨앗일 뿐 실제 작업 트리가 아니다.**
스크립트들이 `$WS/vlm_gate` 를 하드코딩하고 있다. 그쪽을 고치지 말 것.

## 서버에만 있는 것 (로컬에서 못 연다)

| 경로 | 크기 | 내용 |
|---|---|---|
| `$WS/assets/frame_cache_robocasa` | 37G | 262k 프레임 uint8 memmap |
| `$WS/assets/vla_gateC` | 36G | gateC 가중치 |
| `$WS/assets/datasets` | 2.0G | allex 등 |
| `$WS/assets/labels/robocasa/*.parquet` | 47M | 라벨 (phase5 = 247,887행) |
| `$WS/assets/modules_A/` | 27M | 학생 A' 체크포인트들 |
| `$WS/assets/robocasa_task_embeddings.npz` | 893K | eval 필수 |
| `$WS/vlm_gate/output/_gate_distill/` | 249G | 라벨 jsonl, 타일 26만장 |
| `$WS/vlm_gate/out/` | 47M | slurm 로그 (`%A_%a-%x.out`) |

**타일 디렉터리(`output/_gate_distill/luna_robocasa_full/tiles`)에 `ls`/`du`/글롭 금지.**
평평한 구조에 260,031개 PNG다. `output/_gate_distill/tiles_manifest.txt`를 읽어라.

**보관 정책:** `/rlwrld-unified-checkpoints`는 7일 미접근 시 오브젝트 스토리지로 이관,
90일 후 삭제. 읽기만으로는 보호되지 않는다(과거에 37G 프레임 캐시가 이렇게 사라졌다).
계속 쓰는 자산은 반드시 `$WS/assets/` (홈)에 둔다.

## conda 환경

`source ~/miniconda3/etc/profile.d/conda.sh` 를 먼저 해야 `conda activate`가 된다.

| 환경 | 용도 | 비고 |
|---|---|---|
| `quant_gate` | 정책 서버 (gr00t 추론) | |
| `quant_gate_eval` | robocasa 클라이언트, A' 학습, 집계 | numpy 1.23.5 / transformers 4.51.3 **고정** |
| `cosmos_judge` | cosmos VLM 라벨러 | |
| `vlm_judge` | 구 판정기 | |

`quant_gate_eval`의 numpy·transformers 버전을 올리지 말 것. 과거에 이것 때문에
평가가 10일간 조용히 죽었다 (robocasa가 numpy 1.23.x를 요구 → transformers 5.x가
gr00t import를 깨뜨림 → npz가 numpy2 pickle로 저장됨).

## slurm 정책

- `--wckey=project-short-name:sub_fast` **필수**
- `MODEL_OUTPUT_DIR` 필수, **`/rlwrld-unified-checkpoints/hojin2/` 로 시작해야 한다** (가드가 접두사만 검사).
  단 **실제 산출물은 거기 쓰지 않는다** — 90일 삭제 정책 때문에 `--out-dir`은 홈 `assets/`로 주고
  이 변수는 규정 충족용으로만 설정한다.
- 파티션: **학습 = `sjw_alinlab`**, **평가·라벨링 = `background`**
- `srun` 옵션은 `--gpus` / `--job-name` / `--wckey` / `--exclude` 만.
  `--time`, `-p`는 시스템 기본값이 붙으므로 지정하지 않는다.
- 자원이 부족해도 이상한 파티션으로 우회하지 않는다.
- 죽는 노드: `--exclude=worker-node100,worker-node1,worker-node104,worker-node3`
- `background` 파티션은 선점된다. 재큐 시 결과 파일이 비워질 수 있으니
  에피소드 단위 sidecar로 남길 것 (LIBERO에서 실제로 당했다).

## git 리모트

| 저장소 | 리모트 | 브랜치 |
|---|---|---|
| 본 저장소 | `quant` = `rakybond007/GR00T-action-quantization` | `action-quantization-gate-v2` |
| | `origin` = `ismty0805/Isaac-GR00T` (업스트림, 푸시 금지) | |
| alin-skills | `alinlab/alin-skills` | `add-work-board` (PR 대기 중) |
