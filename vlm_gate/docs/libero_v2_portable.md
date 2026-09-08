# LIBERO v2 전량 라벨링 (portable)

이 절차는 `nvidia/Cosmos3-Nano`가 LIBERO의 두 카메라를 보고 v2 문항 A~E에
각각 1~5 등급을 매기는 전량 추론용 절차다. 기본은 매 프레임(`TILE_STRIDE=1`)과
배치 8이며, 마지막 네 프레임은 descriptor의 미래 창이 없어 제외된다. 현재 데이터
기준 기대 라벨 수는 `273465 - 4 * 1693 = 266693`이다.

경로는 아래 환경 변수로 지정한다. 지정하지 않으면 기존 sjw 경로를 그대로 쓴다.

```bash
export VLM_GATE_ROOT=/path/to/quantization_label_gr00t/vlm_gate
export LIBERO_DATASET=/path/to/libero_gr00t_delta
export WORK=/path/to/work
export LIBERO_TILE_OUT="$WORK/output/_gate_distill/libero_full"
export LIBERO_MANIFEST="$WORK/output/_gate_distill/libero_tiles_manifest.txt"
export LIBERO_LABEL_OUT="$WORK/output/_gate_distill"
```

## 타일 생성

데이터셋의 `meta/info.json`에 따라 MP4와 parquet 내장 image를 자동으로 읽는다.
front view를 왼쪽, wrist view를 오른쪽에 붙이고 기존과 동일하게 절반 크기로 줄인다.
샤드 수 `N`을 정한 뒤 모든 샤드에서 같은 `N`을 유지한다.

```bash
export TILE_STRIDE=1
for s in $(seq 0 3); do
  python "$VLM_GATE_ROOT/scripts/gen_libero_tiles_shard.py" "$s" 4
done
python "$VLM_GATE_ROOT/scripts/gen_libero_tiles_shard.py" merge 4
wc -l "$LIBERO_MANIFEST"                 # 273465 기대
```

`merge`는 샤드 매니페스트 하나라도 없으면 실패한다. 8장짜리 I/O 스모크는 본 출력과
섞이지 않는 별도 `LIBERO_TILE_OUT`/`LIBERO_MANIFEST`에서 `TILE_LIMIT=8`로 실행한다.

## Cosmos3-Nano와 라벨러

A6000에서는 저장소에서 검증된 Transformers cu124 경로를 사용한다. vLLM 경로는
CUDA 13 런타임을 요구해 550 계열 드라이버와 맞지 않는다. 환경을 새로 만드는 경우:

```bash
uv venv --python 3.13 /path/to/cosmos_judge_venv
uv pip install --python /path/to/cosmos_judge_venv/bin/python \
  --torch-backend=cu124 'torch==2.6.0' 'torchvision==0.21.0' \
  'transformers==5.12.1' accelerate av pillow 'safetensors>=0.8.0'
```

GPU 할당 안에서 서버를 띄우고 로그의 `JUDGE READY`를 확인한다.

```bash
export PYTHONPATH="$VLM_GATE_ROOT/scripts"
GATE_SYSTEM=aligned CUDA_VISIBLE_DEVICES=0 \
  /path/to/cosmos_judge_venv/bin/python -u \
  "$VLM_GATE_ROOT/scripts/vlm_gate_cosmos.py" --serve \
  --model nvidia/Cosmos3-Nano --host 127.0.0.1 --port 8123 \
  > /path/to/work/cosmos.log 2>&1 &
```

먼저 별도 `TAG`와 출력 디렉터리에서 정확히 한 배치로 스모크한다. 그 뒤 전량 실행은
`LIMIT`을 제거하고 샤드별 프로세스를 실행한다.

```bash
PROMPT_VER=v2 LIBERO_BATCH=8 LIMIT=8 TAG=libero_v2_smoke \
  python "$VLM_GATE_ROOT/scripts/libero_label_chunks.py" 8123 0 1

export PROMPT_VER=v2 LIBERO_BATCH=8 TAG=libero_v2
for s in $(seq 0 3); do
  python "$VLM_GATE_ROOT/scripts/libero_label_chunks.py" 8123 "$s" 4
done
```

재개할 때 `LIBERO_BATCH`, `PROMPT_VER`, 샤드 수는 바꾸지 않는다. 기존 JSONL에
metadata가 없거나 셋 중 하나가 다르면 라벨러가 출력 정리나 metadata 덮어쓰기 전에
실패한다. 원격에서 pending이던 실행의 샤드 수나 설정을 확인할 수 없다면 그 출력을
이어 쓰지 말고 새 `LIBERO_LABEL_OUT`에서 시작한다.

## 완결성 검사와 배속 라벨 패키징

모든 샤드 JSONL만 합친 뒤 정본 감사 도구로 행 수, 중복, 손상된 행, 빈 샤드와 문항
분산을 검사한다. `_meta.json`은 합치지 않는다.

```bash
QGATE_OUTPUT="$WORK/output" bin/qgate labels libero_v2_s4 --expected 266693 -v
python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['LIBERO_LABEL_OUT'])
with (root / 'libero_v2_merged.jsonl').open('w') as out:
    for p in sorted(root.glob('libero_v2_s4_*.jsonl')):
        out.write(p.read_text())
PY

LIBERO_DATASET="$LIBERO_DATASET" RATIO_RULE=levels \
  python "$VLM_GATE_ROOT/scripts/ratio_label.py" libero \
  "$WORK/output/_gate_distill/libero_v2_merged.jsonl" \
  "$WORK/output/_gate_distill/libero_v2_ratio.jsonl"

python "$VLM_GATE_ROOT/scripts/pack_ratio_labels.py" libero \
  "$WORK/output/_gate_distill/libero_v2_ratio.parquet" \
  --out "$WORK/dist/libero_ratio"
```

`levels`가 현재 실제 기본 규칙이다. 라벨 의미를 유지하려면 명시적으로 적는다.
패키징 로그에서 266693행, 1693개 에피소드, 40개 태스크를 확인하고 `ratio`가
1.0/1.5/2.0/2.5 이외 값을 갖지 않는지 확인한다. 외부 업로드는 별도 승인 후
`pack_ratio_labels.py --push <repo_id>`로 수행한다.

## 이 서버 검증 결과 (2026-09-09)

A6000 한 장에서 Cosmos3-Nano revision
`7a312c868bcce8e40b3eb40861300a9d0ba3fde1`로 실제 LIBERO 이미지 8개를
v2/5등급/batch 8/text 모드로 판정해 8행, 누락 0을 확인했다. 같은 에피소드의
인접 프레임 검사이므로 이 결과만으로 전체 태스크의 라벨 품질을 보장하지 않는다.

공통 클라이언트 `VLMGate.judge_batch`가 `mode` 인자를 받지 못하던 오류를
수정했다. 서버 오류의 상세 본문도 이제 로그에서 확인할 수 있다.

GPU 여러 장이 보이면 `device_map="auto"`가 모델을 여러 GPU에 나눌 수 있다.
검증 노드에서는 이때 통신이 멈췄고, 할당된 GPU 한 장만 보이게 하자 통과했다.
각 판정 서버는 할당된 GPU 하나만 사용하도록 지정한다. Slurm의 GPU 번호나 UUID를
보존해야 하므로 무조건 물리 GPU 0을 선택하지 않는다.
