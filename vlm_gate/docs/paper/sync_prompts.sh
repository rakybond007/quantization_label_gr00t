#!/bin/bash
# 부록용 프롬프트 사본을 레포의 정본 모음에서 다시 만든다.
#
# 정본은 vlm_gate/prompts/<벤치>_<판>_FULL.txt 다 -- scripts/sync_prompts_folder.py 가
# 권위 있는 원본(각 checks .py 의 GUIDANCE/ASK)에서 생성하고 --check 로 어긋남을 잡는다.
# 여기는 그 사본의 사본이며, Overleaf 에 올릴 때 폴더를 통째로 옮기기 위해 있다.
# output/ 이나 HF 를 뒤지지 않는다: 정본이 둘이 되면 언젠가 갈라진다.
#
#   bash sync_prompts.sh          복사
#   bash sync_prompts.sh --check  어긋난 파일이 있으면 1 로 끝난다
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../../prompts"
FINAL="robocasa_v22 libero_v3d humandata_v3 openarm_v2 allex_v4c dexjoco_v3"
rc=0
for name in $FINAL; do
  src="$SRC/${name}_FULL.txt"; dst="$HERE/prompts/$name.txt"
  [ -f "$src" ] || { echo "[ERR] 정본이 없다: $src"; exit 1; }
  if [ "${1:-}" = "--check" ]; then
    cmp -s "$src" "$dst" && printf "  %-20s 동일\n" "$name" || { printf "  %-20s 어긋남\n" "$name"; rc=1; }
  else
    cp "$src" "$dst"; printf "  %-20s <- %s (%s줄)\n" "$name.txt" "${name}_FULL.txt" "$(wc -l < "$src")"
  fi
done
exit $rc
