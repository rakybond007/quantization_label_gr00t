#!/bin/bash
# Render one labelled clip per allex subtask, so the four can be compared
# side by side instead of watched in sequence.
#
#   bash vlm_gate/scripts/render_allex_pertask.sh [frames]
#
# The combined render (allex_v2_render.py with no ONLY_TASK) stays what it is:
# episode 0 end to end followed by the four segments. This produces the same
# segments as separate files.
set -u
WS="$HOME/quantization_agent_workspace"
WANT="${1:-260}"
OUT="$WS/assets/videos/pertask"
mkdir -p "$OUT"
cd "$WS" || exit 1

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate quant_gate_eval

for t in "Bring Object" "Rotate Box" "Pass Object" "Rotate PolyBag"; do
  slug=$(echo "$t" | tr 'A-Z ' 'a-z_')
  echo "=== $t -> $slug.mp4 (${WANT} frames) ==="
  ONLY_TASK="$t" WANT="$WANT" \
    python vlm_gate/scripts/allex_v2_render.py "$OUT/$slug.mp4" 2>&1 | tail -2
done

echo "=== done ==="
ls -l "$OUT"
