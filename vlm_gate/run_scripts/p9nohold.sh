set -u
E=$HOME/quantization_agent_workspace/quantization_label_gr00t/vlm_gate/experiments/proprio_history
# 홀드아웃 없는 전량 학습 경로 스모크. --max-tasks 를 안 걸어 334종을 다 본다.
for A in proprio baseline; do
  D=$E/artifacts/p9nohold_$A
  rm -rf $D
  echo "########## $A ##########"
  bash $E/run_phase9.sh $D $A --episodes-per-task 2 --frames-per-episode 2 \
       --epochs 2 --bs 64 --num-workers 0 --preload 2>&1 | tail -6
  echo "--- 산출물 ---"; ls $D 2>/dev/null
  python3 - "$D" <<'PY'
import json, sys
try:
    s = json.load(open(sys.argv[1] + "/summary.json"))
    print("  저장된 에폭:", s.get("best_epoch") or s.get("epoch") or "?",
          "| 학습행", s.get("train_rows"), "| 검증행", s.get("val_rows"))
except Exception as e:
    print("  summary 못 읽음:", e)
PY
done
