set -u
E=$HOME/quantization_agent_workspace/quantization_label_gr00t/vlm_gate/experiments/proprio_history
# 실패한 경로를 지나가는 스모크: --max-tasks 를 안 건다. 334종 지시문을 다 보므로
# 에피소드가 하나뿐인 9종을 실제로 만난다. 앞 스모크는 8종만 뽑아서 못 만났다.
for A in proprio baseline; do
  D=$E/artifacts/p9fix_$A
  rm -rf $D
  echo "########## $A ##########"
  bash $E/run_phase9.sh $D $A --episodes-per-task 2 --frames-per-episode 2 \
       --epochs 1 --bs 64 --num-workers 0 --preload 2>&1 | tail -6
  echo "--- 산출물 ---"
  ls -la $D 2>/dev/null | tail -5
done
