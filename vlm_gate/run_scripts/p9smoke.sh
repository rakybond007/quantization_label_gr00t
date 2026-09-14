set -u
E=$HOME/quantization_agent_workspace/quantization_label_gr00t/vlm_gate/experiments/proprio_history
for A in proprio baseline; do
  D=$E/artifacts/p9smoke_$A
  rm -rf $D
  echo "########## $A ##########"
  bash $E/run_phase9.sh $D $A --max-tasks 8 --episodes-per-task 6 --frames-per-episode 32 \
       --epochs 4 --bs 64 --num-workers 0 --preload 2>&1 | tail -8
done
