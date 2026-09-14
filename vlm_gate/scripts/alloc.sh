#!/usr/bin/env bash
# tmux 창 하나(1:alloc)에 srun 할당을 잡아 두고 그 셸로 명령을 보낸다.
#
# 문항 깎기는 라벨->채점->수정->재라벨의 짧은 반복이다. 라운드마다 sbatch 를 던지면
# 큐 대기와 판정기 모델 로딩이 매번 붙는데 실제 작업은 5~20분이다. 할당을 붙잡아
# 두면 대기열에 한 번만 선다. **창은 하나만 쓴다.**
set -uo pipefail
W="${ALLOC_WIN:-1:alloc}"
D="$HOME/quantization_agent_workspace/vlm_gate/_tmp/alloc"; mkdir -p "$D"
case "${1:-}" in
up) shift
  tmux send-keys -t "$W" "srun --wckey=project-short-name:sub_fast --partition=debug \
${1:---gpus=1} --time=2:55:00 --pty bash -l" C-m
  echo "할당 요청 (창 $W)" ;;
run|bg) m=$1; shift; cmd="$*"
  n=$(date +%s%N); o="$D/$n.out"; r="$D/$n.rc"
  b=$(printf '%s' "$cmd" | base64 | tr -d '\n')
  tmux send-keys -t "$W" "{ eval \"\$(echo $b | base64 -d)\"; } > $o 2>&1; echo \$? > $r" C-m
  [ "$m" = bg ] && { echo "$o"; exit 0; }
  for _ in $(seq 1 5400); do [ -f "$r" ] && break; sleep 2; done
  cat "$o" 2>/dev/null; e=$(cat "$r" 2>/dev/null || echo 1); echo "[종료코드 $e]"; exit "$e" ;;
tail) tmux capture-pane -pt "$W" -S -"${2:-40}" ;;
where) tmux send-keys -t "$W" 'echo NODE=$(hostname) JOB=${SLURM_JOB_ID:-none}' C-m
  sleep 3; tmux capture-pane -pt "$W" -S -8 | grep -E "^NODE=" | tail -1 ;;
free) tmux send-keys -t "$W" C-d; echo "반납" ;;
*) sed -n '3,8p' "$0" ;;
esac
