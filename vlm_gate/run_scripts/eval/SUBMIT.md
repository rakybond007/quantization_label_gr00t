# 내는 법 — `bundle-sbatch`

`sbatch` 를 직접 부르지 않는다. 규칙 전문은 `vlm_gate/local/BUNDLE_SBATCH.md`.

실행마다 달라지는 것(`--job-name` · `--wckey` · `--partition` · `--time` ·
`--array`)은 **스크립트가 아니라 여기 명령줄에** 있다. 스크립트에는 어디서
돌든 같은 것만 남는다(`--nodes` · `--gpus` · `--exclude` · `--requeue`).

## 게이트 평가 (GPU 2 — VLA + 판정기)

```bash
CK=$HOME/multigpu_workspace/Isaac-GR00T/ckpt/robocasa/groot/groot_n1_5_bs64_baseline/checkpoint-60000

MODE=stage1 bundle-sbatch \
  --job-kind eval \
  --code-git-root $HOME/quantization_agent_workspace \
  --checkpoint "$CK" \
  --array 0-7 \
  -- \
  --job-name='gate_eval_graded_vlm_grades_stage1_robocasa_24tasks_50ep_arr8' \
  --wckey=project-short-name:sub_fast \
  --partition=background \
  --time=12:00:00 \
  -- \
  ./vlm_gate/run_scripts/eval/gate_eval_graded.sh
```

`MODE=stage2` 로 한 번 더. **`--array` 는 붙임표 없이, 첫 `--` 앞에 한 번만.**

## 균일 대조군 (GPU 1 — 판정기를 안 씀)

**게이트 판의 실측 `rate_mean` 을 넣는다.** 예상값이 아니다.

```bash
RATE=1.99 bundle-sbatch \
  --job-kind eval \
  --code-git-root $HOME/quantization_agent_workspace \
  --checkpoint "$CK" \
  --array 0-7 \
  -- \
  --job-name='uniform_control_matched_mean_ratio_robocasa_24tasks_50ep_arr8' \
  --wckey=project-short-name:sub_fast \
  --partition=background \
  --time=12:00:00 \
  -- \
  ./vlm_gate/run_scripts/eval/uniform_control.sh
```

## 걸리는 자리

- **체크포인트는 실재하는 디렉터리여야 하고 심링크는 거부된다.** `eval` 은 필수다
- **GPU 파티션은 GPU 를 요청한 잡만 받는다.** CPU 만 쓰는 것은 `--partition=cpu`
- **`MODEL_OUTPUT_DIR` 을 직접 주지 않는다.** 감싸개가 넣는다
- job-name 50자 이상
- **`submission_started` 뒤에는 같은 요청을 다시 내지 않는다.** 거부로 보여도 그렇다

## 출력이 어디로 가나

감싸개가 `CODE_OUTPUT_DIR` 을 주고 각 배열 작업이 그 아래 `<번호>/` 를 만든다.
스크립트가 그것을 `OUTPUT_BASE` 로 쓴다.

**번들 저장소는 관리 저장소다.** 보존 정책이 옮기고 나중에 지운다. 결과를 오래
두려면 사용자 저장소로 옮긴다.
