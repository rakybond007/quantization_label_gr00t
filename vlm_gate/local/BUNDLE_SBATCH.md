# 제출 방식이 바뀌었다 — `bundle-sbatch`

`sbatch` 를 직접 부르지 않는다. `bundle-sbatch` 가 감싼다. 출처는 인프라팀
`bundle-sbatch quick guide v0.1` 이고 원문은 사용자 `~/Downloads` 에 있다.
저장소 코드는 https://github.com/RLWRLD/bundle-sbatch 다.

## 무엇이 달라졌나

**실행마다 달라지는 설정은 스크립트가 아니라 명령줄에 둔다.**

    #SBATCH 에서 뺀다        --job-name · --wckey · --partition · --time
    #SBATCH 에서 지운다      --output · --error · --open-mode
                             --export · --export-file · --get-user-env
                             --parsable · --quiet · --wait · --test-only
    남기는 것은 긴 형식으로   --nodes=1 · --gpus=2   (`-N 1` · `--gpus 1` 안 됨)

**배열은 감싸개만 만든다.** `#SBATCH --array` 도, 슬럼 구역의 `--array=...` 도,
물려받은 `SBATCH_ARRAY_INX` 도 전부 금지다. `--array SPEC` 을 첫 `--` 앞에 한
번만 준다(`--array=SPEC` 아님, 붙임표 없음).

**`MODEL_OUTPUT_DIR` 을 직접 export 하지 않는다.** 감싸개가 넣어 주고
`CODE_OUTPUT_DIR` 과 같은 값이다. `--export` 로 주는 것도 금지다.

## 모양

    bundle-sbatch \
      --job-kind eval \
      --code-git-root $HOME/quantization_agent_workspace \
      --checkpoint <실제 디렉터리, 심링크 안 됨> \
      --array 0-7 \
      -- \
      --job-name='<50자 이상>' \
      --wckey=project-short-name:sub_fast \
      --partition=background \
      --time=12:00:00 \
      -- \
      ./그스크립트.sbatch 인자들

첫 `--` 가 감싸개 옵션의 끝, 둘째 `--` 가 슬럼 옵션의 끝이고 그 뒤가 스크립트다.
슬럼 옵션 값은 **붙임표 긴 형식**이어야 한다(`--name=value`).

## 잡 종류와 체크포인트

    train          --checkpoint 필수. 새로 학습이면 `from_scratch`,
                   이어서면 실제 체크포인트 디렉터리
    eval           --checkpoint 필수. 실제 체크포인트 디렉터리
    data_process   --checkpoint 금지
    other          --checkpoint 금지

체크포인트는 **실재하는 디렉터리**여야 하고 심링크는 거부된다. 관리 루트 밖에
있으면 `.cache/huggingface/download/` 아래 내려받기 메타데이터가 온전해야 한다.

## 배열 작업이 해야 하는 일

감싸개는 부모 출력 뿌리만 준다. **각 작업이 자기 번호를 검사하고, 자기 디렉터리를
만들고, 확인하고, 두 변수를 다시 묶은 뒤에** 쓰기 시작해야 한다.

    task_id=${SLURM_ARRAY_TASK_ID:?}
    case "$task_id" in 0|1|2|3|4|5|6|7) ;; *) exit 2 ;; esac
    task_dir="${CODE_OUTPUT_DIR}/${task_id}"
    mkdir -p -- "$task_dir"
    mode=$(stat -c %a -- "$task_dir")
    if ! test -d "$task_dir" || test -L "$task_dir" || ! test -O "$task_dir" \
       || test "$mode" != 755; then exit 2; fi
    export CODE_OUTPUT_DIR="$task_dir"
    export MODEL_OUTPUT_DIR="$task_dir"

## 되돌리면 안 되는 것

**`submission_started` 가 찍힌 뒤에는 같은 요청을 다시 내지 않는다.** `sbatch` 가
거부했다고 나와도, 잡 번호가 없어도, 결과가 불분명해도 마찬가지다. 슬럼과 남은
번들을 들여다보고 나온 진단을 따른다. **이것은 우리 규칙(먼저 들어간 잡을 함부로
취소하지 않는다)과 같은 방향이다.**

배열 제출이 0 으로 끝나고 출력이 온전하면 **부모 잡 번호 하나**가 나온다.
`123_7` 같은 작업 모양 출력은 파싱된 것이 아니다.

## GPU 파티션은 GPU 잡만 받는다

찌르다 나온 규칙이다. `--gpus` 없이 `background` 로 내면 거부된다.

    sbatch: error: GPU 파티션에는 GPU를 요청한 잡만 제출할 수 있습니다.
                   CPU 전용 잡은 --partition=cpu 를 사용하세요.

라벨 후처리·변환처럼 CPU 만 쓰는 것은 `--partition=cpu` 다.

## 우리 쪽에서 덜 바뀌는 것

파티션 규칙은 그대로다 — 학습 `sjw_alinlab` · 평가·라벨링 `background` ·
시험 `debug`. job-name 50자 이상도 그대로다(가이드가 같은 말을 한다).

`srun` · `salloc` · 대화형은 감싸개 범위 밖이라 **예전 방식 그대로** 쓴다.
스모크를 `srun` 으로 잡아 두고 그 안에서 돌리는 방식은 안 바뀐다.

## 출력이 어디로 가나

번들 저장소는 **관리 저장소이지 영구 저장소가 아니다.** 보존 정책이 옮기고
나중에 지운다. 오래 두어야 할 산출물은 사용자 저장소로 옮긴다.
