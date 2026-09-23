# 논문 부록용 프롬프트 자료

    promptbox.tex                프리앰블용 상자 정의 (tcolorbox breakable + fvextra)
    prompt_construction_alg.tex  문항 구성 절차 수도알고리즘 (algorithm + algpseudocode, 30줄)
    prompts/robocasa_v22.txt     라벨링에 실제로 쓴 프롬프트 전문 (40줄)
    prompts/libero_v3d.txt       (35줄)
    prompts/humandata_v3.txt     (35줄)
    prompts/openarm_v2.txt       (41줄)
    sync_prompts.sh              vlm_gate/output/*/PROMPT.txt 에서 위 4개를 다시 복사

## 왜 이미지가 아니라 tex 인가

프롬프트 4종은 35~41줄 · 2.6~3.2 KB · 순수 ASCII 라 그대로 조판된다. 이미지로 넣으면
검색·복사가 안 되고, 확대하면 흐려지고, 접근성 점검에서 감점이고, **무엇보다 실제
라벨링에 쓴 문구와 논문이 갈라진다.** `\VerbatimInput` 으로 원문 파일을 그대로 읽으면
갈라질 수가 없다 -- 프롬프트가 기여인 논문에서 이것이 가장 중요하다.

한 줄이 400~580자라 줄 접기가 필수다. 그 옵션(`breaklines`)은 fancyvrb 가 아니라
**fvextra** 에 있다. `promptbox.tex` 가 fvextra 를 부른다. 3.3in 단폭(2단 논문 폭)에서
컴파일해 overfull 0 을 확인했다.

## Overleaf 에서

1. 이 디렉터리의 `promptbox.tex`, `prompt_construction_alg.tex`, `prompts/*.txt` 를
   프로젝트에 올린다. **`prompts/` 폴더 구조를 그대로** 둔다 -- `\VerbatimInput` 의
   경로는 main.tex 기준 상대경로다.
2. 프리앰블에 `\input{promptbox}` 와 `\usepackage{algorithm,algpseudocode}`.
3. 부록에서

       \begin{promptbox}{RoboCasa (v22)}
         \VerbatimInput{prompts/robocasa_v22.txt}
       \end{promptbox}

   같은 상자를 데이터셋마다 하나씩, 그리고 `\input{prompt_construction_alg}`.

tcolorbox · fvextra · algpseudocode 는 모두 TeX Live 에 있어 Overleaf 에서 따로 설치할
것이 없다. txt 파일은 올려두기만 하면 컴파일 때 읽힌다.

## 프롬프트가 바뀌면

`bash sync_prompts.sh` 로 다시 복사한다. 여기 있는 txt 를 손으로 고치지 않는다.
