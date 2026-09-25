# 논문 부록용 프롬프트 자료

    promptbox.tex                프리앰블용 상자 정의 (tcolorbox breakable + fvextra)
    prompt_construction_alg.tex  문항 구성 절차 -- Procedure 플로트 안의 수도코드 (algpseudocode)
    prompt_construction_prose.tex  같은 내용의 줄글 판 (두 문단, 프리앰블 불필요) -- 논문에는 이쪽을 쓴다
    prompts/robocasa_v22.txt     라벨링에 실제로 쓴 프롬프트 전문 (40줄)
    prompts/libero_v3d.txt       (35줄)
    prompts/humandata_v3.txt     (35줄)
    prompts/openarm_v2.txt       (41줄)
    prompts/allex_v4c.txt        (35줄)
    prompts/dexjoco_v3.txt       (38줄)
    sync_prompts.sh              vlm_gate/prompts/<벤치>_<판>_FULL.txt 에서 복사 · --check 로 대조

**정본은 `vlm_gate/prompts/*_FULL.txt` 다.** 그 폴더는 `scripts/sync_prompts_folder.py` 가
각 checks .py 의 GUIDANCE/ASK 에서 생성하고, 판정기가 실제로 받는 전문(SYSTEM +
이미지 + 지시문 + 계산 사실 + view_note + 문항)을 순서 그대로 담는다. 여기 사본은
Overleaf 에 폴더째 올리려고 둔 것이고, `output/` 이나 HF 를 따로 보지 않는다.

## 왜 이미지가 아니라 tex 인가

프롬프트 6종은 35~41줄 · 2.5~3.2 KB 라 그대로 조판된다(dexjoco 에 em-dash 두 개가
있을 뿐 나머지는 순수 ASCII). 이미지로 넣으면
검색·복사가 안 되고, 확대하면 흐려지고, 접근성 점검에서 감점이고, **무엇보다 실제
라벨링에 쓴 문구와 논문이 갈라진다.** `\VerbatimInput` 으로 원문 파일을 그대로 읽으면
갈라질 수가 없다 -- 프롬프트가 기여인 논문에서 이것이 가장 중요하다.

한 줄이 400~580자라 줄 접기가 필수다. 그 옵션(`breaklines`)은 fancyvrb 가 아니라
**fvextra** 에 있다. `breakanywhere` 는 쓰지 않는다 -- 글자 중간에서 끊으면 UTF-8
다바이트 문자가 바이트 단위로 쪼개져 컴파일이 죽는다(dexjoco 의 em-dash 에서 실제로
그랬다). 공백에서만 끊어도 3.3in 단폭(2단 논문 폭)에서 6종 모두 overfull 0 이고, 합쳐서
약 7쪽이다.

## Overleaf 에서

1. 이 디렉터리의 `promptbox.tex`, `prompt_construction_alg.tex`, `prompts/*.txt` 를
   프로젝트에 올린다. **`prompts/` 폴더 구조를 그대로** 둔다 -- `\VerbatimInput` 의
   경로는 main.tex 기준 상대경로다.
2. 프리앰블에 `\input{promptbox}` 와

       \usepackage{algorithm,algpseudocode}
       \usepackage{newfloat}
       \DeclareFloatingEnvironment[fileext=lop,listname={List of Procedures},
                                   name=Procedure,placement=tbp]{procedure}

   **Algorithm 이 아니라 Procedure 플로트다.** 사람이 문구를 고치고 VLM 이 채점하는
   절차라 알고리즘이라고 부르면 형식성을 과장한다. 캡션에 "pseudo-code of the
   protocol" 이라고 밝히고 수동 단계에 `[manual]` 을 붙였다. 본문에서는
   `Procedure~\ref{proc:prompt}` 로 가리킨다. 논문에 진짜 알고리즘이 있어도 번호가
   섞이지 않는다.
3. 부록에서

       \begin{promptbox}{RoboCasa (v22)}
         \VerbatimInput{prompts/robocasa_v22.txt}
       \end{promptbox}

   같은 상자를 데이터셋마다 하나씩, 그리고 `\input{prompt_construction_alg}`.

tcolorbox · fvextra · algpseudocode 는 모두 TeX Live 에 있어 Overleaf 에서 따로 설치할
것이 없다. txt 파일은 올려두기만 하면 컴파일 때 읽힌다.

## 프롬프트가 바뀌면

먼저 `python scripts/sync_prompts_folder.py` 로 정본 모음을 갱신하고, 그다음
`bash sync_prompts.sh` 로 여기를 맞춘다. `bash sync_prompts.sh --check` 가 어긋남을
잡는다. 여기 있는 txt 를 손으로 고치지 않는다.
