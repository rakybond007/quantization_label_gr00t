# 최종 VLM 프롬프트 (사본)

`scripts/sync_prompts_folder.py` 가 원본에서 생성한다. **여기를 고치지 말 것** --
권위는 아래 표의 원본에 있고, 이 폴더는 사람이 열어 보기 쉽게 모아둔 것이다.
어긋남 검사: `python scripts/sync_prompts_folder.py --check`

문구가 어디 사는지가 벤치마크마다 다르다. 그래서 한때 부록이 libero 최종을
`prompts/libero_v2.txt`(v2 판)로 가리키는 사고가 있었다.

| 벤치마크 | 판 | 권위 있는 원본 | GUIDANCE sha1 | QUESTION sha1 | 부호·가중치 |
|---|---|---|---|---|---|
| robocasa | v7 | `scripts/phase9_checks_v7.py 의 GUIDANCE / ASK` | `c3ba33024565` | `9b446e769f82` | `scripts/phase9_checks_v7.py` |
| libero | v3c | `analysis/_evolver/_libero/libero_guidance_v3c.txt · analysis/_evolver/_libero/libero_questions_v3c.txt` | `053844846ac9` | `697bf64ffaca` | `scripts/libero_v3c_checks.py` |
| allex | v4c | `scripts/allex_v4c_checks.py 의 GUIDANCE / ASK` | `733d71476881` | `6fc571516e1d` | `scripts/allex_v4c_checks.py` |

## 파일

벤치마크마다 네 개다.

```
<벤치>_<판>_guidance.txt        무엇을 판단하는 일인지
<벤치>_<판>_questions.txt       등급 척도와 문항
<벤치>_<판>_facts_example.txt   계산 사실이 실제로 나간 형태
<벤치>_<판>_sign_weight.txt     문항별 부호와 가중치
```

프롬프트는 **계산 사실 + GUIDANCE + QUESTION** 세 덩이로 조립된다.
계산 사실은 프롬프트의 일부다 -- 빼고 물으면 다른 질문이 된다.

## 최종이 아닌 파일들 (역사·다른 실험)

이 폴더에 먼저 있던 파일들이다. **어느 것도 최종 라벨링 프롬프트가 아니다.**
이름이 판을 말해 주지 않아 실제로 혼동을 일으켰으므로 여기 적어 둔다.

| 파일 | 무엇인가 |
|---|---|
| `libero_v2.txt` | libero **v2** 조립 전문. v3c 가 그 B·D 를 재조준했으므로 최신이 아니다 |
| `robocasa_phase9.txt` | v7 GUIDANCE 는 맞지만 **문항이 v7 이 아니다** (앞선 판) |
| `robocasa_phase9_direct.txt` | 자가집계 연구의 직접 질문 판 |
| `robocasa_phase9_selfagg.txt` | 자가집계 연구: 관찰을 FOR/AGAINST 로 주고 모델이 합산 |
| `robocasa_phase9_v3_selfagg.txt` | 같은 연구의 v3 판 |
| `robocasa_phase9_v7_selfagg.txt` | 같은 연구의 v7 판 |

자가집계 계열은 `_tmp/selfagg_bias/` 의 편향 연구용이다 -- 등급 문항으로
라벨을 만든 경로와 **다른 질문 형태**다.

최종은 위 표의 `<벤치>_<판>_*.txt` 뿐이다.
