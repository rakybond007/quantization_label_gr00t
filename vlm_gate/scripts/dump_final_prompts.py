"""최종 프롬프트 전문을 부록 형태로 뽑는다. **손으로 옮겨 적지 않는다.**

문항 모듈에서 직접 읽으므로 코드와 부록이 어긋날 수 없다. 논문에 싣는 것은
이 산출물이고, 바뀌면 다시 돌린다.

    python dump_final_prompts.py > docs/appendix/FINAL_PROMPTS.md
"""
import hashlib
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BASE = os.path.dirname(HERE)

# (벤치마크, 모듈, 계산 사실을 만드는 함수의 위치, 한 줄 설명)
# 문구가 모듈에 없는 판은 텍스트 파일 경로를 적는다.
TEXT_FILE = {"libero_v3c_checks": "prompts/libero_v2.txt"}

FINAL = [
    ("robocasa", "phase9_checks_v7",
     "scripts/robocasa_descriptors.py: facts_text",
     "24 태스크 · 압축 여부 게이트. v1→v7, 네 판. 문구는 모듈 상수(GUIDANCE/ASK)에 "
     "있고 계산 사실은 함수가 청크마다 만든다 -- 그래서 이 부록에는 사실 예시가 "
     "따로 붙는다. libero 는 조립된 텍스트 파일 하나라 예시가 파일 안에 이미 있다."),
    ("libero", "libero_v3c_checks",
     "scripts/libero_checks.py: facts",
     "40 태스크 · 압축 여부 게이트. **문구가 checks 모듈에 없다** -- 모듈은 부호·가중치만 "
     "담고, 문항 본문은 analysis/_evolver/_libero/libero_questions_v3c.txt 와 "
     "libero_guidance_v3c.txt 에 있다. prompts/libero_v2.txt 는 v2 판이고 최신이 아니다. "
     "libero_v2_checks.py ~ libero_v3c_checks.py 가 바이트 단위로 같은 것은 그 모듈이 "
     "부호·가중치만 담기 때문이며, 문항이 안 바뀌었다는 뜻이 아니다 -- v3a 는 D, v3b 는 "
     "B 를 재조준했고 v3c 가 둘을 합쳤다."),
    ("allex", "allex_v4c_checks",
     "scripts/allex_v3_checks.py: facts_v3",
     "단일 배속 게이트. 서브태스크 라벨 없이 지시문만. v4→v4c, 세 판."),
]


def sha(s):
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def emit(bench, modname, facts_at, note):
    M = importlib.import_module(modname)
    Q = tuple(sorted(M.SIGN))
    print(f"## {bench}\n")
    print(f"{note}\n")
    print(f"- 모듈 `scripts/{modname}.py`")
    print(f"- 계산 사실 `{facts_at}`")
    print(f"- 등급 수 {M.NGRADE} · 문항 {len(Q)}개")
    risk = [q for q in Q if M.SIGN[q] < 0]
    gain = [q for q in Q if M.SIGN[q] > 0]
    print(f"- 감점 {''.join(risk)} (가중 합 {sum(M.WEIGHT[q] for q in risk):.2f}) · "
          f"가점 {''.join(gain)} (가중 합 {sum(M.WEIGHT[q] for q in gain):.2f})")
    nm = getattr(M, "NAME", {})
    print()
    print("| 문항 | 이름 | 부호 | 가중 |")
    print("|---|---|---|---|")
    for q in Q:
        print(f"| {q} | {nm.get(q, '-')} | {'감점' if M.SIGN[q] < 0 else '가점'} "
              f"| {M.WEIGHT[q]:.3f} |")
    g = getattr(M, "GUIDANCE", None)
    a = getattr(M, "ASK", None)
    if g:
        print(f"\n### GUIDANCE  (sha1 `{sha(g)}`)\n\n```\n{g}\n```")
    if a:
        print(f"\n### QUESTION  (sha1 `{sha(a)}`)\n\n```\n{a}\n```")
    if not (g and a):
        tf = TEXT_FILE.get(modname)
        body = open(f"{BASE}/{tf}").read().rstrip() if tf else ""
        print(f"\n### 프롬프트 전문  (`{tf}`, sha1 `{sha(body)}`)\n")
        print("문구가 모듈이 아니라 이 파일에 있다. 실제로 판정기에 나가는 형태 "
              "그대로이며, 맨 위 INSTRUCTION 블록이 계산 사실이다.\n")
        print(f"```\n{body}\n```")
    print("\n### conf 계산\n")
    print("```")
    print("g(등급) = (기댓값등급 - 1) / (NGRADE - 1)")
    print("conf    = (1 + Σ_가점 w·g - Σ_감점 w·g) / 2")
    print("```")
    print("기댓값 등급은 등급 토큰 자리의 softmax 분포에서 Σ (i+1)·p(i) 로 낸다.")
    print("정수 등급으로 계산하면 conf 가 계단이 되어 역치가 안 먹는다(하네스 R4).")
    print("\n---\n")


print("# 부록: 최종 VLM 프롬프트 전문\n")
print("`scripts/dump_final_prompts.py` 가 문항 모듈에서 직접 읽어 생성한다. "
      "손으로 옮겨 적지 않는다.\n")
print("판정기는 `nvidia/Cosmos3-Nano` 이고 그리디 디코딩으로 텍스트를 답한다. "
      "등급 토큰 자리의 로짓은 그 답이 얼마나 확실했는지를 덧붙이는 데만 쓰며, "
      "강제 슬롯 argmax 로 답을 짓지 않는다.\n")
print("프롬프트는 세 덩이로 조립된다 -- **계산 사실**(지시문 + 액션에서 계산한 "
      "사실 문장), **GUIDANCE**(무엇을 판단하는 일인지), **QUESTION**(등급 척도와 "
      "문항). 계산 사실은 프롬프트의 일부다. 빼고 물으면 다른 질문이 된다.\n")
print("---\n")
for row in FINAL:
    emit(*row)
