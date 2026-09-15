"""세 벤치마크의 최종 프롬프트를 `prompts/` 한 곳에 모은다.

왜 생성기인가
------------
문구가 벤치마크마다 다른 곳에 산다 -- robocasa 와 allex 는 모듈 상수, libero 는
`analysis/_evolver/_libero/` 의 텍스트 파일이다. 손으로 옮겨 모으면 원본이 바뀔 때
조용히 어긋나고, 실제로 그렇게 어긋난 적이 있다 (부록이 libero 최종을
`prompts/libero_v2.txt` 로 가리켰는데 그것은 v2 판이었다).

그래서 이 스크립트가 **원본에서 읽어** `prompts/` 를 다시 쓴다. 사람이 열어 보기
쉬운 사본이고, 권위는 언제나 원본에 있다. `--check` 로 어긋남을 검사한다.

    python scripts/sync_prompts_folder.py           # prompts/ 를 다시 쓴다
    python scripts/sync_prompts_folder.py --check    # 어긋나면 종료코드 1
"""
import argparse
import hashlib
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

OUT = os.path.join(ROOT, "prompts")

# 사실 예시는 데이터셋이 있어야 만들 수 있으므로 여기에 붙여 둔다. 각 줄 옆에
# 그것을 낸 함수를 적어 두어, 의심스러우면 직접 다시 낼 수 있게 한다.
FACTS_EXAMPLE = {
    "robocasa": (
        "# scripts/robocasa_descriptors.py: facts_text(descriptors(action, f))\n"
        "# phase9_two_sided.py:132 이 `instr + \"\\n\" + facts_text(x)` 로 조립한다.\n"
        "# robocasa_mg_gr00t_300 episode 0, frame 0 / frame 48 에서 실제로 나온 출력.\n"
        "\n"
        "MEASURED FROM THE PLANNED MOTION over the next ~1 second (these are computed "
        "facts, not estimates): the gripper stays open throughout; the end-effector keeps "
        "a consistent direction; it is among the slowest motions in this dataset -- "
        "faster than 3% of all moments here (mean step 0.17, peak 0.50); it is not "
        "creeping along with something held; it is not decelerating to a stop.\n"
        "\n"
        "MEASURED FROM THE PLANNED MOTION over the next ~1 second (these are computed "
        "facts, not estimates): the gripper stays open throughout; the end-effector keeps "
        "a consistent direction; it is on the fast side for this dataset -- faster than "
        "70% of all moments here (mean step 0.74, peak 0.95); it is not creeping along "
        "with something held; it is not decelerating to a stop.\n"
    ),
    "libero": (
        "# scripts/libero_checks.py: facts\n"
        "# 에피소드 지시문 한 줄이 먼저 오고 그 다음이 계산 사실이다.\n"
        "\n"
        "pick up the black bowl on the cookie box and place it on the plate\n"
        "MEASURED FROM THE PLANNED MOTION over the chunk ahead (these are computed facts, "
        "not estimates): the gripper stays closed throughout; the end-effector keeps a "
        "consistent direction; it is moving at a normal pace (mean step 0.42, peak 0.61); "
        "it is holding something while creeping along; it is decelerating to a near stop.\n"
    ),
    "allex": (
        "# scripts/allex_facts.py: facts(descriptors(A, WR, WL, f, 16))\n"
        "# 압축 요구는 한계 위반이 아니라 녹화 분포 대비로 말한다 -- 데이터셋에 없는 값이\n"
        "# 로봇의 한계를 뜻하지는 않기 때문이다.\n"
        "\n"
        "The robot was told: Bring the package over, orient barcode up, then place it on "
        "the conveyor.\n"
        "(계산 사실은 청크마다 다르다. allex_facts.py 를 직접 호출해 확인할 것.)\n"
    ),
}

BENCH = [
    # (이름, 판, 문구 출처, 부호·가중치 모듈, 설명)
    ("robocasa", "v7", ("module", "phase9_checks_v7"), "phase9_checks_v7",
     # 이미지 3장이다. scripts/phase9_two_sided.py:122 가 타일을 3등분해
     # gate.judge(views, ...) 로 넘긴다. build_messages 에 6장 분기가 있지만
     # robocasa 라벨링은 그 경로를 타지 않는다 -- 여기에 6 을 적었다가 실제로
     # 나가지 않는 "Images 1-3 are NOW / 4-6 are LATER" 문구를 전문에 싣는
     # 사고가 있었다. **장수는 라벨러 코드에서 확인하고 적는다.**
     "24 태스크 · 압축 여부 게이트", 3, "cosmos"),
    ("libero", "v3c",
     ("files", "analysis/_evolver/_libero/libero_guidance_v3c.txt",
      "analysis/_evolver/_libero/libero_questions_v3c.txt"),
     "libero_v3c_checks", "40 태스크 · 압축 여부 게이트", 3, "cosmos"),
    ("allex", "v4c", ("module", "allex_v4c_checks"), "allex_v4c_checks",
     "단일 배속 게이트 · 서브태스크 라벨 없이 지시문만", 2, "gemini_api"),
]


def sha(t):
    return hashlib.sha1(t.encode()).hexdigest()[:12]


def read_source(src):
    if src[0] == "module":
        m = importlib.import_module(src[1])
        return getattr(m, "GUIDANCE", "").rstrip("\n"), getattr(m, "ASK", "").rstrip("\n")
    g = open(os.path.join(ROOT, src[1])).read().rstrip("\n")
    q = open(os.path.join(ROOT, src[2])).read().rstrip("\n")
    return g, q


def assembled(name, ver, note, where, g, q, sign, weight, nm, ngrade, nviews, path):
    """판정기가 실제로 받는 순서 그대로 한 파일에 담는다.

    조립은 `scripts/vlm_gate.py: build_messages` 가 한다. 단순히
    사실+GUIDANCE+QUESTION 이 아니다 -- SYSTEM 이 먼저 있고, GUIDANCE 는 그 뒤에
    "Additional learned guidance" 로 붙고, 이미지 개수에 따라 view_note 가 또 붙고,
    사용자 쪽은 "Task: {지시문+사실}" + view_note + QUESTION 순서다. 쪼개 놓으면
    이 순서가 안 보여서, 합친본을 이 폴더의 대표 파일로 둔다.
    """
    tbl = "\n".join(
        f"  {k}  {'감점' if sign[k] < 0 else '가점'}  {weight.get(k, float('nan')):.3f}"
        f"  {nm.get(k, '')}" for k in sorted(sign))
    weights_block = f"""==================== 부호 · 가중치 ====================
등급 수 {ngrade} · 문항 {len(sign)}개

{tbl}

conf = (1 + Σ_가점 w·g - Σ_감점 w·g) / 2       g = (기댓값등급 - 1) / (등급수 - 1)
기댓값 등급은 등급 토큰 자리의 softmax 에서 Σ (i+1)·p(i). 정수 등급으로 계산하면
conf 가 계단이 되어 역치가 안 먹는다(하네스 R4). Gemini 경로는 logprob 을 못 받으므로
정수 등급을 그대로 쓴다 -- conf 가 계단이다.
"""
    facts_block = f"""==================== 계산 사실 (실제로 나간 형태) ====================
{FACTS_EXAMPLE[name].rstrip()}
"""

    if path == "gemini_api":
        # Gemini/Sonnet API 판정기는 vlm_gate 를 거치지 않는다. role:user 하나뿐이고
        # SYSTEM 블록도 view_note 도 없다 -- 과제 틀은 GUIDANCE 가 담당한다.
        # scripts/allex_litellm_run.py 의 조립을 그대로 따른다.
        return f"""# {name} {ver} -- {note}
# 판정기가 실제로 받는 순서 그대로 조립한 것이다.
# 조립 코드: scripts/allex_litellm_run.py (LiteLLM 프록시 경유 Gemini 3.8 Flash)
# 문구 원본: {where}
# 이 파일은 scripts/sync_prompts_folder.py 가 생성한다. 여기를 고치지 말 것.
#
# **SYSTEM 블록이 없다.** 이 경로는 vlm_gate 를 거치지 않으므로 vlm_gate.SYSTEM
# ("Answer with exactly one word: YES or NO.") 이 나가지 않는다. 과제 틀은 아래
# GUIDANCE 가 담당하고, 정답표를 주지 않는다 -- 그것이 이 경로의 설계다.
# robocasa/libero (Cosmos 경로) 는 SYSTEM 이 함께 나가며 그 SYSTEM 이 답 매핑을
# 유출한다 (docs/PROMPT_SYSTEM_LEAK.md).
#
# 이미지 {nviews}장이 아래 글 앞, 같은 user 메시지 안에 들어간다.

==================== [USER] 이미지 {nviews}장 + 아래 글 (메시지 하나) ====================
{g}

The two images are the left and right camera views of this one moment.

The robot was told: {{에피소드 지시문}}

{{계산 사실 -- 아래 절 참조}}

{q}

{facts_block}
{weights_block}"""

    import vlm_gate as VG
    sys_text = VG.SYSTEM
    if g:
        sys_text = (VG.SYSTEM + "\n\nAdditional learned guidance (from prior "
                    "evaluations):\n" + g.strip())
    # view_note 는 이미지 개수로 갈린다. build_messages 의 분기를 그대로 읽어 온다.
    import re as _re
    src = open(os.path.join(HERE, "vlm_gate.py")).read()
    body = src[src.index("def build_messages"):src.index("def parse_decision")]
    notes = _re.findall(r'view_note = \(?\s*("(?:[^"\\]|\\.)*"(?:\s*"(?:[^"\\]|\\.)*")*)\)?',
                        body)
    def lit(x):
        return "".join(_re.findall(r'"((?:[^"\\]|\\.)*)"', x)).replace('\\n', '\n')
    view_note = lit(notes[0 if nviews == 6 else (1 if nviews >= 3 else 2)]) if notes else "?"

    return f"""# {name} {ver} -- {note}
# 판정기가 실제로 받는 순서 그대로 조립한 것이다.
# 조립 코드: scripts/vlm_gate.py: build_messages
# 문구 원본: {where}
# 부호·가중치: 아래 맨 끝
# 이 파일은 scripts/sync_prompts_folder.py 가 생성한다. 여기를 고치지 말 것.
#
# 이미지 {nviews}장이 SYSTEM 다음, 아래 [USER] 글 앞에 들어간다.

==================== [SYSTEM] ====================
{sys_text}

==================== [USER] 이미지 {nviews}장 + 아래 글 ====================
Task: {{지시문 + 계산 사실 -- 아래 '계산 사실' 절 참조}}
{view_note}
{q}

==================== 계산 사실 (Task: 줄에 들어가는 실제 형태) ====================
{FACTS_EXAMPLE[name].rstrip()}

==================== 부호 · 가중치 ====================
등급 수 {ngrade} · 문항 {len(sign)}개

{tbl}

conf = (1 + Σ_가점 w·g - Σ_감점 w·g) / 2       g = (기댓값등급 - 1) / (등급수 - 1)
기댓값 등급은 등급 토큰 자리의 softmax 에서 Σ (i+1)·p(i). 정수 등급으로 계산하면
conf 가 계단이 되어 역치가 안 먹는다(하네스 R4).
"""


def build():
    files, rows = {}, []
    for name, ver, src, wmod, note, nviews, path in BENCH:
        g, q = read_source(src)
        M = importlib.import_module(wmod)
        sign, weight = getattr(M, "SIGN", {}), getattr(M, "WEIGHT", {})
        nm = getattr(M, "NAME", {})
        where = (f"scripts/{src[1]}.py 의 GUIDANCE / ASK" if src[0] == "module"
                 else f"{src[1]} · {src[2]}")
        head = (f"# {name} {ver} -- {note}\n"
                f"# 권위 있는 원본: {where}\n"
                f"# 이 파일은 scripts/sync_prompts_folder.py 가 생성한 사본이다. "
                f"여기를 고치지 말 것.\n\n")
        files[f"{name}_{ver}_FULL.txt"] = assembled(
            name, ver, note, where, g, q, sign, weight, nm,
            getattr(M, "NGRADE", "?"), nviews, path)
        files[f"{name}_{ver}_guidance.txt"] = head + g + "\n"
        files[f"{name}_{ver}_questions.txt"] = head + q + "\n"
        files[f"{name}_{ver}_facts_example.txt"] = FACTS_EXAMPLE[name]
        tbl = "\n".join(
            f"  {k}  {'감점' if sign[k] < 0 else '가점'}  {weight.get(k, float('nan')):.3f}"
            f"  {nm.get(k, '')}" for k in sorted(sign))
        files[f"{name}_{ver}_sign_weight.txt"] = (
            head + f"등급 수 {getattr(M, 'NGRADE', '?')} · 문항 {len(sign)}개\n\n" + tbl + "\n")
        rows.append((name, ver, where, sha(g), sha(q), wmod))
    return files, rows


def readme(rows):
    out = ["# 최종 VLM 프롬프트 (사본)", "",
           "`scripts/sync_prompts_folder.py` 가 원본에서 생성한다. **여기를 고치지 말 것** --",
           "권위는 아래 표의 원본에 있고, 이 폴더는 사람이 열어 보기 쉽게 모아둔 것이다.",
           "어긋남 검사: `python scripts/sync_prompts_folder.py --check`", "",
           "문구가 어디 사는지가 벤치마크마다 다르다. 그래서 한때 부록이 libero 최종을",
           "`prompts/libero_v2.txt`(v2 판)로 가리키는 사고가 있었다.", "",
           "| 벤치마크 | 판 | 권위 있는 원본 | GUIDANCE sha1 | QUESTION sha1 | 부호·가중치 |",
           "|---|---|---|---|---|---|"]
    for name, ver, where, gs, qs, wmod in rows:
        out.append(f"| {name} | {ver} | `{where}` | `{gs}` | `{qs}` | `scripts/{wmod}.py` |")
    out += ["", "## 먼저 이것만 열면 된다", "",
            "```",
            "robocasa_v7_FULL.txt      <- 판정기가 실제로 받는 전문, 순서 그대로",
            "libero_v3c_FULL.txt",
            "allex_v4c_FULL.txt",
            "```", "",
            "조립이 단순히 사실+GUIDANCE+QUESTION 이 아니다. `scripts/vlm_gate.py:",
            "build_messages` 가 이렇게 만든다:", "",
            "```",
            "[SYSTEM]  SYSTEM 상수",
            "          + \"Additional learned guidance (from prior evaluations):\" + GUIDANCE",
            "[USER]    이미지 N장",
            "          + \"Task: \" + 에피소드 지시문 + 계산 사실",
            "          + view_note  (이미지 개수로 갈린다: 6장 / 3장 / 그 외)",
            "          + QUESTION",
            "```", "",
            "**SYSTEM 과 view_note 를 빼고 보면 프롬프트를 잘못 읽는다.** SYSTEM 이 과제",
            "자체를 규정하고(YES/NO 압축 가능성), view_note 가 이미지 배치를 설명한다.",
            "합친본은 둘을 포함하고, 아래 조각 파일들은 포함하지 않는다.", "",
            "## 조각 파일 (원본과 1:1 로 대조할 때)", "",
            "```",
            "<벤치>_<판>_guidance.txt        GUIDANCE 만",
            "<벤치>_<판>_questions.txt       QUESTION 만 (등급 척도 + 문항)",
            "<벤치>_<판>_facts_example.txt   계산 사실이 실제로 나간 형태",
            "<벤치>_<판>_sign_weight.txt     문항별 부호와 가중치",
            "```", "",
            "계산 사실은 프롬프트의 일부다 -- 빼고 물으면 다른 질문이 된다.", "",
            "## 최종이 아닌 파일들 (역사·다른 실험)", "",
            "이 폴더에 먼저 있던 파일들이다. **어느 것도 최종 라벨링 프롬프트가 아니다.**",
            "이름이 판을 말해 주지 않아 실제로 혼동을 일으켰으므로 여기 적어 둔다.", "",
            "| 파일 | 무엇인가 |",
            "|---|---|",
            "| `libero_v2.txt` | libero **v2** 조립 전문. v3c 가 그 B·D 를 재조준했으므로 최신이 아니다 |",
            "| `robocasa_phase9.txt` | v7 GUIDANCE 는 맞지만 **문항이 v7 이 아니다** (앞선 판) |",
            "| `robocasa_phase9_direct.txt` | 자가집계 연구의 직접 질문 판 |",
            "| `robocasa_phase9_selfagg.txt` | 자가집계 연구: 관찰을 FOR/AGAINST 로 주고 모델이 합산 |",
            "| `robocasa_phase9_v3_selfagg.txt` | 같은 연구의 v3 판 |",
            "| `robocasa_phase9_v7_selfagg.txt` | 같은 연구의 v7 판 |", "",
            "자가집계 계열은 `_tmp/selfagg_bias/` 의 편향 연구용이다 -- 등급 문항으로",
            "라벨을 만든 경로와 **다른 질문 형태**다.", "",
            "최종은 위 표의 `<벤치>_<판>_*.txt` 뿐이다.", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="어긋나면 종료코드 1")
    a = ap.parse_args()
    files, rows = build()
    files["README.md"] = readme(rows)
    os.makedirs(OUT, exist_ok=True)
    bad = []
    for n, body in sorted(files.items()):
        p = os.path.join(OUT, n)
        cur = open(p).read() if os.path.exists(p) else None
        if cur == body:
            continue
        if a.check:
            bad.append(n)
        else:
            open(p, "w").write(body)
            print(f"  {'고침' if cur is not None else '새로'}  prompts/{n}")
    if a.check:
        if bad:
            print("원본과 어긋난 사본: " + ", ".join(bad))
            print("고치려면: python scripts/sync_prompts_folder.py")
            raise SystemExit(1)
        print(f"prompts/ {len(files)}개 파일 모두 원본과 일치")
    else:
        print(f"prompts/ {len(files)}개 파일 동기화 완료")


if __name__ == "__main__":
    main()
