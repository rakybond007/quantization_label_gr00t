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
     "24 태스크 · 압축 여부 게이트"),
    ("libero", "v3c",
     ("files", "analysis/_evolver/_libero/libero_guidance_v3c.txt",
      "analysis/_evolver/_libero/libero_questions_v3c.txt"),
     "libero_v3c_checks", "40 태스크 · 압축 여부 게이트"),
    ("allex", "v4c", ("module", "allex_v4c_checks"), "allex_v4c_checks",
     "단일 배속 게이트 · 서브태스크 라벨 없이 지시문만"),
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


def build():
    files, rows = {}, []
    for name, ver, src, wmod, note in BENCH:
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
    out += ["", "## 파일", "",
            "벤치마크마다 네 개다.", "",
            "```",
            "<벤치>_<판>_guidance.txt        무엇을 판단하는 일인지",
            "<벤치>_<판>_questions.txt       등급 척도와 문항",
            "<벤치>_<판>_facts_example.txt   계산 사실이 실제로 나간 형태",
            "<벤치>_<판>_sign_weight.txt     문항별 부호와 가중치",
            "```", "",
            "프롬프트는 **계산 사실 + GUIDANCE + QUESTION** 세 덩이로 조립된다.",
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
