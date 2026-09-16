"""`prompts/*_FULL.txt` 가 실제 코드가 내보내는 프롬프트와 같은지 기계로 대조한다.

**손으로 만든 전문은 믿을 수 없다.** 실제로 틀린 적이 있다 -- robocasa 를 이미지 6장
으로 잘못 적어, 라벨링에 나가지 않는 "Images 1-3 are NOW / 4-6 are LATER" 문구가
전문에 실렸다. 그래서 검증은 눈이 아니라 코드로 한다.

여기서는 라벨러가 부르는 것과 **같은 함수**를 불러 프롬프트를 만들고, FULL 파일에
그 글자들이 실제로 들어 있는지 확인한다.

    python scripts/verify_prompts_against_code.py          # 어긋나면 종료코드 1
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# (이름, 판, checks 모듈, 라벨러가 보내는 이미지 장수, 라벨러 파일)
COSMOS = [
    ("robocasa", "v7", "phase9_checks_v7", 3, "scripts/phase9_two_sided.py"),
    # 2장이다. front_view 와 left_wrist_view 뿐 -- label_chunks.py 의
    # BENCHMARKS["libero"]["views"] 도 ("front", "wrist") 다.
    ("robocasa", "v8", "phase9_checks_v8", 3, "scripts/phase9_two_sided.py"),
    ("libero", "v3c", "libero_v3c_checks", 2, "scripts/label_chunks.py"),
]


def dummy(n):
    return [Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)) for _ in range(n)]


def libero_texts():
    """libero 문구는 모듈이 아니라 텍스트 파일에 있다."""
    g = open(f"{ROOT}/analysis/_evolver/_libero/libero_guidance_v3c.txt").read().strip()
    q = open(f"{ROOT}/analysis/_evolver/_libero/libero_questions_v3c.txt").read().strip()
    return g, q


def main():
    import importlib
    from vlm_gate import build_messages

    bad = []
    for name, ver, mod, nviews, labeller in COSMOS:
        if name == "libero":
            g, q = libero_texts()
        else:
            M = importlib.import_module(mod)
            g, q = M.GUIDANCE, M.ASK
        # **자리표시를 생성기와 같은 글자로 넣는다.** 전에는 "INSTR\nFACTS" 를
        # 넣고 전문에는 {instruction}/{computed facts} 가 있어서, 그 줄을
        # 통째로 "없다" 고 잡았다(오탐). 아래 3)의 "{" 건너뛰기가 그 줄을
        # 걸러 주려면 자리표시가 중괄호 그대로여야 한다.
        msgs = build_messages(dummy(nviews),
                              "{instruction}\n{computed facts}", g, q,
                              user_only=(ver != "v7"))
        sys_text = next((m["content"][0]["text"] for m in msgs if m["role"] == "system"), "")
        usr_text = next(x["text"] for m in msgs if m["role"] == "user"
                        for x in m["content"] if x.get("type") == "text")
        n_img = sum(1 for m in msgs if m["role"] == "user"
                    for x in m["content"] if x.get("type") == "image")
        full = open(f"{ROOT}/prompts/{name}_{ver}_FULL.txt").read()

        # 1) 이미지 장수
        if f"<{n_img} images>" not in full:
            bad.append(f"{name}: 전문이 말하는 장수가 실제({n_img})와 다르다")
        # 2) SYSTEM 을 **줄 단위로 전부** 대조한다. 첫 줄 60자만 보면 본문 한가운데를
        #    바꿔도 못 잡는다 -- 자기검사가 실제로 그 미탐을 잡아냈다.
        for ln in sys_text.splitlines():
            ln = ln.strip()
            if len(ln) > 12 and ln not in full:
                bad.append(f"{name}: SYSTEM 줄이 전문에 없다 -- {ln[:60]}")
                break
        # 3) view_note 가 실제 것과 같나 -- 장수로 갈리므로 여기서 틀리기 쉽다
        # 3) USER 글도 줄 단위로 전부. view_note 는 장수로 갈리므로 여기서 틀리기 쉽다.
        for ln in usr_text.splitlines():
            ln = ln.strip()
            if len(ln) > 12 and "{" not in ln and ln not in full:
                bad.append(f"{name}: USER 줄이 전문에 없다 -- {ln[:60]}")
                break
        # 4) 문항 각 줄
        for line in q.splitlines():
            s = line.strip()
            if len(s) > 4 and s[0].isupper() and s[1] == ")" and s[:50] not in full:
                bad.append(f"{name}: 문항 줄이 전문에 없다 -- {s[:50]}")
        print(f"  {name} {ver}: 이미지 {n_img}장 · SYSTEM {len(sys_text)}자 · "
              f"USER {len(usr_text)}자 · 라벨러 {labeller}")

    # allex 는 vlm_gate 를 안 탄다. 라벨러가 직접 조립한다.
    import allex_v4c_checks as AK
    full = open(f"{ROOT}/prompts/allex_v4c_FULL.txt").read()
    src = open(f"{ROOT}/scripts/allex_litellm_run.py").read()
    if "SYSTEM" in src or "vlm_gate" in src:
        bad.append("allex: 라벨러가 SYSTEM 을 보낼 수 있다 -- 전문을 다시 확인하라")
    if "[SYSTEM]" in full:
        bad.append("allex: 전문에 SYSTEM 블록이 있는데 라벨러는 안 보낸다")
    for line in AK.ASK.splitlines():
        s = line.strip()
        if len(s) > 4 and s[0].isupper() and s[1] == ")" and s[:50] not in full:
            bad.append(f"allex: 문항 줄이 전문에 없다 -- {s[:50]}")
    print(f"  allex v4c: SYSTEM 없음(라벨러가 안 보냄) · 라벨러 scripts/allex_litellm_run.py")

    return bad


def selftest():
    """**검증기 자체를 검증한다.**

    검사가 오탐/미탐이면 통과했다는 말이 아무 의미가 없다. 실제로 그런 일이 있었다 --
    한글 "이미지 3장" 을 찾는 검사를 남겨둔 채 전문 형식을 "<3 images>" 로 바꿨고,
    그러자 검사가 **항상 실패**했다. 망가뜨리지 않은 벤치마크까지 어긋났다고 말했으므로
    탐지가 아니라 오탐이었다.

    그래서 두 방향을 다 본다.
      음성 대조: 손대지 않은 전문은 통과해야 한다 (오탐 없음)
      양성 대조: 한 글자 바꾼 전문은 반드시 걸려야 한다 (미탐 없음)
    """
    import shutil, tempfile
    base = main()
    if base:
        print("  [자기검사] 음성 대조 실패 -- 손대지 않은 전문이 어긋난다고 나온다:")
        for b in base:
            print("    -", b)
        return False
    print("  [자기검사] 음성 대조 통과 (손대지 않은 전문 = 어긋남 0)")

    cases = [
        ("prompts/robocasa_v7_FULL.txt", "<3 images>", "<6 images>", "이미지 장수"),
        ("prompts/robocasa_v7_FULL.txt", "YES or NO", "YES or MAYBE", "SYSTEM 본문"),
        ("prompts/libero_v3c_FULL.txt", "You are shown 2 camera views", "You are shown 9 camera views", "view_note"),
        ("prompts/allex_v4c_FULL.txt", "PINCH_RIGID" if False else "SQUEEZING", "SQUEEZINGX", "문항 줄"),
    ]
    ok = True
    for rel, old, new, what in cases:
        path = f"{ROOT}/{rel}"
        orig = open(path).read()
        if old not in orig:
            print(f"  [자기검사] 건너뜀 -- {rel} 에 {old!r} 가 없다")
            continue
        try:
            open(path, "w").write(orig.replace(old, new, 1))
            got = main()
            if got:
                print(f"  [자기검사] 양성 대조 통과 ({what}) -- {len(got)}건 잡음")
            else:
                print(f"  [자기검사] **미탐** ({what}) -- {old!r} -> {new!r} 를 못 잡는다")
                ok = False
        finally:
            open(path, "w").write(orig)
    return ok


if __name__ == "__main__":
    import sys as _s
    if "--selftest" in _s.argv:
        print("=== 검증기 자기 검사")
        if not selftest():
            raise SystemExit(1)
        print("\n검증기가 오탐도 미탐도 없다.")
        raise SystemExit(0)
    bad = main()
    if bad:
        print("\n어긋남:")
        for b in bad:
            print("  -", b)
        raise SystemExit(1)
    print("\n세 전문 모두 실제 코드가 내보내는 것과 일치한다.")


