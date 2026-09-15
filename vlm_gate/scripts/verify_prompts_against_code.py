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
    ("libero", "v3c", "libero_v3c_checks", 3, "scripts/label_chunks.py"),
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
        msgs = build_messages(dummy(nviews), "INSTR\nFACTS", g, q)
        sys_text = next((m["content"][0]["text"] for m in msgs if m["role"] == "system"), "")
        usr_text = next(x["text"] for m in msgs if m["role"] == "user"
                        for x in m["content"] if x.get("type") == "text")
        n_img = sum(1 for m in msgs if m["role"] == "user"
                    for x in m["content"] if x.get("type") == "image")
        full = open(f"{ROOT}/prompts/{name}_{ver}_FULL.txt").read()

        # 1) 이미지 장수
        if f"이미지 {n_img}장" not in full:
            bad.append(f"{name}: 전문이 말하는 장수가 실제({n_img})와 다르다")
        # 2) SYSTEM 전문이 통째로 들어 있나
        if sys_text and sys_text.split("\n")[0][:60] not in full:
            bad.append(f"{name}: SYSTEM 첫 줄이 전문에 없다")
        # 3) view_note 가 실제 것과 같나 -- 장수로 갈리므로 여기서 틀리기 쉽다
        note = [l for l in usr_text.splitlines() if l.startswith("You are shown")]
        if note and note[0][:70] not in full:
            bad.append(f"{name}: view_note 가 실제와 다르다 (실제: {note[0][:70]}...)")
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

    if bad:
        print("\n어긋남:")
        for b in bad:
            print("  -", b)
        raise SystemExit(1)
    print("\n세 전문 모두 실제 코드가 내보내는 것과 일치한다.")


if __name__ == "__main__":
    main()
