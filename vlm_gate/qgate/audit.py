"""평가 실행 하나가 믿을 만한지 검사한다.

같은 검사를 두 번 손으로 했다 -- robocasa 에서 한 번, libero 에서 한 번. 세
번째가 오기 전에 도구로 만든다.

검사 넷이다.

    1. 중복 에피소드     같은 인덱스가 두 번 이상 나오는가
    2. 에피소드 수       태스크마다 기대한 수(보통 50)가 맞는가
    3. 헤더 복수         `Total success rate:` 가 파일당 하나인가
    4. 헤더 대조         헤더가 말하는 성공률과 에피소드 줄을 다시 센 값이 같은가

**넷째가 이 도구가 있는 이유다.** 선점·재개될 때마다 헤더가 덧붙는데 마지막
헤더가 전체를 반영한다는 보장이 없다. robocasa 에서 헤더를 읽고 계산했다가
"구동 한계를 풀면 압축 손상이 회복된다(t +2.37)" 라는 없는 결론이 나왔고,
에피소드 줄에서 다시 세니 사라졌다.

`evalscan` 은 이미 에피소드 줄만 읽으므로 안전하다. 이 도구는 **원본 파일이
성한지**를 보는 것이고, 남이 낸 결과를 처음 가져다 쓸 때 한 번 돌린다.

    python -m qgate.audit <실행 디렉터리> [<실행 디렉터리> ...] [--expect 50]
"""
import re
import sys
from collections import Counter
from pathlib import Path

_EP = re.compile(
    r"episode\s+(\d+)\s+is_success:\s*\[?\s*(True|False)\s*\]?", re.IGNORECASE)
# 탭으로 적는 형식도 있다: "  0\tTrue\t168"
_EP_TSV = re.compile(r"^\s*(\d+)\t(True|False)\b")
_HDR = re.compile(r"Total success rate:\s*([0-9.]+)")


def _episodes(path):
    """(에피소드 인덱스, 성공) 목록. 두 형식을 다 읽는다."""
    out = []
    for line in open(path, errors="replace"):
        m = _EP.search(line) or _EP_TSV.match(line)
        if m:
            out.append((int(m.group(1)), m.group(2).lower() == "true"))
    return out


def audit_run(root, expect=50):
    root = Path(root)
    files = sorted(set(list(root.glob("*/*_results.txt"))
                       + list(root.glob("*/*/*_results.txt"))
                       + list(root.glob("*/prediction.txt"))
                       + list(root.glob("*/*/prediction.txt"))))
    rep = {"root": str(root), "files": len(files), "episodes": 0, "success": 0,
           "dup": [], "wrong_n": [], "multi_hdr": [], "hdr_mismatch": []}
    for f in files:
        eps = _episodes(f)
        if not eps:
            continue
        idx = [i for i, _ in eps]
        c = Counter(idx)
        if any(v > 1 for v in c.values()):
            rep["dup"].append((str(f), [i for i, v in c.items() if v > 1][:5]))
        if expect and len(eps) != expect:
            rep["wrong_n"].append((str(f), len(eps)))
        hdrs = _HDR.findall(open(f, errors="replace").read())
        if len(hdrs) > 1:
            rep["multi_hdr"].append((str(f), len(hdrs)))
        if hdrs:
            counted = sum(s for _, s in eps) / len(eps)
            if abs(float(hdrs[-1]) - counted) > 5e-4:
                rep["hdr_mismatch"].append(
                    (str(f), float(hdrs[-1]), round(counted, 4)))
        rep["episodes"] += len(eps)
        rep["success"] += sum(s for _, s in eps)
    rep["rate"] = (rep["success"] / rep["episodes"]) if rep["episodes"] else None
    return rep


def _show(rep):
    bad = sum(len(rep[k]) for k in ("dup", "wrong_n", "multi_hdr", "hdr_mismatch"))
    rate = f"{rep['rate']:.4f}" if rep["rate"] is not None else "-"
    print(f"\n{rep['root']}")
    print(f"  파일 {rep['files']} · 에피소드 {rep['episodes']} · 성공률 {rate}"
          + ("   깨끗함" if not bad else f"   문제 {bad}건"))
    for key, label in (("dup", "중복 에피소드"), ("wrong_n", "에피소드 수 다름"),
                       ("multi_hdr", "헤더 복수"), ("hdr_mismatch", "헤더와 재계산 불일치")):
        for row in rep[key][:6]:
            print(f"    [{label}] {row}")
        if len(rep[key]) > 6:
            print(f"    [{label}] ... 외 {len(rep[key]) - 6}건")
    return bad


def main(argv):
    expect = 50
    roots = []
    it = iter(argv)
    for a in it:
        if a == "--expect":
            expect = int(next(it))
        else:
            roots.append(a)
    if not roots:
        print(__doc__)
        return 2
    bad = 0
    for r in roots:
        bad += _show(audit_run(r, expect))
    print(f"\n{'전부 깨끗함' if not bad else f'문제 총 {bad}건 -- 그 실행은 쓰기 전에 확인할 것'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
