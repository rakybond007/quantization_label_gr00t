"""서브태스크 상한을 다시 잡아 allex conf 라벨을 다시 굽는다. VLM 재실행 없음.

배달본은 청크마다 `p`(1단 신뢰도)와 `K_max`(서브태스크 사전상한을 2단 문항이
청크별로 조인 값)를 같이 담고 있고, 라벨은

    K    = snap(1 + p*(K_max - 1))          grid = [1, 2, 2.5, 3]
    hojin = clip((K - 1) / (3 - 1), 0, 1)

로 만들어졌다. 이 관계는 배달본에서 **99.885% 재현된다**(최근접 스냅). 따라서
`K_max` 만 옮기면 나머지는 다시 계산된다.

상한을 옮길 때 청크별 조임을 뭉개면 안 된다 -- 2단 문항이 청크마다 다르게 조인
결과가 `K_max` 의 분포다. 그래서 **비율로** 옮긴다. 서브태스크 L 의 현재 상한을
C_L(그 서브태스크 `K_max` 의 최댓값), 새 목표를 T_L 이라 하면

    K_max'(청크) = 1 + (K_max(청크) - 1) * (T_L - 1) / (C_L - 1)

이면 그 서브태스크 안의 상대 순서와 조임 비율이 그대로 보존되고 꼭대기만 T_L 로
간다. T_L = C_L 이면 항등이다.

    python allex_rescale_ceiling.py <입력.json> <출력.json> --ceiling "Rotate Box=2.0,Pass Object=2.5"
    python allex_rescale_ceiling.py <입력.json> <출력.json> --scale 0.8      # 전 서브태스크 일괄

**주의.** 이 스크립트는 사전상한만 옮긴다. 2단 문항이 쓴 물리 한계
(merge_limit_rad 등)는 그 녹화본에서 보정된 값이고 다시 계산하지 않는다.
입력 파일은 절대 덮어쓰지 않는다.
"""
import argparse, collections, datetime, json, sys

GRID_DEFAULT = [1.0, 2.0, 2.5, 3.0]
# 눈금을 늘린 격자. 상한을 내리면 원값 1+p*(K_max-1) 이 1.5 를 못 넘는 청크가
# 많아지는데, 현행 격자는 1 과 2 사이가 비어 있어 그 전부가 1.0 으로 떨어진다
# (Rotate Box 는 상한 1.75 에서 99.7% 가, Rotate PolyBag 은 100% 가 1.0 이 됐다).
# 1.5 · 1.75 눈금이 그 구간을 받는다. **쓰는 쪽이 분수 배속을 실행할 수 있어야
# 한다.** hojin = (K-1)/(max(grid)-1) 은 연속값이라 그대로 동작한다.
GRID_FINE = [1.0, 1.5, 1.75, 2.0, 2.5, 3.0]

# scripts/allex_v2_common.py 의 현행 명세 (2026-09-05 운용자 값). 배달본은
# 2026-09-01 생성이라 이 개정 **이전** 판이다 -- 그래서 재조정이 필요하다.
# 2026-09-14 운용자 지시값.
# Rotate PolyBag 은 3.0. 운용자 판단은 "봉투는 빨리 뒤집어도 된다" 인데 1단
# 신뢰도가 거꾸로 나왔다(p 평균 0.189 로 네 서브태스크 중 최저, Rotate Box 는
# 0.376). 상한을 열어 순서를 바로잡는다 -- p 자체는 건드리지 않는다.
SPEC_CEILING = {"Bring Object": 2.0, "Pass Object": 2.5,
                "Rotate Box": 1.75, "Rotate PolyBag": 3.0}
# 하한은 전 서브태스크 1.5. (allex_v2_common.py 는 Rotate Box 만 1.5 이고
# 나머지 기본이 2.0 이지만, 운용자 지시로 전부 1.5 로 맞춘다.)
SPEC_FLOOR = {}
DEFAULT_FLOOR = 1.5


def snap(x, grid):
    return min(grid, key=lambda g: (abs(g - x), g))


def hojin_of(k, kref):
    return max(0.0, min(1.0, (k - 1.0) / (kref - 1.0)))


def rebuild(chunks, grid, kref):
    """저장된 p·K_max 로 K 와 hojin 을 다시 만든다."""
    for c in chunks:
        c["K"] = snap(1.0 + c["p"] * (c["K_max"] - 1.0), grid)
        c["hojin"] = hojin_of(c["K"], kref)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--ceiling", default="",
                    help='"Rotate Box=2.0,Pass Object=2.5" 형태. 안 적은 서브태스크는 그대로.')
    ap.add_argument("--scale", type=float, default=None,
                    help="전 서브태스크의 (상한-1) 에 곱할 값. --ceiling 과 같이 못 쓴다.")
    ap.add_argument("--spec", action="store_true",
                    help="allex_v2_common.py 의 현행 상한·하한 명세를 그대로 적용한다.")
    ap.add_argument("--floor", default="",
                    help='"Rotate Box=1.5,Bring Object=2.0" 형태. 하한 미만인 K_max 를 끌어올린다.')
    ap.add_argument("--grid", default="",
                    help='K 가 스냅되는 격자. "fine" = [1,1.5,1.75,2,2.5,3], '
                         '"1,1.5,2" 처럼 직접 줘도 된다. 기본은 파일에 적힌 격자.')
    ap.add_argument("--verify-only", action="store_true",
                    help="상한을 안 바꾸고 저장된 K 가 공식으로 재현되는지만 본다.")
    a = ap.parse_args()
    if a.ceiling and a.scale is not None:
        sys.exit("--ceiling 과 --scale 은 같이 못 쓴다")

    d = json.load(open(a.src))
    C = d["chunks"]
    grid_src = list(d.get("params", {}).get("grid", GRID_DEFAULT))
    kref = max(grid_src)      # hojin 정규화 기준. 격자를 늘려도 안 바꾼다.
    grid = grid_src

    # 1) 재현 검증. **파일에 적힌 격자로** 한다 -- 늘린 격자로 검증하면
    #    당연히 안 맞는다. 원본이 그 격자로 만들어졌는지부터 확인하는 단계다.
    ok = sum(abs(snap(1.0 + c["p"] * (c["K_max"] - 1.0), grid_src) - c["K"]) < 1e-9
             for c in C)
    print(f"[검증] 저장된 K 재현율 {ok}/{len(C)} = {ok/len(C):.4%}")
    if ok / len(C) < 0.99:
        sys.exit("재현율이 99% 미만이다. 공식이나 grid 가 다르다 -- 멈춘다.")
    if a.verify_only:
        return
    if a.grid:
        grid = (list(GRID_FINE) if a.grid.strip() == "fine"
                else sorted(float(x) for x in a.grid.split(",")))
        if max(grid) > kref:
            sys.exit(f"격자 최댓값 {max(grid)} 이 hojin 기준 {kref} 를 넘는다")
        print(f"[격자] {grid_src} -> {grid}  (hojin 기준 {kref} 고정)")

    by = collections.defaultdict(list)
    for c in C:
        by[c["label"]].append(c)
    cur = {lb: max(r["K_max"] for r in rows) for lb, rows in by.items()}

    tgt = dict(cur)
    flr = {}
    if a.spec:
        for lb in cur:
            if lb not in SPEC_CEILING:
                sys.exit(f"명세에 없는 서브태스크 {lb!r} -- --spec 을 못 쓴다")
            tgt[lb] = SPEC_CEILING[lb]
            flr[lb] = SPEC_FLOOR.get(lb, DEFAULT_FLOOR)
    if a.scale is not None:
        tgt = {lb: 1.0 + (v - 1.0) * a.scale for lb, v in cur.items()}
    for part in filter(None, (x.strip() for x in a.ceiling.split(","))):
        lb, _, v = part.rpartition("=")
        lb = lb.strip()
        if lb not in cur:
            sys.exit(f"모르는 서브태스크 {lb!r}. 있는 것: {sorted(cur)}")
        tgt[lb] = float(v)
    for part in filter(None, (x.strip() for x in a.floor.split(","))):
        lb, _, v = part.rpartition("=")
        lb = lb.strip()
        if lb not in cur:
            sys.exit(f"모르는 서브태스크 {lb!r}. 있는 것: {sorted(cur)}")
        flr[lb] = float(v)
    for lb in cur:
        if flr.get(lb, 0.0) > tgt[lb]:
            sys.exit(f"{lb}: 하한 {flr[lb]} 이 상한 {tgt[lb]} 보다 크다")

    before = collections.Counter(c["K"] for c in C)
    mean_before = {lb: sum(r["K"] for r in rows) / len(rows)
                   for lb, rows in by.items()}
    for lb, rows in by.items():
        c0, t0 = cur[lb], tgt[lb]
        if abs(c0 - 1.0) < 1e-9:      # 상한이 이미 1 이면 옮길 여지가 없다
            continue
        f = (t0 - 1.0) / (c0 - 1.0)
        lo = flr.get(lb)
        for r in rows:
            v = 1.0 + (r["K_max"] - 1.0) * f
            if lo is not None:
                v = max(lo, v)          # 하한: 2단이 과하게 조인 칸을 끌어올린다
            r["K_max"] = min(v, t0)
    rebuild(C, grid, kref)
    after = collections.Counter(c["K"] for c in C)

    print(f"\n{'서브태스크':22s} {'현상한':>7s} {'새상한':>7s} {'하한':>6s} {'평균K 전':>8s} {'후':>6s}")
    for lb in sorted(by):
        ks = [c["K"] for c in by[lb]]
        lo = flr.get(lb)
        print(f"  {lb:20s} {cur[lb]:7.3f} {tgt[lb]:7.3f} "
              f"{('-' if lo is None else f'{lo:.2f}'):>6s} "
              f"{mean_before[lb]:8.3f} {sum(ks)/len(ks):6.3f}")
    print(f"\nK 분포 전: {dict(sorted(before.items()))}")
    print(f"K 분포 후: {dict(sorted(after.items()))}")

    d["k_stats"] = {"n_chunks": len(C),
                    "distribution_pct": {str(k): round(100 * v / len(C), 2)
                                         for k, v in sorted(after.items())},
                    "mean_K": round(sum(c["K"] for c in C) / len(C), 4)}
    d["overall_mean"] = round(sum(c["hojin"] for c in C) / len(C), 4)
    d.setdefault("params", {})["rescaled_ceilings"] = {k: round(v, 4) for k, v in tgt.items()}
    if flr:
        d["params"]["rescaled_floors"] = {k: round(v, 4) for k, v in flr.items()}
    d["params"]["rescaled_from"] = {k: round(v, 4) for k, v in cur.items()}
    if grid != grid_src:
        d["params"]["grid"] = grid
        d["params"]["grid_before_rescale"] = grid_src
    d["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    d["description"] = (d.get("description", "").split(" | rescaled")[0]
                        + " | rescaled: per-subtask prior ceiling moved, "
                          "per-chunk clamp ratio preserved (allex_rescale_ceiling.py)")
    json.dump(d, open(a.dst, "w"), ensure_ascii=False)
    print(f"\n평균 hojin {d['overall_mean']} · 평균 K {d['k_stats']['mean_K']} -> {a.dst}")


if __name__ == "__main__":
    main()
