"""분수 배속을 위한 블록 경계.

고정 K 압축은 길이 K 인 블록을 균일하게 깐다. 그래서 K 는 정수여야 하고,
배속이 1x · 2x · 3x 로만 뛴다. 손상표를 그 세 칸으로만 만들면 태스크마다
"어디서 깨지는가" 를 못 잡는다 -- allex 는 운용자가 1.5·2·2.5 를 다 재서
태스크별 띠를 갖고 있는데 시뮬은 못 갖는다.

블록 길이는 정수여야 하지만 **평균**은 정수가 아니어도 된다. 길이 1 과 2 를
섞어 깔면 평균 1.5 가 나온다. 배속 r 은 들어간 스텝 수 나누기 나온 스텝 수이므로,
그 평균이 곧 배속이다.

    T=16, r=1.5  ->  블록 10개, 길이가 2 여섯 · 1 넷      16/10 = 1.60
    T=16, r=2.5  ->  블록  6개, 길이가 3 넷 · 2 둘        16/6  = 2.67

길이를 몰아 놓지 않고 고르게 흩는 것이 중요하다. `2 2 2 2 2 2 1 1 1 1` 은 앞쪽
절반만 압축한 것이라 그 청크의 배속이 균일하지 않다. Bresenham 과 같은 방식으로
누적 오차를 굴려 `2 1 2 2 1 2 2 1 2 1` 처럼 흩는다.

집계 방식은 건드리지 않는다. robocasa 는 델타라 블록을 더하고 dexjoco 는 절대
목표라 블록의 마지막을 쓴다 -- 그건 그대로 두고 경계만 바꾸므로 양쪽에 같이 쓴다.

    from fractional_blocks import blocks_for
    blocks_for(16, 2.0)   # [(0,2), (2,4), ...]        정수면 기존과 같다
    blocks_for(16, 1.5)   # 길이 1 과 2 가 섞인 열 개

## 분할기가 둘이다 -- 이름만 보고 못 가린다

**둘은 서로 다른 실행 경로를 모사한다. 무엇을 재는지에 따라 골라야 한다.**

    blocks_for(T, r, carry)   분수 배속 경로(`frac_compress_chunk`).
                              꼬리를 블록에 섞고 잔여를 다음 청크로 넘겨
                              **요청 배속을 실현한다.**
    block_sizes(horizon, k)   옛 고정 K 경로(`compress_chunk`). 남는 꼬리를
                              길이 1 로 낱개로 붙인다. 그래서 **요청값보다
                              낮은 배속이 나온다.**

    요청      block_sizes 실현    blocks_for 실현
     1.5          1.455              1.50
     2.0          2.000              2.00
     2.5          2.286              2.50
     3.0          2.667              2.99
     4.0          4.000              4.00

정수에서는 둘이 같다. 갈리는 것은 분수뿐이다.

`RATIOS_ARE_NOT_K.md` 의 "K=3 은 실제로 2.67" 이 `block_sizes` 쪽 값이다.
사다리의 `baseline_compress_K1p5` · `K2p5` 는 1.5 · 2.5 로 기록돼 있으므로
분수 경로로 돈 것이고, 배속 라벨의 눈금(1.0·1.5·2.0·2.5)도 그쪽이다.

**옛 고정 K 실행을 모사할 때만 `block_sizes` 를 쓴다.** 라벨 눈금이 실제로
어떻게 실행되는지를 재는 자리에 그것을 쓰면 실행되지 않는 경계를 재게 된다.
`chord_error.py` 가 처음에 그렇게 짜였고 자체검사에서 걸렸다.
"""


def blocks_for(T, r, carry=0.0):
    """청크 사이로 잔여를 넘기는 판. (블록, 다음 carry) 를 돌려준다.

    청크가 16스텝이라 짧아서 한 청크 안에서는 1.5 를 정확히 못 만든다(11블록이면
    1.455, 10블록이면 1.60). 압축은 에피소드 내내 반복되므로, 이번에 덜 압축한
    만큼을 다음 청크가 갚으면 **에피소드 전체 배속**이 요청값에 수렴한다.

    carry 는 "지금까지 냈어야 할 출력 스텝 수 − 실제로 낸 수" 다.
    """
    if T <= 0:
        return [], carry
    if r <= 1.0:
        return [(i, i + 1) for i in range(T)], 0.0
    want = T / float(r) + carry          # 이번에 내야 할 출력 스텝 수
    n = max(1, min(T, int(round(want))))
    return _split(T, n), want - n


def _split(T, n):
    """길이 T 를 n 개 블록으로. 몫과 나머지를 앞뒤로 몰지 않고 흩는다.

    남는 꼬리를 길이 1 로 따로 두지 않고 블록에 섞는다 -- 꼬리를 raw 로 두면
    그 청크의 실제 배속이 요청값보다 낮아진다.
    """
    if n >= T:
        return [(i, i + 1) for i in range(T)]
    base, rem = divmod(T, n)
    out, start, acc = [], 0, 0
    for i in range(n):
        acc += rem
        extra = 0
        if acc >= n:                            # 누적 오차가 한 칸을 채우면 길이를 하나 늘린다
            acc -= n
            extra = 1
        end = start + base + extra
        out.append((start, end))
        start = end
    if start != T:                              # 반올림이 어긋나면 마지막이 흡수한다
        out[-1] = (out[-1][0], T)
    return out


def realised_ratio(T, r, chunks=1):
    """청크를 chunks 개 이어 붙였을 때 실제로 나오는 배속."""
    carry, tot_in, tot_out = 0.0, 0, 0
    for _ in range(chunks):
        b, carry = blocks_for(T, r, carry)
        tot_in += T
        tot_out += len(b)
    return tot_in / tot_out if tot_out else 1.0


# ---------------------------------------------------------------------------
# F_level 과 같은 규칙. `papers/reproducing/FLARE/flare/action_head_flevel.py`
# 의 `block_sizes` · `replan_rows` 를 그대로 옮긴 것이다. 배속별 디코더가 이
# 경계로 학습되므로, 평가의 균일 배속도 같은 경계를 써야 사다리 칸이 디코더
# 칸과 같은 것을 가리킨다.
#
# 위의 `blocks_for` 는 청크 사이로 잔여를 넘겨 평균을 맞추는 판이라 창마다
# 행 수가 달라진다. 평가에는 쓸 수 있지만 **디코더에는 못 쓴다** -- 디코더는
# 출력 개수가 고정이고 직전 창의 잔여를 알 수 없다. 결합을 할 것이므로
# 아래 두 함수를 쓴다.
# ---------------------------------------------------------------------------


def block_sizes(horizon, k):
    """16스텝 청크를 K 로 묶은 블록 길이. 소수 K 는 floor 와 ceil 을 교대로 놓는다.

        K=1    [1]*16
        K=1.5  [1,2,1,2,1,2,1,2,1,2,1]   11행
        K=2    [2]*8
        K=2.5  [2,3,2,3,2,3,1]            7행
        K=3    [3,3,3,3,3,1]              6행
        K=4    [4]*4
    """
    kf = float(k)
    k = int(kf) if kf == int(kf) else kf
    if k <= 1:
        return [1] * horizon
    lo = int(k)
    hi = lo if isinstance(k, int) else lo + 1
    pattern = [lo] if lo == hi else [lo, hi]
    out, cur, j = [], 0, 0
    while cur < horizon:
        n = pattern[j % len(pattern)]
        if cur + n > horizon:
            break
        out.append(n)
        cur += n
        j += 1
    out += [1] * (horizon - cur)
    return out


def replan_rows(sizes, replan_steps):
    """실행할 행 수. 앞에서부터 더한 원본 스텝 수가 `replan_steps` 에 **가장 가까운**
    행 수를 고른다. 같은 거리면 더 많이 실행하는 쪽.

    5 이하로만 자르면 6스텝을 못 써서 K=1.5 가 실제 1.27 로 내려앉는다. 가장
    가까운 쪽을 고르면 창이 4~6스텝으로 흔들리는 대신 배속이 정확히 떨어진다:

        replan 5 에서  K=1 -> 5행(5)  1.5 -> 4행(6)  2 -> 3행(6)
                       2.5 -> 2행(5)  3 -> 2행(6)    4 -> 1행(4)
    """
    best, best_d, cum = 1, None, 0
    for r, n in enumerate(sizes, start=1):
        cum += n
        d = abs(cum - replan_steps)
        if best_d is None or d <= best_d:
            best, best_d = r, d
        if cum >= replan_steps:
            break
    return best


def replan_rows_carry(sizes, replan_steps, carry=0.0):
    """실행할 행 수를 **장부**로 고른다. `(행 수, 다음 carry)` 를 돌려준다.

    `replan_rows` 는 매번 replan_steps 에 가장 가까운 하나로 고정이라, K 마다
    재예측 주기가 4·5·6 으로 갈린다. 그러면 "level 2 를 많이 쓴 에피소드" 가
    압축 이득과 재예측을 덜 한 이득을 같이 받아서 둘을 못 가른다.

    carry 는 "지금까지 replan_steps 씩 썼어야 할 양 − 실제로 쓴 양" 이다.
    앞만 보므로 추론에서도 쓸 수 있다 -- 다음에 어느 level 이 뽑힐지 알 필요가
    없고, level 이 정해진 뒤에 이 함수를 부르면 된다.

        K=1.5  6,4,6,4,...   재예측 평균 5, 배속 10/7 = 1.429
        K=2    6,4,6,4,...   재예측 평균 5, 배속 10/5 = 2.000
        K=3    6,3,6,...     재예측 평균 5, 배속 15/5 = 3.000
        K=4    4,8,4,4,...   재예측 평균 5, 배속 20/5 = 4.000

    `replan_rows` 와 달리 K=2.5 만 빼면 창 길이가 번갈아 든다. 배속은 K=1.5 만
    1.429 이고(블록이 [1,2] 교대라 홀수 행에서 반 스텝이 어긋난다) 나머지는
    요청값에 정확히 떨어진다.
    """
    want = replan_steps + carry
    best, best_d, best_raw = 1, None, sizes[0] if sizes else 1
    cum = 0
    for r, n in enumerate(sizes, start=1):
        cum += n
        d = abs(cum - want)
        if best_d is None or d <= best_d:      # 같으면 더 많이 실행하는 쪽
            best, best_d, best_raw = r, d, cum
        if cum >= want + max(sizes):           # 더 가 봐야 멀어지기만 한다
            break
    return best, want - best_raw


if __name__ == "__main__":
    for T in (16, 20):
        print(f"T={T}")
        for r in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
            b, _ = blocks_for(T, r)
            lens = [e - s for s, e in b]
            print(f"  요청 {r:>4}  ->  첫 청크 블록 {len(b):>2}개  길이 {lens}")
            print(f"          20청크 이어서 실제 {realised_ratio(T, r, 20):.4f}")
