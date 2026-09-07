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


if __name__ == "__main__":
    for T in (16, 20):
        print(f"T={T}")
        for r in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
            b, _ = blocks_for(T, r)
            lens = [e - s for s, e in b]
            print(f"  요청 {r:>4}  ->  첫 청크 블록 {len(b):>2}개  길이 {lens}")
            print(f"          20청크 이어서 실제 {realised_ratio(T, r, 20):.4f}")
