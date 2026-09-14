import sys
sys.path.insert(0, "vlm_gate/scripts")
from fractional_blocks import block_sizes, replan_rows, replan_rows_carry
print("%5s | %-7s %-7s | %-7s %-7s | %s" % (
    "K", "고정행", "고정배속", "장부주기", "장부배속", "장부 창 길이"))
for k in (1, 1.5, 2, 2.5, 3, 4):
    sz = block_sizes(16, k)
    r0 = replan_rows(sz, 5); raw0 = sum(sz[:r0])
    carry, R, O, seq = 0.0, 0, 0, []
    for _ in range(400):
        r, carry = replan_rows_carry(sz, 5, carry)
        raw = sum(sz[:r]); R += raw; O += r
        if len(seq) < 8: seq.append(raw)
    print("%5s | %-7s %-7.3f | %-7.3f %-7.3f | %s" % (
        k, "%d행=%d" % (r0, raw0), raw0 / r0, R / 400, R / O, seq))
