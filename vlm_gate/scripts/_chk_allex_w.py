import os, sys
os.environ.setdefault("ALLEX_CHECKS", "CLAMP,LOOSE,SHOVE,FLIP,FREE")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import allex_v3_checks as C
print("ACTIVE", C.ACTIVE)
neg = sum(C.WEIGHT[q] for q in C.ACTIVE if C.SIGN[q] < 0)
pos = sum(C.WEIGHT[q] for q in C.ACTIVE if C.SIGN[q] > 0)
print(f"감점합 {neg:.3f}  가점합 {pos:.3f}")
for q in C.ACTIVE:
    print(f"  {q:6s} {'감점' if C.SIGN[q]<0 else '가점'} {C.WEIGHT[q]:.3f}")
print("\nconfidence 는 side() 에서 **걸린 것들의 가중평균**이라 합이 1 이 아니어도 된다:")
print("  sum(W*v) / sum(W)  -- 분모가 걸린 문항의 무게 합이다")
