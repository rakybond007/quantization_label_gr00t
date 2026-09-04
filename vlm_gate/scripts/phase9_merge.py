"""앞선 워커와 역순 워커의 라벨을 합친다. 겹치는 것은 (ep, f) 로 걸러낸다.

두 쌍이 같은 목록을 양끝에서 걸어왔으므로 만나는 지점의 에피소드 하나가
양쪽에 있을 수 있다. 행이 (ep, f) 로 키가 잡혀 있어 마지막 것만 남기면 된다 --
같은 청크를 두 번 물어도 배치 폭이 같으면 같은 답이 나온다.

    python phase9_merge.py [out.jsonl]
"""
import glob
import json
import os
import sys
import collections

OUT = os.path.expanduser(os.environ.get(
    "PHASE9_OUT", "~/quantization_agent_workspace/vlm_gate/output/_gate_distill/phase9_full"))
DEST = sys.argv[1] if len(sys.argv) > 1 else f"{OUT}/labels_merged.jsonl"

seen, dup, bad = {}, 0, 0
per_file = {}
for p in sorted(glob.glob(f"{OUT}/labels_*.jsonl")):
    if p.endswith("labels_merged.jsonl"):
        continue
    n = 0
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            bad += 1
            continue
        k = (r["ep"], r["f"])
        if k in seen:
            dup += 1
        seen[k] = r
        n += 1
    per_file[os.path.basename(p)] = n
    print(f"  {os.path.basename(p):<24} {n:>9,}행")

with open(DEST, "w") as fh:
    for k in sorted(seen):
        fh.write(json.dumps(seen[k], ensure_ascii=False) + "\n")

full = sum(1 for r in seen.values() if all(r.get(q) is not None for q in "ABCDE"))
grades = collections.Counter(r.get("A") for r in seen.values())
print(f"\n  읽은 행 {sum(per_file.values()):,}   겹침 {dup:,}   깨진 줄 {bad}")
print(f"  고유 (ep, f)  {len(seen):,}")
print(f"  다섯 문항 다 파싱된 비율  {100*full/max(1,len(seen)):.1f}%")
print(f"  A 등급 분포  " + "  ".join(f"{g}:{100*n/len(seen):.1f}%" for g, n in sorted(grades.items()) if g))
print(f"  -> {DEST}")
