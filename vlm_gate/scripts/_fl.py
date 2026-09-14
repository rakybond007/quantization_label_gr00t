from pathlib import Path
import numpy as np
D = Path("/sjw_alinlab2/home/taekwan/Data/flevel_libero_157729_ckpt60000")
SU = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
def load(root):
    o = {}
    for f in sorted(Path(root).glob("*/*_results.txt")):
        n = s = st = 0; sn = 0
        for line in open(f):
            q = line.strip().split("\t")
            if len(q) >= 2 and q[1].strip() in ("True", "False"):
                n += 1
                if q[1].strip() == "True":
                    s += 1
                    if len(q) >= 3:
                        try: st += float(q[2]); sn += 1
                        except ValueError: pass
        if n: o[f.parent.name + "/" + f.stem.split("_")[0]] = (s, n, st, sn)
    return o
arms = [d.name for d in sorted(D.iterdir()) if d.is_dir()]
print("%-22s %8s %8s %8s %8s %8s %7s" % ("arm", *SU, "전체", "성공스텝"))
for a in arms:
    d = load(D / a)
    if not d: continue
    cells = []
    for su in SU:
        ks = [k for k in d if k.startswith(su)]
        cells.append(sum(d[k][0] for k in ks) / max(1, sum(d[k][1] for k in ks)))
    tot = sum(d[k][0] for k in d) / max(1, sum(d[k][1] for k in d))
    stp = sum(d[k][2] for k in d) / max(1, sum(d[k][3] for k in d))
    print("%-22s %8.3f %8.3f %8.3f %8.3f %8.3f %7.1f" % (a, *cells, tot, stp))
