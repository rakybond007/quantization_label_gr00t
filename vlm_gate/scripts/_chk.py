import json, os
p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "analysis", "eval_results", "eval_results.json")
d = json.load(open(p))
for bench in ("libero", "robocasa"):
    r = d["runs"].get(bench, {})
    print(f"=== {bench}  실행 {len(r)}")
    for n in sorted(r):
        t = r[n]
        ex = "태스크별 O" if t.get("tasks") else "태스크별 X"
        print(f"  {n:36s} {t['success']:.4f}  {t.get('n_tasks')}태스크  {ex}")
