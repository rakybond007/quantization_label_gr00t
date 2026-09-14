import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
from libero.libero import benchmark
bd = benchmark.get_benchmark_dict()
for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
    b = bd[suite]()
    for i in range(b.n_tasks):
        t = b.get_task(i)
        print("%s\t%d\t%s" % (suite, i, t.language))
