"""Task index -> language instruction, for every LIBERO suite.

Evaluation writes its results into directories named `<suite>_<task_idx>`, and
the labels carry only the instruction sentence. Nothing on disk joins the two,
so a label set cannot be scored against the measured per-task compression cost
the way RoboCasa's can — RoboCasa's episodes.jsonl happens to carry a task class
token, LIBERO's does not.

The mapping lives in the benchmark package, so it has to be enumerated once and
written down. Output: analysis/libero_task_index.json
    {"libero_10": {"0": "...", ...}, "libero_goal": {...}, ...}
"""
import json
import os

from libero.libero import benchmark

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
OUT = os.path.expanduser(
    "~/quantization_agent_workspace/vlm_gate/analysis/libero_task_index.json")

bd = benchmark.get_benchmark_dict()
out = {}
for s in SUITES:
    suite = bd[s]()
    out[s] = {}
    for i in range(suite.n_tasks):
        t = suite.get_task(i)
        out[s][str(i)] = t.language
    print(f"{s}: {suite.n_tasks} tasks", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"저장 {OUT}  총 {sum(len(v) for v in out.values())} 태스크")
