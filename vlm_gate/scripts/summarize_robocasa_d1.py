"""Diagnostic pilot report, never an automatic semantic approval."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np


def read(path):
    return {(r['episode_index'],r['frame_index']):r for r in map(json.loads,Path(path).read_text().splitlines())}


def main():
    p=argparse.ArgumentParser();p.add_argument('--labels',required=True);p.add_argument('--compare');p.add_argument('--out',required=True)
    a=p.parse_args(); rows=read(a.labels);bytask=defaultdict(list)
    for row in rows.values(): bytask[row['task']].append(row)
    report=dict(rows=len(rows),tasks={},semantic_review_passed=False,
                notice='Ratios and grade histograms are diagnostics, not success rates or acceptance quotas.')
    for task,items in sorted(bytask.items()):
        report['tasks'][task]=dict(rows=len(items),ratios=dict(Counter(str(r['ratio']) for r in items)),
            semantic_invalid=sum(not r['ratio_valid'] for r in items),contact_fixed=sum(r['legacy_fixed'] for r in items),
            grades={q:dict(Counter(r[q] for r in items)) for q in 'ABCD'})
    if a.compare:
        other=read(a.compare);keys=sorted(rows.keys()&other.keys())
        report['batch_comparison']=dict(rows=len(keys),
            category_disagreements=sum(rows[k][q]!=other[k][q] for k in keys for q in 'ABCD'),
            ratio_flips=sum(rows[k]['ratio']!=other[k]['ratio'] for k in keys),
            max_probability_difference=max((float(np.abs(np.asarray(rows[k]['category_probs'])-np.asarray(other[k]['category_probs'])).max()) for k in keys),default=0))
    Path(a.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='tasks'},indent=2))


if __name__=='__main__': main()
