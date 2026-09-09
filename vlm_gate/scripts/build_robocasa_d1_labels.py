"""Validate immutable raw annotation and write the canonical D1 parquet."""
import argparse
import heapq
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from robocasa_d1_reader import Reader, digest
from robocasa_d1_policy import CATEGORIES, select_ratio


def raw_rows(path):
    prev = (-1,-1)
    with Path(path).open() as f:
        for line in f:
            row = json.loads(line)
            key = (row['episode_index'],row['frame_index'])
            if key <= prev: raise ValueError(f'unsorted/duplicate raw key {path}: {key}')
            prev = key
            yield row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--raw',nargs='+',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    ds=Reader(a.dataset)
    manifest=json.loads(Path(a.manifest).read_text())
    if ds.fingerprint != manifest['dataset_fingerprint']: raise ValueError('dataset mismatch')
    fingerprint=None
    for path in a.raw:
        meta=json.loads(Path(path).with_suffix('.meta.json').read_text())
        if meta['manifest_sha256'] != digest(a.manifest): raise ValueError('raw/manifest mismatch')
        semantic={k:v for k,v in meta.items() if k not in ('shard','num_shards')}
        if fingerprint is not None and semantic != fingerprint: raise ValueError('raw semantic mismatch')
        fingerprint=semantic
    fields=[pa.field('episode_index',pa.int32()),pa.field('frame_index',pa.int32()),pa.field('index',pa.int64()),
            pa.field('task',pa.string()),pa.field('instruction',pa.string())]
    for q in 'ABCD':
        fields.extend([pa.field(q,pa.string()),pa.field('gp_'+q,pa.list_(pa.float64(),6)),pa.field('eg_'+q,pa.float64())])
    for name in ('compression_score','risk_score','support_score','score_low','score_high','task_cap','ratio','ratio_before_contact'):
        fields.append(pa.field(name,pa.float64()))
    fields.extend([pa.field('ratio_valid',pa.bool_()),pa.field('legacy_fixed',pa.bool_()),
                   pa.field('ratio_class',pa.int8()),pa.field('ratio_reason',pa.string()),
                   pa.field('ratio_label',pa.list_(pa.float32(),2)),pa.field('grade_prob_label',pa.list_(pa.float32(),28))])
    schema=pa.schema(fields)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists(): raise ValueError('refusing to overwrite canonical labels')
    temporary=out.with_suffix('.partial.parquet')
    expected=(iter(map(tuple,manifest['rows'])) if manifest['rows'] is not None else
              ((ep,f) for ep,m in sorted(ds.episodes.items()) for f in range(0,m['length']-16,manifest['stride'])))
    merged=heapq.merge(*(raw_rows(x) for x in a.raw),key=lambda r:(r['episode_index'],r['frame_index']))
    count=0; buffer=[]; counts={str(k):0 for k in (1.,1.5,2.,2.5)}; invalid=0
    with pq.ParquetWriter(temporary,schema,compression='zstd') as writer:
        for row in merged:
            key=(row['episode_index'],row['frame_index'])
            if key != next(expected,None): raise ValueError(f'missing/extra/duplicate key: {key}')
            meta=ds.episodes[key[0]]
            if row['task'] != meta['tasks'][1] or row['instruction'] != meta['tasks'][0]: raise ValueError('task mismatch')
            picks=[row[q] for q in 'ABCD']; probs=row['category_probs']
            result=select_ratio(probs,picks,row['task_cap'],row['legacy_fixed'])
            for k in ('ratio','ratio_class','ratio_valid','ratio_reason'):
                if result[k] != row[k]: raise ValueError(f'policy recomputation differs: {key} {k}')
            record={f.name:row.get(f.name) for f in fields}
            for i,q in enumerate('ABCD'):
                record['gp_'+q]=probs[i]
                record['eg_'+q]=result['expected_grade_conditional_known'][i] if not result['unknown'][i] else None
            record['ratio_label']=[float(result['ratio_class']),float(result['ratio_valid'])]
            record['grade_prob_label']=[v for prob in probs for v in prob]+[1.]*4
            buffer.append(record); count+=1; counts[str(row['ratio'])]+=1; invalid+=not row['ratio_valid']
            if len(buffer)>=8192:
                writer.write_table(pa.Table.from_pylist(buffer,schema=schema)); buffer=[]
        if buffer: writer.write_table(pa.Table.from_pylist(buffer,schema=schema))
    if next(expected,None) is not None or count != manifest['eligible_rows']: raise ValueError('incomplete raw coverage')
    temporary.replace(out)
    report=dict(rows=count,semantic_invalid=invalid,ratio_counts=counts,grade_schema='d1v2_4x6_soft',
                category_order=CATEGORIES,source_manifest=manifest,raw_fingerprint=fingerprint,
                canonical_sha256=digest(out))
    out.with_suffix('.manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_manifest','raw_fingerprint')},indent=2))


if __name__=='__main__': main()
