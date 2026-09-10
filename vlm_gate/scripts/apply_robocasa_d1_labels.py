"""Portable numpy/pyarrow D1 overlay. Source data is never modified."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True); p.add_argument('--labels',required=True)
    p.add_argument('--out',required=True); p.add_argument('--dry-run',action='store_true')
    p.add_argument('--mode',choices=['sidecar','parquet'],default='sidecar')
    p.add_argument('--pilot',action='store_true',help='allow explicitly sparse pilot, never a full training release')
    a=p.parse_args(); root=Path(a.dataset); labels=Path(a.labels); out=Path(a.out)
    report=json.loads(labels.with_suffix('.manifest.json').read_text())
    manifest=report['source_manifest']
    if digest(labels)!=report['canonical_sha256']: raise ValueError('canonical checksum mismatch')
    for name,sha in manifest['dataset_fingerprint'].items():
        if digest(root/'meta'/name)!=sha: raise ValueError(f'dataset fingerprint mismatch {name}')
    if manifest['mode']=='pilot' and not a.pilot: raise ValueError('pilot requires explicit --pilot')
    if not isinstance(manifest['stride'],int) or manifest['stride']<1: raise ValueError('invalid stride')
    episodes={r['episode_index']:r for r in map(json.loads,(root/'meta/episodes.jsonl').read_text().splitlines())}
    info=json.loads((root/'meta/info.json').read_text())
    table=pq.read_table(labels)
    ep=table['episode_index'].to_numpy(); frame=table['frame_index'].to_numpy()
    order=np.lexsort((frame,ep))
    if not np.array_equal(order,np.arange(len(ep))): raise ValueError('unsorted labels')
    if len(ep)>1 and np.any((ep[1:]==ep[:-1])&(frame[1:]==frame[:-1])): raise ValueError('duplicate labels')
    if set(ep)-set(episodes): raise ValueError('unknown episodes')
    ratio=np.asarray(table['ratio_label'].to_pylist(),dtype=np.float32)
    grade=np.asarray(table['grade_prob_label'].to_pylist(),dtype=np.float32)
    global_indices=table['index'].to_numpy()
    if ratio.shape!=(len(ep),2) or grade.shape!=(len(ep),28) or not np.isfinite(grade).all(): raise ValueError('bad carriers')
    if not np.isin(ratio[:,0],[0,1,2,3]).all() or not np.isin(ratio[:,1],[0,1]).all(): raise ValueError('bad ratio')
    if (grade[:,:24]<0).any() or not np.allclose(grade[:,:24].reshape(-1,4,6).sum(-1),1,atol=1e-4): raise ValueError('bad probabilities')
    if not np.all(grade[:,24:]==1): raise ValueError('raw annotation question validity mismatch')
    caps={}; wrote=0; total=0
    if not a.dry_run:
        out.mkdir(parents=True,exist_ok=True)
        if any(out.iterdir()): raise ValueError('output must be an empty directory')
    for episode,meta in sorted(episodes.items()):
        lo,hi=np.searchsorted(ep,[episode,episode+1]); fs=frame[lo:hi]; n=meta['length']
        if manifest['mode']=='full' and not np.array_equal(fs,np.arange(0,n-16,manifest['stride'])): raise ValueError(f'coverage ep{episode}')
        if np.any(fs<0) or np.any(fs>=n-16): raise ValueError('label outside full context')
        base_path=root/info['data_path'].format(episode_chunk=episode//info['chunks_size'],episode_index=episode)
        identity=pq.read_table(base_path,columns=['episode_index','index'])
        if len(identity)!=n or not np.all(identity['episode_index'].to_numpy()==episode): raise ValueError('base identity mismatch')
        if not np.array_equal(identity['index'].to_numpy()[fs],global_indices[lo:hi]): raise ValueError('global/local index mismatch')
        tasks=table['task'].slice(int(lo),int(hi-lo)).to_pylist()
        instructions=table['instruction'].slice(int(lo),int(hi-lo)).to_pylist()
        if any(t!=meta['tasks'][1] for t in tasks) or any(t!=meta['tasks'][0] for t in instructions): raise ValueError('instruction mismatch')
        for ins,cap in zip(instructions,table['task_cap'].slice(int(lo),int(hi-lo)).to_pylist()):
            if ins in caps and caps[ins]!=cap: raise ValueError('ambiguous instruction cap')
            caps[ins]=cap
        r=np.zeros((n,2),np.float32); g=np.zeros((n,28),np.float32)
        r[fs]=ratio[lo:hi]; g[fs]=grade[lo:hi]
        if not a.dry_run:
            dest=(out/f'episode_{episode:06d}.parquet' if a.mode=='sidecar' else
                  out/info['data_path'].format(episode_chunk=episode//info['chunks_size'],episode_index=episode))
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists(): raise ValueError(f'partial overlay exists: {dest}; use a fresh output')
            rc=pa.array(r.tolist(),type=pa.list_(pa.float32(),2)); gc=pa.array(g.tolist(),type=pa.list_(pa.float32(),28))
            if a.mode=='sidecar':
                result=pa.table({'episode_index':np.full(n,episode,np.int32),'frame_index':np.arange(n,dtype=np.int32),
                                 'ratio_label':rc,'grade_prob_label':gc})
            else:
                result=pq.read_table(base_path)
                if 'ratio_label' in result.column_names or 'grade_prob_label' in result.column_names:
                    raise ValueError('base already contains supervision; use clean source')
                result=result.append_column('ratio_label',rc).append_column('grade_prob_label',gc)
            pq.write_table(result,dest,compression='zstd')
        wrote+=1; total+=n
    output_manifest=dict(format_version=2,grade_schema='d1v2_4x6_soft',category_order=['1','2','3','4','5','U'],
                         dataset_episodes_sha256=digest(root/'meta/episodes.jsonl'),
                         dataset_info_sha256=digest(root/'meta/info.json'),
                         dataset_modality_sha256=digest(root/'meta/modality.json'),
                         source_sha256=report['canonical_sha256'],ratio_grid=[1,1.5,2,2.5],
                         instruction_caps=caps,episodes=wrote,base_frames=total,labelled_frames=len(ep),
                         masked_frames=total-len(ep),tail=16,stride=manifest['stride'],
                         unlabelled_policy='mask_no_propagation',mode=manifest['mode'],
                         annotation_fingerprint=report['raw_fingerprint'])
    output_manifest['overlay_mode']=a.mode
    if not a.dry_run:
        if a.mode=='parquet':
            shutil.copytree(root/'meta',out/'meta')
            if (root/'videos').exists(): (out/'videos').symlink_to((root/'videos').resolve(),target_is_directory=True)
            modality=json.loads((out/'meta/modality.json').read_text())
            stats=json.loads((out/'meta/stats.json').read_text())
            missing=total-len(ep)
            for name,values in [('ratio_label',ratio),('grade_prob_label',grade)]:
                width=values.shape[1]
                info['features'][name]={'dtype':'float32','shape':[width],'names':[f'{name}_{i}' for i in range(width)]}
                modality['action'][name]={'original_key':name,'start':0,'end':width,'absolute':True,'dtype':'float32'}
                columns=[np.concatenate([values[:,j].astype(np.float64),np.zeros(missing)]) for j in range(width)]
                stats[name]={k:[float(fn(v)) for v in columns] for k,fn in [
                    ('min',np.min),('max',np.max),('mean',np.mean),('std',np.std),
                    ('q01',lambda x:np.quantile(x,.01)),('q99',lambda x:np.quantile(x,.99))]}
            for name,value in [('info.json',info),('modality.json',modality),('stats.json',stats)]:
                (out/'meta'/name).write_text(json.dumps(value,indent=2)+'\n')
        (out/'manifest.json').write_text(json.dumps(output_manifest,indent=2)+'\n')
    print(json.dumps({k:output_manifest[k] for k in ('episodes','base_frames','labelled_frames','masked_frames','mode')},indent=2))


if __name__=='__main__': main()
