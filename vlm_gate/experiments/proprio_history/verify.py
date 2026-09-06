"""Meaningful smoke checks: causality, gradients, adapter identity and timings."""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from model import ExecutedActionHistory, SmallGateProprioHistory, SmallGateBaseline, pack_history
from predict import Predictor
from train import MotionFrames, write_json


def timed(fn, iterations=100, repeats=5):
    for _ in range(10):
        fn()
    values=[]
    for _ in range(repeats):
        torch.cuda.synchronize(); start=time.perf_counter()
        for _ in range(iterations):
            fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter()-start)*1000/iterations)
    return {"median_ms":float(np.median(values)),"min_ms":float(min(values)),
            "max_ms":float(max(values)),"repeats_ms":values}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",required=True); args=p.parse_args()
    out=Path(args.run_dir)
    torch.set_num_threads(4); torch.manual_seed(91)
    predictor=Predictor(out/"checkpoint.pt")
    model=predictor.model; config=model.config
    ck=torch.load(out/"checkpoint.pt",map_location="cpu",weights_only=False)
    settings=ck["args"]
    labels=pd.read_parquet(out/"split.parquet")
    data=MotionFrames(labels,settings["dataset_path"],settings["cache_dir"],
                      predictor.embeddings,config["history"])
    checks={}
    # Strictly past slicing: perturb current AND every future action.
    a=np.arange(40*12,dtype=np.float32).reshape(40,12)
    for f in (0,1,7,16,31):
        old,mask=pack_history(a,f,16)
        altered=a.copy(); altered[f:]=-12345
        new,newmask=pack_history(altered,f,16)
        np.testing.assert_array_equal(old,new); np.testing.assert_array_equal(mask,newmask)
        assert mask.sum()==min(f,16)
    checks["current_and_future_actions_excluded"]=True
    ring=ExecutedActionHistory()
    assert ring.arrays()[1].sum()==0
    for row in a[:20]:
        ring.append_executed(row)
    np.testing.assert_array_equal(ring.arrays()[0],a[4:20])
    ring.reset(); assert ring.arrays()[1].sum()==0
    checks["executed_ring_order_trim_and_episode_reset"]=True
    train_idx=np.flatnonzero(labels.split.to_numpy()=="train")
    val_idx=np.flatnonzero(labels.split.to_numpy()=="val")
    assert not set(labels.iloc[train_idx].episode_index)&set(labels.iloc[val_idx].episode_index)
    checks["episode_disjoint_split"]=True
    expected=data.states[train_idx].mean(axis=0,dtype=np.float64).astype(np.float32)
    np.testing.assert_allclose(model.state_mean.cpu().numpy(),expected,rtol=1e-6,atol=1e-6)
    expected_a=data.actions[train_idx][data.mask[train_idx].astype(bool)].mean(axis=0,dtype=np.float64).astype(np.float32)
    np.testing.assert_allclose(model.action_mean.cpu().numpy(),expected_a,rtol=1e-6,atol=1e-6)
    checks["normalization_uses_training_rows_and_valid_history_only"]=True
    batch=[t.cuda() for t in next(iter(torch.utils.data.DataLoader(
        torch.utils.data.Subset(data,train_idx[:64]),batch_size=64)))]
    inputs,y=batch[:-1],batch[-1]
    # Reload round trip does not change predictions; uses the full real input tuple.
    clone=SmallGateProprioHistory(**config).cuda().eval(); clone.load_state_dict(ck["model"])
    with torch.inference_mode():
        expected=model(*inputs)
        torch.testing.assert_close(clone(*inputs),expected,rtol=0,atol=0)
    checks["checkpoint_exact_reload"]=True
    i=int(train_idx[0]); sample=data[i]; image=sample[0].numpy()
    images=[np.rint(image[j*3:(j+1)*3].transpose(1,2,0)*255).astype(np.uint8) for j in range(3)]
    response=predictor.predict(images,labels.iloc[i].task,data.states[i],data.actions[i],data.mask[i])
    with torch.inference_mode():
        direct=float(model(*[t[None].cuda() for t in sample[:-1]]).sigmoid().item())
    assert abs(response["confidence"]-direct)<1e-6
    checks["raw_RGB_state_action_inference_adapter_matches_training"]=True
    # Padded values cannot affect outputs after normalization.
    pad=[t[:1].clone() for t in inputs]; pad[-1].zero_()
    with torch.inference_mode():
        p0=model(*pad); pad[-2].fill_(12345); p1=model(*pad)
        torch.testing.assert_close(p0,p1,rtol=0,atol=0)
    checks["masked_padding_cannot_change_prediction"]=True
    sensitivities={}
    for name,j in (("proprio",2),("past_actions",3)):
        local=[t.detach().clone() for t in inputs]
        local[j].requires_grad_(True)
        gradient=torch.autograd.grad(model(*local).sum(),local[j])[0]
        norm=float(gradient.norm())
        assert np.isfinite(norm) and norm>0
        sensitivities[name+"_input_gradient_norm"]=norm
    checks["both_new_inputs_affect_predictions"]=True
    # Overfit a fixed real minibatch: proves optimization reaches the actual model.
    probe=copy.deepcopy(model).eval()  # Freeze BN running statistics, not parameters.
    optimizer=torch.optim.AdamW(probe.parameters(),lr=1e-3)
    loss_fn=torch.nn.BCEWithLogitsLoss()
    tiny=[t[:16] for t in inputs]; target=y[:16]
    with torch.no_grad():
        first=float(loss_fn(probe(*tiny),target))
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss=loss_fn(probe(*tiny),target); loss.backward(); optimizer.step()
    with torch.no_grad():
        last=float(loss_fn(probe(*tiny),target))
    assert np.isfinite(last) and last<first-1e-3,(first,last)
    checks["fixed_real_minibatch_loss_decreases"]={"before":first,"after":last,"steps":100}
    # Check replacement AUC tie handling against the previous pairwise definition.
    scores=np.array([.1,.5,.5,.7,.2,.9]); target_auc=np.array([0,1,0,1,0,1])
    pos=scores[target_auc==1]; neg=scores[target_auc==0]
    pairwise=np.mean([(p>n)+.5*(p==n) for p in pos for n in neg])
    assert abs(roc_auc_score(target_auc,scores)-pairwise)<1e-12
    checks["fast_auc_matches_pairwise_with_ties"]=True
    baseline=SmallGateBaseline(text_dim=config["text_dim"]).cuda().eval()
    timings={}
    # FP32, same GPU/process, input tensors already resident. Not HTTP latency.
    with torch.inference_mode():
        for size in (1,len(y)):
            x=[t[:size] for t in inputs]
            timings[f"baseline_forward_bs{size}"]=timed(lambda:baseline(*x[:2]))
            timings[f"motion_forward_bs{size}"]=timed(lambda:model(*x))
    baseline.train(); bench_motion=copy.deepcopy(model).train()
    for name,m,x in (("baseline",baseline,inputs[:2]),("motion",bench_motion,inputs)):
        opt=torch.optim.AdamW(m.parameters(),lr=3e-4)
        def step():
            opt.zero_grad(set_to_none=True)
            loss_fn(m(*x),y).backward(); opt.step()
        timings[name+"_train_step_bs64"]=timed(step,iterations=30,repeats=5)
    result={"checks":checks,"input_sensitivity":sensitivities,"gpu":torch.cuda.get_device_name(),
            "baseline_params":sum(p.numel() for p in baseline.parameters()),
            "motion_params":sum(p.numel() for p in model.parameters()),
            "timings":timings,"timing_scope":"FP32 resident tensors, synchronized wall-clock, includes launches; excludes I/O/HTTP; same GPU, no policy contention benchmark"}
    write_json(out/"verification.json",result)
    print("VERIFICATION_COMPLETE",result,flush=True)


if __name__=="__main__":
    main()
