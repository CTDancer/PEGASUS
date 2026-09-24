"""Objective-free closed-form calibration of direct terminal Koopman geometry.

Loads the successful v4 Stage-A checkpoint only for its frozen LKF, immutable
anchor and information metric.  The old exact-composition A and propagated J are
not migrated.  For each configured chain C this program estimates directly:

    E[r(X_1)|X_t0,C] ~= A_C r(X_t0)
    d/dv_k E[r(X_1^v)|X_t0,C] = M_{C,k}

from hard physical chain rollouts.  It is DDP-safe and contains no optimizer.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from .distributed import cleanup_distributed, initialize_distributed
from .lkf.data import ChunkedCleanPeptideBatchDataset, build_clean_loader
from .terminal_controlled_koopman_checkpoint import (
    initialize_terminal_from_controlled_checkpoint,
    save_terminal_controlled_koopman_checkpoint,
)
from .terminal_controlled_koopman_geometry import (
    chain_name,
    chain_sufficient_statistics,
    fit_direct_operator,
    fit_terminal_response,
    hard_chain_samples,
    parse_chains,
    prepare_chain_start,
    spectral_summary,
)
from .terminal_controlled_koopman_model import TerminalControlledKoopmanConfig
from .utils import seed_everything, write_csv, write_json


def run(args,ctx)->Path:
    seed_everything(int(args.seed)+ctx.rank)
    chains=parse_chains(args.chains)
    cfg=TerminalControlledKoopmanConfig(chains=chains,direct_operator_ridge=float(args.direct_operator_ridge),soft_temperature=float(args.soft_temperature))
    model,_=initialize_terminal_from_controlled_checkpoint(
        args.controlled_stage_a_checkpoint,
        base_lkf_checkpoint=args.lkf_checkpoint,
        config=cfg,
        strict_base_sha=not bool(args.allow_base_sha_mismatch),
    )
    device=ctx.device
    model.to(device).eval()

    if ctx.is_main:
        print("Building calibration train index...",flush=True)
    t0=time.time()
    ds=ChunkedCleanPeptideBatchDataset(args.dataset_root,args.train_split,max_sequences=int(args.max_batch_sequences),min_sequences=2)
    sampler=DistributedSampler(ds,num_replicas=ctx.world_size,rank=ctx.rank,shuffle=True,seed=int(args.seed),drop_last=False) if ctx.distributed else None
    loader=build_clean_loader(ds,shuffle=sampler is None,sampler=sampler,num_workers=int(args.num_workers))
    if ctx.is_main:
        st=ds.length_statistics(); print(f"Calibration index done: chunks={st['chunks']} sequences={st['sequences']} seconds={time.time()-t0:.1f}",flush=True)
        print(f"Calibrating direct terminal geometry on DDP x{ctx.world_size}; chains={len(chains)}; continuations={args.continuations}",flush=True)

    d=model.anchor_dim; maxD=model.max_innovation_dim; C=len(chains)
    xtx=[torch.zeros(d,d,device=device,dtype=torch.float64) for _ in chains]
    ytx=[torch.zeros(d,d,device=device,dtype=torch.float64) for _ in chains]
    nstate=[torch.zeros((),device=device,dtype=torch.float64) for _ in chains]
    pulse_cross=[]; pulse_counts=[]
    for c in chains:
        pulse_cross.append([torch.zeros(d,maxD,device=device,dtype=torch.float64) for _ in range(len(c)-1)])
        pulse_counts.append([torch.zeros(maxD,device=device,dtype=torch.float64) for _ in range(len(c)-1)])

    target_steps=int(args.steps_per_chain)*C
    it=iter(loader); epoch=0
    per_chain_steps=[0]*C
    step=0
    while step<target_steps:
        try: batch=next(it)
        except StopIteration:
            epoch+=1
            if sampler is not None: sampler.set_epoch(epoch)
            it=iter(loader); batch=next(it)
        ci=step%C
        c=chains[ci]
        xclean=torch.as_tensor(batch,device=device,dtype=torch.long)
        g=torch.Generator(device=device); g.manual_seed(int(args.seed)+1000003*step+97*ctx.rank)
        xstart=prepare_chain_start(model,xclean,c,generator=g)
        r,xis,_=hard_chain_samples(model,xstart,c,continuations=int(args.continuations),generator=g)
        xx,yy,nn,blocks=chain_sufficient_statistics(model,xstart,c,r,xis)
        xtx[ci]+=xx; ytx[ci]+=yy; nstate[ci]+=nn
        D=xis[0].shape[-1]
        for pi,(cross,count) in enumerate(blocks):
            pulse_cross[ci][pi][:,:D]+=cross
            pulse_counts[ci][pi][:D]+=count
        per_chain_steps[ci]+=1; step+=1
        if ctx.is_main and (step%max(1,int(args.log_every))==0 or step==target_steps):
            print(f"[calibrate] step={step}/{target_steps} chain={chain_name(c)}",flush=True)

    # All ranks accumulated the same chain schedule on different data.  Reduce
    # complete sufficient statistics only after all local CUDA sampling is done.
    if ctx.distributed:
        for ci,c in enumerate(chains):
            for z in (xtx[ci],ytx[ci],nstate[ci]): dist.all_reduce(z,op=dist.ReduceOp.SUM)
            for pi in range(len(c)-1):
                dist.all_reduce(pulse_cross[ci][pi],op=dist.ReduceOp.SUM)
                dist.all_reduce(pulse_counts[ci][pi],op=dist.ReduceOp.SUM)

    rows=[]
    for ci,c in enumerate(chains):
        A=fit_direct_operator(xtx[ci],ytx[ci],ridge=float(args.direct_operator_ridge))
        model.chain_mean_operators[model._chain_keys[ci]].data.copy_(A)
        model.chain_mean_counts[ci]=nstate[ci]
        blocks=[]
        for pi in range(len(c)-1):
            M=fit_terminal_response(pulse_cross[ci][pi],pulse_counts[ci][pi])
            key=model._pulse_keys[model.pulse_response_index(c,pi)]
            model.terminal_responses[key].data.copy_(M)
            model.terminal_response_counts[model.pulse_response_index(c,pi)].copy_(pulse_counts[ci][pi])
            blocks.append(M)
        spec=spectral_summary(model.gramian(c,token_length=int(args.diagnostic_token_length),normalized=True))
        rows.append({"chain":chain_name(c),"state_count":float(nstate[ci].item()),"min_column_count":float(min(pc[:model.terminal_response_block(c,0,token_length=int(args.diagnostic_token_length)).shape[1]].min().item() for pc in pulse_counts[ci])),**spec})
    model.geometry_calibrated.fill_(True)

    out=Path(args.output_dir).expanduser().resolve()/args.run_name
    if ctx.is_main:
        out.mkdir(parents=True,exist_ok=True)
        write_csv(out/"calibration_geometry.csv",rows)
        summary={"chains":[list(c) for c in chains],"steps_per_chain":int(args.steps_per_chain),"continuations":int(args.continuations),"world_size":ctx.world_size,"direct_operator_ridge":float(args.direct_operator_ridge),"geometry":"direct chain-specific A_C + direct terminal Stein M_C,k","discarded":"v4 exact-composition A and propagated A@J"}
        write_json(out/"calibration_summary.json",summary)
        ckpt=save_terminal_controlled_koopman_checkpoint(
            out/"checkpoints"/"calibrated.pt",model,
            base_lkf_checkpoint=args.lkf_checkpoint,
            source_controlled_checkpoint=args.controlled_stage_a_checkpoint,
            stage="CALIBRATION",epoch=0,global_step=target_steps,
            training_args=vars(args),metrics=summary,
        )
        print(f"Saved calibrated direct-terminal checkpoint: {ckpt}",flush=True)
    return out/"checkpoints"/"calibrated.pt"


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lkf-checkpoint",required=True); p.add_argument("--controlled-stage-a-checkpoint",required=True)
    p.add_argument("--allow-base-sha-mismatch",action="store_true")
    p.add_argument("--dataset-root",required=True); p.add_argument("--train-split",default="train")
    p.add_argument("--output-dir",default="results"); p.add_argument("--run-name",default="terminal_controlled_koopman_calibration")
    p.add_argument("--device",default="cuda"); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--chains",default="0,1;0.25,1;0.5,1;0.75,1;0,0.25,0.5,1;0,0.5,1;0,0.75,1;0.25,0.5,0.75,1;0.25,0.75,1;0.5,0.75,1")
    p.add_argument("--continuations",type=int,default=8); p.add_argument("--steps-per-chain",type=int,default=32)
    p.add_argument("--max-batch-sequences",type=int,default=32); p.add_argument("--num-workers",type=int,default=2); p.add_argument("--log-every",type=int,default=10)
    p.add_argument("--direct-operator-ridge",type=float,default=1e-4); p.add_argument("--soft-temperature",type=float,default=0.5)
    p.add_argument("--diagnostic-token-length",type=int,default=14,help="token length including CLS/EOS used only for calibration spectrum reporting")
    return p


def main():
    args=build_parser().parse_args(); ctx=initialize_distributed(args.device)
    try: run(args,ctx)
    finally: cleanup_distributed(ctx)

if __name__=="__main__": main()
