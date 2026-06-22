#!/usr/bin/env python3
"""Quick crossover test: GPT-2 ASP vs AdamW at 400 steps. CPU, ~10min."""
import json, logging, sys, time, gc
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("qck")

ML, BS, NS = 512, 4, 400
NTr, NEv = 256, 100

def dl(tok, sp, n):
    ds=load_dataset("wikitext","wikitext-2-raw-v1",split=sp).select(range(min(n,len(ds))))
    ds=ds.map(lambda ex:tok(ex["text"],truncation=True,max_length=ML,padding="max_length"),batched=True,remove_columns=["text"])
    ds.set_format(type="torch",columns=["input_ids","attention_mask"])
    return DataLoader(ds,batch_size=BS,shuffle=(sp=="train"),
        collate_fn=lambda b:{"input_ids":torch.stack([x["input_ids"]for x in b]),
            "attention_mask":torch.stack([x["attention_mask"]for x in b]),
            "labels":torch.stack([x["input_ids"]for x in b])})

def ppl(m,dl,dev):
    m.eval();tl=0.0;tt=0
    with torch.no_grad():
        for b in dl:
            b={k:v.to(dev)for k,v in b.items()}
            lo=m(**b).loss;nt=b["attention_mask"].sum().item()
            tl+=lo.item()*nt;tt+=nt
    return round(float(torch.exp(torch.tensor(tl/max(tt,1))).item()),4)

def run(name,hf,tok,tr_dl,ev_dl,dev,is_asp=False):
    label=f"{name}_{'ASP' if is_asp else 'AdamW'}"
    logger.info("RUN %s (%d steps)",label,NS)
    t0=time.time()
    m=AutoModelForCausalLM.from_pretrained(hf,trust_remote_code=False,local_files_only=True)
    if is_asp: m=m.to(torch.float32)
    m=m.to(dev)
    history=[]
    if is_asp:
        # ASP cycles: ALS(1)→SGD(33)→Perturb(1)
        sgd_pc=50; cyc=NS//(sgd_pc+2)
        logger.info("ASP: %d cycles",cyc)
        gs=0
        for c in range(cyc):
            # ALS: tiny noise to lm_head
            m.eval()
            with torch.no_grad():
                if hasattr(m,'lm_head'):
                    m.lm_head.weight.data+=1e-5*torch.randn_like(m.lm_head.weight.data)
            gs+=1
            # SGD
            opt=torch.optim.SGD(m.parameters(),lr=1e-4,momentum=0.9,weight_decay=0.01)
            m.train();ss=0
            for b in tr_dl:
                if ss>=sgd_pc or gs>=NS: break
                b={k:v.to(dev)for k,v in b.items()};opt.zero_grad()
                loss=m(**b).loss;loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
                opt.step();gs+=1;ss+=1
                if gs%100==0:
                    p=ppl(m,ev_dl,dev);history.append({"step":gs,"ppl":p})
                    logger.info("  ASP step=%d ppl=%.2f",gs,p);m.train()
            # Perturb
            if gs<NS:
                with torch.no_grad():
                    for p in m.parameters():p.add_(1e-3*torch.randn_like(p))
                gs+=1
    else:
        opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=0.01)
        m.train();gs=0
        for b in tr_dl:
            if gs>=NS: break
            b={k:v.to(dev)for k,v in b.items()};opt.zero_grad()
            loss=m(**b).loss;loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
            opt.step();gs+=1
            if gs%100==0:
                p=ppl(m,ev_dl,dev);history.append({"step":gs,"ppl":p})
                logger.info("  ADAMW step=%d ppl=%.2f",gs,p);m.train()
    el=time.time()-t0
    final=ppl(m,ev_dl,dev)
    history.append({"step":NS,"ppl":final})
    logger.info("FINAL %s: PPL=%.4f (%.0fs)",label,final,el)
    del m;gc.collect()
    return {"name":label,"ppl":final,"time_s":int(el),"history":history}

def main():
    tok=AutoTokenizer.from_pretrained("gpt2",local_files_only=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tr_dl=dl(tok,"train",NTr); ev_dl=dl(tok,"test",NEv)
    dev=torch.device("cpu")
    results=[]
    results.append(run("GPT2","gpt2",tok,tr_dl,ev_dl,dev,False))
    results.append(run("GPT2","gpt2",tok,tr_dl,ev_dl,dev,True))
    print()
    for r in results: print(f"{r['name']}: final={r['ppl']}, time={r['time_s']}s")
    with open("runs/crossover/quick_results.json","w") as f: json.dump(results,f,indent=2)

if __name__=="__main__":
    sys.exit(main())
