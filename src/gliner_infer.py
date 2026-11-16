"""Спаны GLiNER для эволюции: модель × контекст → кэш спанов (низкий порог) на golden / dev_B / val / val_adv.

  python src/gliner_infer.py --model out/xs --tag xs --ctx 0,25
Пишет runs/spans/<tag>_c<ctx>_<set>.pkl и runs/spans/<tag>_timing.json (мс на реплику, GPU/CPU).
"""
import os, sys, json, time, pickle, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--ctx", default="auto", help="auto = из out/<model>/arch.json")
ap.add_argument("--sets", default="golden,dev_B,val,val_adv")
ap.add_argument("--raw_thr", type=float, default=0.1)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--out_dir", default="runs/spans")
ap.add_argument("--limit", type=int, default=0, help="для отладки: первые N звонков набора")
a = ap.parse_args()

from gliner import GLiNER
from gliner.data_processing.tokenizer import WordsSplitter
from cfx.data import splits, load_jsonl
from cfx.gliner_fusion import GlinerSpans

from cfx.gliner_hub import device
dev = device()
model = GLiNER.from_pretrained(a.model).to(dev).eval()
model.data_processor.words_splitter = WordsSplitter("whitespace")
model.config.words_splitter_type = "whitespace"
S = splits()
here = os.path.dirname(os.path.abspath(__file__))
adv = os.path.join(here, "..", "runs", "val_adv_0.3.jsonl")
if os.path.exists(adv):
    S["val_adv"] = load_jsonl(adv)
os.makedirs(a.out_dir, exist_ok=True)
timing = {"device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None}
if a.ctx == "auto":
    ap_ = os.path.join(a.model, "arch.json")
    a.ctx = str(json.load(open(ap_))["ctx"]) if os.path.exists(ap_) else "25"
for ctx in [int(x) for x in a.ctx.split(",")]:
    gl = GlinerSpans(model, context_words=ctx, raw_threshold=a.raw_thr, batch=a.batch)
    for name in a.sets.split(","):
        if name not in S:
            continue
        calls = S[name][: a.limit] if a.limit else S[name]
        t, n_turns, sp = time.time(), 0, {}
        with torch.no_grad():
            for c in calls:
                sp[c["call_id"]] = gl.spans(c); n_turns += len(c["turns"])
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elif dev.type == "mps":
            torch.mps.synchronize()
        el = time.time() - t
        pickle.dump(sp, open(f"{a.out_dir}/{a.tag}_c{ctx}_{name}.pkl", "wb"))
        timing[f"c{ctx}_{name}"] = {"ms_per_turn": round(1000 * el / max(n_turns, 1), 2), "ms_per_call": round(1000 * el / len(calls), 1)}
        print(a.tag, ctx, name, timing[f"c{ctx}_{name}"], flush=True)
json.dump(timing, open(f"{a.out_dir}/{a.tag}_timing.json", "w"), indent=1)
