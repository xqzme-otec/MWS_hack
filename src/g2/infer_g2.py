"""Спаны обученной GLiNER2.5 для эволюции сборки (run_evolve_gcore.py): golden / dev_B / val / val_adv.

  python src/g2/infer_g2.py --model out/g2_best --tag g2
→ runs/spans/<tag>_c<ctx>_<set>.pkl и runs/spans/<tag>_timing.json
"""
import os, sys, json, time, pickle, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import torch
import common as C
from cfx.data import splits, load_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--tag", default="g2")
ap.add_argument("--sets", default="golden,dev_B,val,val_adv"); ap.add_argument("--raw_thr", type=float, default=0.1)
ap.add_argument("--out_dir", default="runs/spans"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
m, arch = C.load(a.model)
dev = C.device(); m.to(dev).eval()
ctx = arch.get("ctx", 0)
S = splits()
adv = os.path.join(HERE, "..", "..", "runs", "val_adv_0.3.jsonl")
if os.path.exists(adv):
    S["val_adv"] = load_jsonl(adv)
os.makedirs(a.out_dir, exist_ok=True)
timing = {"device": str(dev), "arch": arch}
for name in a.sets.split(","):
    if name not in S:
        continue
    calls = S[name][: a.limit] if a.limit else S[name]
    t, sp, nt = time.time(), {}, 0
    for c in calls:
        sp[c["call_id"]] = C.call_spans(m, c, ctx, a.raw_thr); nt += len(c["turns"])
    el = time.time() - t
    pickle.dump(sp, open(f"{a.out_dir}/{a.tag}_c{ctx}_{name}.pkl", "wb"))
    timing[name] = {"ms_per_turn": round(1000 * el / max(nt, 1), 2), "ms_per_call": round(1000 * el / len(calls), 1)}
    print(a.tag, name, timing[name], flush=True)
json.dump(timing, open(f"{a.out_dir}/{a.tag}_timing.json", "w"), indent=1)
