"""Оценка модели: golden / dev_B (внутренние) и val (официальный оценщик v2.1 + строгий без утечки).

  python src/eval_val.py --model models/compactfact_v5.pkl --genes models/genes_best.json
  python src/eval_val.py --model runs/v5_distill.pkl --genes models/genes_best.json --tag distill
Пишет runs/<tag>_val.jsonl (предсказания) и runs/<tag>_eval.json (метрики).
"""
import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, json, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cfx.bench import Bench
from cfx.data import splits
from cfx.objects import refresh

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--genes", default="models/genes_best.json")
ap.add_argument("--tag", default="eval")
ap.add_argument("--sets", default="golden,dev_B,val")
a = ap.parse_args()

ex = pickle.load(open(a.model, "rb")); refresh(ex.objects)
genes = json.load(open(a.genes))
S = splits(); res = {}
KEYS = ("temporal_recall", "state_accuracy", "state_accuracy_official", "fact_precision", "fact_recall", "grounding_turn")
for name in a.sets.split(","):
    t = time.time()
    b = Bench(ex, S[name], name); b.official = name == "val"
    m, pred = b.evaluate(genes)
    res[name] = {k: round(v, 4) for k, v in m.items() if isinstance(v, (int, float))}
    print(f"{name:7s} n={len(S[name]):4d}  " + "  ".join(f"{k}={m[k]:.4f}" for k in KEYS if k in m)
          + f"   ({time.time() - t:.0f} с)", flush=True)
    if name == "val":
        with open(f"runs/{a.tag}_val.jsonl", "w", encoding="utf-8") as fh:
            for c in pred.values():
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
json.dump(res, open(f"runs/{a.tag}_eval.json", "w"), ensure_ascii=False, indent=1)
print("критерий: temporal_recall ≥ 0.4506, state_accuracy ≥ 0.5145 (официальная)")
