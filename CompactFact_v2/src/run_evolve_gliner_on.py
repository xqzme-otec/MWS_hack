"""Лучшая конфигурация с принудительно включённым GLiNER (для честной точки кривой «ресурсы × качество»)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, pickle
sys.path.insert(0, os.path.dirname(__file__))
from cfx.bench import Bench
from cfx.data import splits
from cfx.evolve import evolve, SPEC, SPEC_GLINER
from cfx.objects import refresh
ex = pickle.load(open("runs/v5.pkl", "rb")); refresh(ex.objects); S = splits()
def bench(name):
    b = Bench(ex, S[name], name); sp = pickle.load(open(f"runs/gl_spans_{name}.pkl", "rb"))
    for cid, s in b.scored.items(): s["gl_spans"] = sp.get(cid, [])
    return b
golden, dev = bench("golden"), bench("dev_B")
spec = {k: v for k, v in {**SPEC, **SPEC_GLINER}.items() if k != "gl_on"}
base = {**json.load(open("runs/evo4_best.json")), "gl_on": 1, "gl_w_new": 0.3, "gl_w_boost": 0.2}
best, hist = evolve(golden, dev, spec=spec, base=base, pop=16, gens=8, workers=2, log_path="runs/evo4on_history.json",
                    log=lambda s: print(s, flush=True), confirm_every=4)
best["gl_on"] = 1
json.dump(best, open("runs/evo4on_best.json", "w"), ensure_ascii=False, indent=1)
val = bench("val"); val.official = True
m, _ = val.evaluate(best); print("VAL GLiNER forced on:", m, flush=True)
json.dump(m, open("runs/evo4on_val_metrics.json", "w"), ensure_ascii=False, indent=1)
