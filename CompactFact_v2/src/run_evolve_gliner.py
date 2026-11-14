"""Эволюция 4: параметры движка GLiNER (пороги по меткам, веса слияния) поверх v5 после LoRA-дообучения.

Спаны GLiNER считаются один раз (низкий порог) и кэшируются; эволюция гоняет гены по кэшу.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, pickle, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="runs/v5.pkl"); ap.add_argument("--adapter", default="runs/gliner_xs_lora_adapter.safetensors")
ap.add_argument("--base_gliner", default="../hf/gliner-x-small"); ap.add_argument("--base", default="runs/evo3_best.json")
ap.add_argument("--tag", default="evo4"); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--gens", type=int, default=10)
ap.add_argument("--pop", type=int, default=16); ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()
from cfx.bench import Bench
from cfx.data import splits
from cfx.evolve import evolve, fitness, SPEC, SPEC_GLINER
from cfx.objects import refresh

ex = pickle.load(open(a.model, "rb")); refresh(ex.objects); S = splits()
# torch/GLiNER грузятся только если кэша спанов нет: fork воркеров эволюции после старта
# пула потоков OpenMP в torch приводит к взаимоблокировке (наблюдалось).
gl = None
cache = {}
for name in ("golden", "dev_B", "val"):
    path = f"runs/gl_spans_{name}.pkl"
    if os.path.exists(path):
        cache[name] = pickle.load(open(path, "rb")); continue
    if gl is None:
        import torch
        torch.set_num_threads(a.threads)
        from cfx.gliner_fusion import load_finetuned, GlinerSpans
        gl = GlinerSpans(load_finetuned(a.base_gliner, a.adapter), raw_threshold=0.15)
    t = time.time(); sp = {c["call_id"]: gl.spans(c) for c in S[name]}
    print(f"spans {name}: {len(sp)} calls, {(time.time() - t) / len(sp):.2f} s/call", flush=True)
    pickle.dump(sp, open(path, "wb")); cache[name] = sp
del gl  # дальше GLiNER не нужен: эволюция идёт по кэшу


def bench(name):
    b = Bench(ex, S[name], name)
    for cid, s in b.scored.items():
        s["gl_spans"] = cache[name].get(cid, [])
    return b


golden, dev = bench("golden"), bench("dev_B")
base = {**json.load(open(a.base)), "gl_on": 1}
spec = {**SPEC, **SPEC_GLINER}
best, hist = evolve(golden, dev, spec=spec, base=base, pop=a.pop, gens=a.gens, workers=a.workers,
                    log_path=f"runs/{a.tag}_history.json", log=lambda s: print(s, flush=True))
json.dump(best, open(f"runs/{a.tag}_best.json", "w"), ensure_ascii=False, indent=1)
val = bench("val"); val.official = True
res = {}
for label, g in (("evo4 best (GLiNER on)", best), ("same genes, GLiNER off", {**best, "gl_on": 0}), ("evo3 best (no GLiNER)", json.load(open(a.base)))):
    m, pred = val.evaluate(g); res[label] = m
    print("VAL", label, m, flush=True)
    if label.startswith("evo4"):
        with open(f"runs/{a.tag}_val.jsonl", "w") as fh:
            for c in pred.values(): fh.write(json.dumps(c, ensure_ascii=False) + "\n")
json.dump(res, open(f"runs/{a.tag}_val_metrics.json", "w"), ensure_ascii=False, indent=1)
