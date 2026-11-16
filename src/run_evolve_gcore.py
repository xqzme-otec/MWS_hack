"""Эволюция GLiNER-core: GLiNER — основная модель; гены — модель × контекст (дорогие, из кэша спанов),
префильтр реплик, пороги, нормализация, сдвиг договорённостей + гены сборки (ловушки, объекты, даты, текущее значение).
Фитнес = min(T−T*, S−S*) + 0.15·F1 − 0.02·(задержка, с − 1) — мягкий штраф за задержку на CPU.

  python src/run_evolve_gcore.py --spans results_spans.zip --tag gc1 [--lat runs/bench_gliner_cpu.json --quant int8]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, pickle, glob, zipfile, argparse, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--spans", default="results_spans.zip", help="zip из ноутбука или папка с <tag>_c<ctx>_<set>.pkl")
ap.add_argument("--model", default="models/compactfact_v5.pkl", help="берём из него только банк значений/TF-IDF/объекты/даты/селектор")
ap.add_argument("--base", default="models/genes_best.json")
ap.add_argument("--lat", default="runs/bench_gliner_cpu.json"); ap.add_argument("--quant", default="fp32"); ap.add_argument("--threads_lat", default="t2")
ap.add_argument("--base_ms", type=float, default=60.0)
ap.add_argument("--tag", default="gc1"); ap.add_argument("--pop", type=int, default=20); ap.add_argument("--gens", type=int, default=14)
ap.add_argument("--workers", type=int, default=2); ap.add_argument("--only", default="", help="ограничить конфиги, напр. xs_c0,xs_c25")
ap.add_argument("--no_stop", action="store_true", help="не останавливаться при достижении цели (искать максимум)")
ap.add_argument("--hybrid", action="store_true", help="GLiNER + кандидаты бустинга (гены gc_hyb_*)")
ap.add_argument("--no_prefilter", action="store_true", help="GLiNER обрабатывает ВСЕ реплики (префильтр выключен)")
a = ap.parse_args()
from cfx.bench import GCBench
from cfx.data import splits, load_jsonl
from cfx.evolve import evolve, SPEC
from cfx import gliner_core
from cfx.objects import refresh

# ---- спаны
d = a.spans
if d.endswith(".zip"):
    out = "runs/spans_in"; os.makedirs(out, exist_ok=True)
    zipfile.ZipFile(d).extractall(out); d = out
files = glob.glob(f"{d}/**/*.pkl", recursive=True)
spans = {}
for f in files:
    m = re.match(r"(.+)_c(\d+)_(golden|dev_B|val|val_adv)\.pkl$", os.path.basename(f))
    if m:
        spans.setdefault(m[3], {})[f"{m[1]}_c{m[2]}"] = pickle.load(open(f, "rb"))
cfgs = sorted(set(spans.get("golden", {})) & set(spans.get("dev_B", {})))
if a.only:
    cfgs = [c for c in cfgs if c in a.only.split(",")]
assert cfgs, f"нет спанов golden+dev_B в {d}"
# ---- задержка на CPU: мс на реплику по архитектуре/контексту
lat = json.load(open(a.lat)) if os.path.exists(a.lat) else {}
def per_turn(cfg):
    tag, ctx = cfg.rsplit("_c", 1)
    arch = tag.split("_")[0]
    k = f"{arch}|{a.quant}|{a.threads_lat}|c{ctx}"
    return lat.get(k, {}).get("ms_per_turn", 150.0)
ptm = {c: per_turn(c) for c in cfgs}
print("конфигурации:", {c: f"{ptm[c]} мс/реплику" for c in cfgs}, flush=True)

ex = pickle.load(open(a.model, "rb")); refresh(ex.objects); S = splits()
S["val_adv"] = load_jsonl("runs/val_adv_0.3.jsonl") if os.path.exists("runs/val_adv_0.3.jsonl") else []
bench = lambda name: GCBench(ex, S[name], {c: spans[name][c] for c in cfgs if c in spans.get(name, {})}, ptm, a.base_ms, name, hybrid=a.hybrid)
golden, dev = bench("golden"), bench("dev_B")
spec = {k: v for k, v in SPEC.items() if not k.startswith("thr_")}   # пороги бустинга не нужны: p даёт GLiNER
spec.update(gliner_core.SPEC); spec["gc_cfg"] = ("choice", cfgs)
if a.hybrid:
    spec.update(gliner_core.HYB_SPEC)
base = {**json.load(open(a.base)), **gliner_core.GENES, "gc_cfg": cfgs[0],
        **{f"thr_{x}": 0.0 for x in gliner_core.ATTRS}}
if a.no_prefilter:
    for k_ in ("gc_pref_words", "gc_pref_sim"):
        spec.pop(k_, None)
    base = {**base, "gc_pref_words": 0, "gc_pref_sim": 0.0}
if a.hybrid:
    base = {"gc_hyb": 1, "gc_hyb_w": 0.5, "gc_hyb_mode": 0, "gc_hyb_min": 0.2, **base}
best, hist = evolve(golden, dev, spec=spec, base=base, pop=a.pop, gens=a.gens, workers=a.workers,
                    log_path=f"runs/{a.tag}_history.json", log=lambda s: print(s, flush=True), stop_at_target=not a.no_stop)
json.dump(best, open(f"runs/{a.tag}_best.json", "w"), ensure_ascii=False, indent=1)
res = {}
for name in ("golden", "dev_B", "val", "val_adv"):
    if not S.get(name) or best["gc_cfg"] not in spans.get(name, {}):
        continue
    b = bench(name); b.official = name in ("val", "val_adv")
    m, pred = b.evaluate(best); res[name] = m
    print("ИТОГ", name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)
    if name == "val":
        with open(f"runs/{a.tag}_val.jsonl", "w") as fh:
            for c in pred.values(): fh.write(json.dumps(c, ensure_ascii=False) + "\n")
json.dump({"best": best, "metrics": res, "per_turn_ms": ptm}, open(f"runs/{a.tag}_result.json", "w"), ensure_ascii=False, indent=1)
