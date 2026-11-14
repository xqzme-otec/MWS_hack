"""Замеры ресурсов по единой методике.

Методика: отдельный процесс на конфигурацию; модель загружается, 5 звонков прогрева, затем N звонков
val последовательно (batch=1, как в HTTP-сервисе); фиксированное число потоков (OMP/torch);
латентность p50/p95/mean — по вызову predict на звонок; пиковая память — ru_maxrss процесса;
пропускная способность = 3600 / mean_latency (звонков в час на один процесс);
«активные параметры» — число параметров/узлов, реально участвующих в расчёте одного звонка:
  · классическая часть: ненулевые TF-IDF-признаки × элементы банков + посещённые узлы деревьев;
  · GLiNER: все параметры энкодера плотные (активны 100% на токен), LoRA слита в веса.
GPU в среде нет: GPU-секунды на 1000 звонков для CPU-конфигураций = 0 (GPU не используется).
"""
import os, sys, json, time, resource, argparse, pickle, platform
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--genes", default="{}"); ap.add_argument("--n", type=int, default=100)
ap.add_argument("--threads", type=int, default=1); ap.add_argument("--gliner", default=""); ap.add_argument("--out", required=True)
ap.add_argument("--name", default=""); ap.add_argument("--denoise", default=""); ap.add_argument("--adapter", default="")
a = ap.parse_args()
os.environ["OMP_NUM_THREADS"] = str(a.threads)
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from cfx.data import splits
from cfx.objects import refresh

t0 = time.time()
ex = pickle.load(open(a.model, "rb")); refresh(ex.objects)
genes = json.load(open(a.genes)) if a.genes.endswith(".json") else json.loads(a.genes)
gl = None
if a.gliner:
    import torch
    torch.set_num_threads(a.threads)
    from gliner import GLiNER
    from cfx.gliner_fusion import GlinerSpans
    if a.adapter:
        from cfx.gliner_fusion import load_finetuned
        gl = GlinerSpans(load_finetuned(a.gliner, a.adapter))
    else:
        gl = GlinerSpans(GLiNER.from_pretrained(a.gliner))
load_s = time.time() - t0
calls = splits()["val"][: a.n + 5]


den = pickle.load(open(a.denoise, "rb")) if a.denoise else None


def run(c):
    if den is not None:
        c = {**c, "turns": [{**t, "text": den(t["text"])} for t in c["turns"]]}
    s = ex.score(c)
    if gl is not None:
        s["gl_spans"] = gl.spans(s["view"])
    return ex.assemble(s, genes)


for c in calls[:5]:
    run(c)
lat = []
for c in calls[5:]:
    t = time.perf_counter(); run(c); lat.append((time.perf_counter() - t) * 1000)
lat = np.array(lat)
# «активные» элементы классической части
nnz = np.mean([ex.vec.transform([t["text"] for t in c["turns"]]).nnz for c in calls[5:25]])
bank = sum(len(b["values"]) for b in ex.banks.values())
trees = {k: sum(len(p.nodes) for it in m._predictors for p in it) for k, m in ex.rankers.items()}
depth = {k: float(np.mean([p.get_max_depth() if hasattr(p, "get_max_depth") else 6 for it in m._predictors for p in it])) for k, m in ex.rankers.items()}
n_iters = {k: len(m._predictors) for k, m in ex.rankers.items()}
classic_params = int(len(ex.vec.vocabulary_) + sum(b["proto"].nnz + b["lex"].nnz for b in ex.banks.values()) + sum(trees.values()))
res = {"name": a.name, "model": a.model, "gliner": a.gliner or None, "threads": a.threads, "n_calls": len(lat),
       "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 1), "p95": round(float(np.percentile(lat, 95)), 1),
                      "mean": round(float(lat.mean()), 1)},
       "calls_per_hour_per_process": round(3600_000 / float(lat.mean()), 0),
       "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1), "load_seconds": round(load_s, 2),
       "classic_stored_values": classic_params, "tfidf_vocab": len(ex.vec.vocabulary_), "bank_values": bank,
       "tree_nodes_total": int(sum(trees.values())), "tree_nodes_visited_per_candidate": int(sum(n_iters[k] * depth[k] for k in n_iters)),
       "tfidf_nnz_per_call": float(nnz), "gpu_seconds_per_1000_calls": 0.0,
       "platform": platform.platform(), "cpu_count": os.cpu_count()}
if gl is not None:
    n = sum(p.numel() for p in gl.model.parameters())
    res["gliner_params"] = int(n)
    res["gliner_active_params_per_token"] = int(n)  # плотная модель: все параметры активны
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
