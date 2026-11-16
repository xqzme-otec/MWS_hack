"""Замер ресурсов по единой методике для всех вариантов системы.

Методика (одинакова для всех строк):
  * один и тот же набор звонков (первые N из data/val.jsonl, иначе examples/calls.jsonl);
  * каждый вариант — в отдельном процессе (чистая память), CPU, torch.set_num_threads(T) (по умолчанию 2);
  * 3 прогревочных звонка не учитываются; задержка = время конвейера на звонок целиком
    (GLiNER + бустинг + сборка), p50 / p95 / среднее;
  * память — пиковый RSS процесса (ru_maxrss) после загрузки и прогона;
  * параметры — уникальные тензоры GLiNER (связанные веса считаются один раз) + узлы деревьев бустинга;
  * GPU-секунды на звонок — 0 (GPU не используется).

    python src/bench_resources.py --n 60 --threads 2
→ reports/resources.json (+ печать таблицы)
"""
import argparse, json, os, platform, resource, subprocess, sys, time

here = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(here, ".."))

VARIANTS = {
    "boosting":          {"mode": "boosting", "gliner": None, "genes": "models/genes_best.json"},
    "gliner":            {"mode": "gliner", "gliner": "models/g13_ru", "genes": "models/genes_gliner_hybrid.json"},
    "hybrid_prefilter":  {"mode": "hybrid", "gliner": "models/g13_ru", "genes": "models/genes_gliner_hybrid_prefilter.json"},
    "hybrid":            {"mode": "hybrid", "gliner": "models/g13_ru", "genes": "models/genes_gliner_hybrid.json"},
    "hybrid_fullvocab":  {"mode": "hybrid", "gliner": "models/g13", "genes": "models/genes_gliner_hybrid.json"},
}


def child(name, n, threads):
    import torch
    sys.path.insert(0, here)
    from cfx.pipeline import Pipeline
    v = VARIANTS[name]
    torch.set_num_threads(threads)
    t = time.time()
    P = Pipeline(v["gliner"] or "models/g13_ru", genes=v["genes"], mode=v["mode"], threads=threads)
    load_s = time.time() - t
    path = os.path.join(ROOT, "data", "val.jsonl")
    if not os.path.exists(path):
        path = os.path.join(ROOT, "examples", "calls.jsonl")
    calls = [json.loads(l) for l in open(path, encoding="utf-8")][:n + 3]
    lat, turns = [], 0
    for i, c in enumerate(calls):
        o = P.extract(c)
        if i >= 3:
            lat.append(o["meta"]["latency_ms"]); turns += len(c["turns"])
    lat.sort()
    params = 0
    disk = 0
    if P.glc is not None:
        seen = set()
        for p in P.model.parameters():
            if p.data_ptr() not in seen:
                seen.add(p.data_ptr()); params += p.numel()
        disk += os.path.getsize(os.path.join(P.gliner_dir, "pytorch_model.bin"))
    trees = 0
    for m in _boosters(P.ex):
        for pred in getattr(m, "_predictors", []):
            for tr in pred:
                trees += len(tr.nodes)
    disk += os.path.getsize(os.path.join(ROOT, "models", "compactfact_v5.pkl"))
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return {"variant": name, "n_calls": len(lat), "mean_turns": round(turns / len(lat), 1), "threads": threads,
            "latency_ms_p50": round(lat[len(lat) // 2], 1), "latency_ms_p95": round(lat[int(0.95 * (len(lat) - 1))], 1),
            "latency_ms_mean": round(sum(lat) / len(lat), 1), "load_s": round(load_s, 1),
            "peak_rss_mb": round(rss_mb), "gliner_params": params, "boosting_tree_nodes": trees,
            "params_total_m": round((params + trees) / 1e6, 2), "disk_mb": round(disk / 2 ** 20, 1),
            "gpu_s_per_call": 0.0}


def _boosters(o, depth=0, seen=None):
    """Все градиентные бустинги внутри экстрактора (ранжировщики, даты, объекты, текущее значение)."""
    seen = [] if seen is None else seen
    if depth > 4:
        return seen
    if hasattr(o, "_predictors"):
        if all(o is not x for x in seen):
            seen.append(o)
        return seen
    d = vars(o) if hasattr(o, "__dict__") else (o if isinstance(o, dict) else None)
    if isinstance(o, (list, tuple)):
        d = dict(enumerate(o))
    for v in (d or {}).values():
        if not isinstance(v, (str, int, float, bytes)):
            _boosters(v, depth + 1, seen)
    return seen


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--child", default="")
    ap.add_argument("--out", default="reports/resources.json")
    a = ap.parse_args()
    if a.child:
        print("RESULT " + json.dumps(child(a.child, a.n, a.threads)), flush=True)
        return
    rows = []
    for name in a.variants.split(","):
        g = VARIANTS[name]["gliner"]
        if g and not os.path.exists(os.path.join(ROOT, g)):
            print(f"[пропуск] {name}: нет {g}"); continue
        env = {**os.environ, "OMP_NUM_THREADS": str(a.threads), "CFX_THREADS": str(a.threads)}
        p = subprocess.run([sys.executable, __file__, "--child", name, "--n", str(a.n), "--threads", str(a.threads)],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(p.stderr[-2000:]); continue
        r = json.loads(line[0][7:]); rows.append(r)
        print(f"{name:18s} p50 {r['latency_ms_p50']:7.1f} мс  p95 {r['latency_ms_p95']:7.1f}  RSS {r['peak_rss_mb']:5d} МБ  "
              f"params {r['params_total_m']:6.2f} M  disk {r['disk_mb']:6.1f} МБ", flush=True)
    out = {"method": __doc__.split("Методика")[1].split("python src")[0].strip(), "machine": {
        "cpu": cpu_name(), "cores": os.cpu_count(), "python": platform.python_version(), "os": platform.platform()},
        "rows": rows}
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.out)), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, a.out), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
