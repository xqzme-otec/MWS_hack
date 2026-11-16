"""Сквозная оценка финального решения официальным оценщиком case_metrics.py (v2.1).

Прогоняет конвейер (GLiNER на всех репликах + бустинг + сборка) по звонкам набора, пишет предсказания,
считает метрики кейса, проверяет схему на 100 % звонков и меряет задержку.

    python src/evaluate.py                                   # val
    python src/evaluate.py --sets val,val_adv --mode hybrid  # + adversarial
    python src/evaluate.py --limit 20                        # быстрый смоук-тест
"""
import argparse, json, os, sys, time

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here); sys.path.insert(0, os.path.join(here, "..", "data"))
ROOT = os.path.join(here, "..")
SETS = {"val": "data/val.jsonl", "val_adv": "data/val_adv_0.3.jsonl"}


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="val")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "gliner", "boosting"])
    ap.add_argument("--gliner", default="models/g13_ru")
    ap.add_argument("--genes", default="models/genes_gliner_hybrid.json")
    ap.add_argument("--boost", default="models/compactfact_v5.pkl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out_dir", default="reports/eval")
    ap.add_argument("--tag", default="", help="имя варианта в файлах отчёта (по умолчанию = mode)")
    a = ap.parse_args()
    import case_metrics as cm
    from cfx.pipeline import Pipeline
    os.makedirs(os.path.join(ROOT, a.out_dir), exist_ok=True)
    P = Pipeline(a.gliner, a.boost, a.genes, mode=a.mode)
    summary = {}
    for name in a.sets.split(","):
        path = os.path.join(ROOT, SETS.get(name, name))
        if not os.path.exists(path):
            if name == "val_adv":
                os.system(f"{sys.executable} {os.path.join(here, 'make_adversarial.py')} --out {path}")
            else:
                sys.exit(f"нет файла {path}: положите данные кейса в data/ (см. README)")
        calls = [json.loads(l) for l in open(path, encoding="utf-8")]
        if a.limit:
            calls = calls[:a.limit]
        P.extract(calls[0])                                   # прогрев
        preds, lat, bad = {}, [], 0
        t0 = time.time()
        for i, c in enumerate(calls):
            o = P.extract(c)
            lat.append(o["meta"]["latency_ms"]); bad += bool(o["meta"]["schema_errors"])
            preds[c["call_id"]] = {"call_id": c["call_id"], "facts": o["facts"]}
            if (i + 1) % 50 == 0:
                print(f"  {name}: {i + 1}/{len(calls)}  {time.time() - t0:.0f} c", flush=True)
        cm.load_catalog(os.path.join(ROOT, "data", "val.jsonl"))
        gold = {c["call_id"]: c for c in calls}
        m = cm.evaluate(gold, preds)
        m.update({"schema_valid_calls": f"{len(calls) - bad}/{len(calls)}",
                  "latency_ms_p50": round(pct(lat, 0.5), 1), "latency_ms_p95": round(pct(lat, 0.95), 1),
                  "latency_ms_mean": round(sum(lat) / len(lat), 1), "threads": int(os.environ.get("CFX_THREADS", "2")),
                  "mode": a.mode, "variant": a.tag or a.mode, "genes": a.genes, "gliner": a.gliner})
        tag = f"{a.tag or a.mode}_{name}" + (f"_n{a.limit}" if a.limit else "")
        with open(os.path.join(ROOT, a.out_dir, f"{tag}_pred.jsonl"), "w", encoding="utf-8") as fo:
            for p in preds.values():
                fo.write(json.dumps(p, ensure_ascii=False) + "\n")
        json.dump(m, open(os.path.join(ROOT, a.out_dir, f"{tag}_metrics.json"), "w"), ensure_ascii=False, indent=1)
        summary[name] = m
        print(f"\n== {name} ({a.mode}) ==")
        for k, v in m.items():
            print(f"  {k:20s} {v}")
    return summary


if __name__ == "__main__":
    main()
