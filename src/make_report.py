"""Сводка результатов: reports/summary.json (для веб-интерфейса), reports/RESULTS.md и графики
кривой «ресурсы × качество» (все варианты системы, а не одна лучшая точка) и эволюции архитектуры.

Источники (все — артефакты прогонов в этом репозитории):
  reports/eval/<вариант>_<набор>_metrics.json   — evaluate.py (сквозной прогон, официальный оценщик)
  reports/resources.json                         — bench_resources.py (единая методика замеров)
  reports/arch_evolution.json                    — эволюция архитектуры GLiNER (Colab)
  data/baseline_pred.jsonl                        — базовый LLM-экстрактор кейса
"""
import json, os, sys

here = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(here, ".."))
sys.path.insert(0, os.path.join(ROOT, "data"))
R = os.path.join(ROOT, "reports")
NAMES = {"boosting": "Бустинг (без GLiNER)", "gliner": "GLiNER g13-ru", "hybrid_prefilter": "Гибрид, GLiNER на ~⅓ реплик",
         "hybrid": "Гибрид, GLiNER на всех репликах ★", "hybrid_fullvocab": "Гибрид, полный словарь mT5 (160M)"}
BASE_LLM = {"name": "базовый LLM-экстрактор (glm-5.3)", "params_b": 744, "note": "семейство GLM-5: 744B параметров, 40B активных (публичные данные)"}


def jl(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def baseline_metrics():
    p = os.path.join(ROOT, "data", "baseline_pred.jsonl")
    g = os.path.join(ROOT, "data", "val.jsonl")
    if not (os.path.exists(p) and os.path.exists(g)):
        return jl(os.path.join(R, "baseline_metrics.json"))
    import case_metrics as cm
    cm.load_catalog(g)
    gold = {c["call_id"]: c for c in map(json.loads, open(g, encoding="utf-8"))}
    pred = {c["call_id"]: c for c in map(json.loads, open(p, encoding="utf-8"))}
    m = cm.evaluate(gold, pred)
    json.dump(m, open(os.path.join(R, "baseline_metrics.json"), "w"), indent=1)
    return m


def main():
    base = baseline_metrics()
    res = jl(os.path.join(R, "resources.json")) or {"rows": [], "machine": {}}
    rr = {r["variant"]: r for r in res["rows"]}
    rows = []
    for v in ["boosting", "gliner", "hybrid_prefilter", "hybrid", "hybrid_fullvocab"]:
        mv = jl(os.path.join(R, "eval", f"{v}_val_metrics.json"))
        if v == "hybrid_fullvocab":   # те же предсказания, что у hybrid (урезание словаря не меняет вывод — проверено)
            mv = jl(os.path.join(R, "eval", "hybrid_val_metrics.json"))
        ma = jl(os.path.join(R, "eval", f"{'hybrid' if v == 'hybrid_fullvocab' else v}_val_adv_metrics.json"))
        if not mv:
            continue
        r = rr.get(v, {})
        rows.append({"variant": v, "name": NAMES[v], "T": mv["temporal_recall"], "S": mv["state_accuracy"],
                     "P": mv["fact_precision"], "R": mv["fact_recall"], "grounding_turn": mv["grounding_turn"],
                     "grounding_span": mv.get("grounding_span"), "schema": mv.get("schema_valid_calls"),
                     "T_adv": ma and ma["temporal_recall"], "S_adv": ma and ma["state_accuracy"],
                     "lat_p50": r.get("latency_ms_p50"), "lat_p95": r.get("latency_ms_p95"), "rss_mb": r.get("peak_rss_mb"),
                     "params_m": r.get("params_total_m"), "disk_mb": r.get("disk_mb"), "gpu_s": 0.0})
    hy = next((x for x in rows if x["variant"] == "hybrid"), None)
    summary = {"headline": {}, "variants": rows, "baseline": base, "machine": res.get("machine"),
               "baseline_llm": BASE_LLM, "thresholds": None}
    if base:
        summary["thresholds"] = {"temporal_recall": round(base["temporal_recall"] - 0.02, 4),
                                 "state_accuracy": round(base["state_accuracy"] - 0.02, 4),
                                 "temporal_recall_reject": round(base["temporal_recall"] - 0.05, 4)}
    if hy:
        summary["headline"] = {"temporal_recall": hy["T"], "state_accuracy": hy["S"], "params_m": hy["params_m"] or 51.6,
                               "latency_ms_p50": hy["lat_p50"], "temporal_recall_adv": hy["T_adv"], "state_accuracy_adv": hy["S_adv"]}
    json.dump(summary, open(os.path.join(R, "summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    plots(rows, base)
    md(summary)
    print(open(os.path.join(R, "RESULTS.md"), encoding="utf-8").read()[:3000])


def plots(rows, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    pts = [r for r in rows if r["lat_p50"] and r["variant"] != "hybrid_fullvocab"]   # тот же вывод, что у hybrid — только в таблице
    if pts:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        for k, (key, lab, col) in enumerate([("T", "temporal_recall", "#e30611"), ("S", "state_accuracy", "#2563eb")]):
            a = ax[k]
            xs = [r["lat_p50"] for r in pts]; ys = [r[key] for r in pts]
            for r in pts:
                star = r["variant"] == "hybrid"
                a.scatter(r["lat_p50"], r[key], s=170 if star else 70, color=col, edgecolor="k" if star else "none", zorder=3,
                          marker="*" if star else "o")
                a.annotate(r["name"].replace(" ★", ""), (r["lat_p50"], r[key]), textcoords="offset points", xytext=(-150, 8) if r["variant"] == "hybrid" else (6, 6), fontsize=8)
                if r.get(key + "_adv"):
                    a.scatter(r["lat_p50"], r[key + "_adv"], s=40, facecolor="none", edgecolor=col, zorder=3)
            if base:
                a.axhline(base[lab], color="#666", ls="--", lw=1); a.text(min(xs), base[lab] + .002, "базовый LLM", fontsize=8, color="#666")
                a.axhline(base[lab] - .02, color="#d97706", ls=":", lw=1.2); a.text(min(xs), base[lab] - .018, "порог (база − 0.02)", fontsize=8, color="#d97706")
            a.axvline(1000, color="#999", lw=.8, ls="-."); a.text(1010, a.get_ylim()[0] + .005, "1 с", fontsize=8, color="#999")
            a.set_xlabel("задержка CPU, мс на звонок (p50, 2 потока)"); a.set_ylabel(lab)
            a.set_title(f"{lab}: ● val   ○ adversarial (ASR-шум 0.3)", fontsize=10)
        fig.suptitle("Кривая «ресурсы × качество»: все варианты системы, единая методика замера", fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(R, "curve_quality_latency.png"), dpi=150); plt.close(fig)

        fig, a = plt.subplots(figsize=(7.5, 4.2))
        for r in pts:
            a.scatter(r["params_m"] * 1e6, r["T"], color="#e30611", s=120 if r["variant"] == "hybrid" else 50, marker="*" if r["variant"] == "hybrid" else "o", zorder=3)
            a.annotate(r["name"].replace(" ★", ""), (r["params_m"] * 1e6, r["T"]), textcoords="offset points", xytext=(5, 5), fontsize=8)
        if base:
            a.scatter(BASE_LLM["params_b"] * 1e9, base["temporal_recall"], color="#444", s=70, marker="s")
            a.annotate("базовый LLM ≈744B", (BASE_LLM["params_b"] * 1e9, base["temporal_recall"]), textcoords="offset points", xytext=(-90, 8), fontsize=8)
            a.scatter(40e9, base["temporal_recall"], color="#999", s=40, marker="s")
            a.annotate("(40B активных)", (40e9, base["temporal_recall"]), textcoords="offset points", xytext=(-40, -14), fontsize=7, color="#777")
        a.set_xscale("log"); a.set_xlabel("параметров (лог. шкала)"); a.set_ylabel("temporal_recall (val)")
        a.set_title("Качество vs размер модели", fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(R, "curve_quality_params.png"), dpi=150); plt.close(fig)

    ae = jl(os.path.join(R, "arch_evolution.json"))
    if ae:
        fig, a = plt.subplots(figsize=(7.5, 4.4))
        cmap = plt.get_cmap("viridis")
        for r in ae["rows"]:
            f = 0.5 * ((r["span_f1_dev"] or 0) + (r["span_f1_dev_noise"] or 0))
            c = cmap((r["layers"] - 3) / 9)
            a.scatter(r["cpu_ms_per_call"], f, s=r["params_m"] * 1.4, color=c, alpha=.85, edgecolor="k" if r["tag"] == "g1_3" else "none", lw=2)
            a.annotate(f"{r['tag']}: {r['layers']}сл{' +LSTM' if r['rnn'] else ''}, w{r['max_width']}, ctx{r['ctx']}", (r["cpu_ms_per_call"], f),
                       textcoords="offset points", xytext=(6, 2), fontsize=7)
        a.set_xlabel("задержка CPU Colab, мс на звонок (медленная машина)"); a.set_ylabel("span-F1 (чистый + шум)/2, dev")
        a.set_title("Эволюция архитектуры GLiNER: 14 кандидатов, размер точки = параметры\nвыбран g1_3 (3 слоя + BiLSTM, ширина 6) → дообучен как g13", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(R, "curve_architecture.png"), dpi=150); plt.close(fig)


def md(s):
    f = lambda x: "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
    i = lambda x: "—" if x is None else f"{x:.0f}"
    L = ["# Результаты", "", f"Машина замеров: {s['machine'].get('cpu', '?')}, {s['machine'].get('cores', '?')} ядер; 2 потока torch; GPU не используется.", ""]
    b = s["baseline"]
    if b:
        L += ["## Критерий 1: качество против базового LLM-экстрактора", "",
              "| | temporal_recall | state_accuracy |", "|---|---|---|",
              f"| базовый LLM-экстрактор (`baseline_pred.jsonl`) | {f(b['temporal_recall'])} | {f(b['state_accuracy'])} |",
              f"| допустимый минимум (база − 0.02) | {f(s['thresholds']['temporal_recall'])} | {f(s['thresholds']['state_accuracy'])} |"]
        h = s["headline"]
        if h:
            L.append(f"| **CompactFact (гибрид)** | **{f(h['temporal_recall'])}** ({h['temporal_recall'] - b['temporal_recall']:+.3f}) | "
                     f"**{f(h['state_accuracy'])}** ({h['state_accuracy'] - b['state_accuracy']:+.3f}) |")
        L.append("")
    L += ["## Все варианты системы (кривая «ресурсы × качество»)", "",
          "| вариант | T val | S val | T adv | S adv | P | R | реплика | спан | схема | p50 мс | p95 мс | RSS МБ | параметры | диск МБ | GPU-с |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["variants"]:
        L.append(f"| {r['name']} | {f(r['T'])} | {f(r['S'])} | {f(r['T_adv'])} | {f(r['S_adv'])} | {f(r['P'])} | {f(r['R'])} | "
                 f"{f(r['grounding_turn'])} | {f(r['grounding_span'])} | {r['schema'] or '—'} | {i(r['lat_p50'])} | {i(r['lat_p95'])} | "
                 f"{r['rss_mb'] or '—'} | {(str(r['params_m']) + ' M') if r['params_m'] else '—'} | {i(r['disk_mb'])} | 0 |")
    if b:
        L.append(f"| базовый LLM-экстрактор | {f(b['temporal_recall'])} | {f(b['state_accuracy'])} | — | — | {f(b['fact_precision'])} | "
                 f"{f(b['fact_recall'])} | {f(b['grounding_turn'])} | — | — | — | — | — | ≈744 000 M¹ | — | GPU |")
        L += ["", f"¹ {s['baseline_llm']['note']}."]
    L += ["", "![кривая](curve_quality_latency.png)", "", "![размер](curve_quality_params.png)", "", "![архитектура](curve_architecture.png)", ""]
    open(os.path.join(R, "RESULTS.md"), "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
