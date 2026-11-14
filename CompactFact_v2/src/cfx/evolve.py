"""Движок эволюции параметров экстрактора.

Геном = словарь генов со спецификацией (тип, границы). Особи оцениваются параллельно
(multiprocessing, fork: кэш кандидатов разделяется между процессами без копирования).

Фитнес выводится из критерия кейса: обе ключевые метрики — temporal_recall и
state_accuracy — должны быть не ниже «база − 0.02». Поэтому основа фитнеса —
минимальный запас по двум метрикам (maximin), а F1 фактов — вторичный член.
Шум: golden из 150 звонков даёт σ≈0.04 по этим метрикам (бутстреп), поэтому
элита периодически перепроверяется на отложенном dev (≈440 звонков), а остановка
требует выполнения цели и на golden, и на dev.
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import random
import time

# целевые значения: база v2.1 на val минус допуск 0.02
TARGET_T = 0.4706 - 0.02
TARGET_S = 0.5249 - 0.02  # строгая state_accuracy базы (без утечки эталона) − допуск

SPEC = {
    "thr_service_state": ("float", 0.05, 0.9), "thr_tariff": ("float", 0.05, 0.9),
    "thr_ticket_status": ("float", 0.05, 0.9), "thr_action": ("float", 0.05, 0.9),
    "thr_agreement": ("float", 0.05, 0.9), "max_per_turn": ("int", 1, 4),
    "trap_offer": ("bool",), "trap_question": ("bool",), "trap_imperative": ("bool",),
    "chain_mode": ("int", 0, 2), "one_open_per_object": ("bool",),
    "date_conf_agreement": ("float", 0.0, 0.8), "date_conf_state": ("float", 0.0, 0.8),
    "obj_back": ("int", 1, 10), "obj_learned": ("bool",), "drop_hidden_closed": ("bool",),
    "current_sel": ("bool",), "cur_mode": ("int", 0, 1), "cur_min_p": ("float", 0.0, 0.6),
}
# гены движка GLiNER (пороги по меткам, веса слияния) — добавляются, когда в кэше есть спаны GLiNER
SPEC_GLINER = {
    "gl_on": ("bool",), "gl_thr_service_state": ("float", 0.15, 0.9), "gl_thr_tariff": ("float", 0.15, 0.9),
    "gl_thr_ticket_status": ("float", 0.15, 0.9), "gl_thr_action": ("float", 0.15, 0.9),
    "gl_thr_agreement": ("float", 0.15, 0.9), "gl_w_boost": ("float", 0.0, 0.6), "gl_w_new": ("float", 0.0, 1.0),
    "gl_min_cov": ("float", 0.2, 1.0),
}


def fitness(m, lat_ms=None, lat_budget=1000.0):
    t = m["temporal_recall"] or 0.0
    s = m["state_accuracy"] or 0.0
    p, r = m["fact_precision"], m["fact_recall"]
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    slack = min(t - TARGET_T, s - TARGET_S)
    pen = 0.0 if lat_ms is None or lat_ms <= lat_budget else 0.05 * (lat_ms / lat_budget - 1)
    return slack + 0.15 * f1 - pen


def random_genome(spec, rng, base=None):
    g = dict(base or {})
    for k, sp in spec.items():
        if base is not None and k in base and rng.random() < 0.5:
            continue
        g[k] = _sample(sp, rng)
    return g


def _sample(sp, rng):
    if sp[0] == "float":
        return round(rng.uniform(sp[1], sp[2]), 3)
    if sp[0] == "int":
        return rng.randint(sp[1], sp[2])
    if sp[0] == "bool":
        return int(rng.random() < 0.5)
    return rng.choice(sp[1])


def mutate(g, spec, rng, rate=0.25, sigma=0.12):
    c = dict(g)
    for k, sp in spec.items():
        if rng.random() > rate:
            continue
        if sp[0] == "float":
            c[k] = round(min(sp[2], max(sp[1], c[k] + rng.gauss(0, sigma * (sp[2] - sp[1])))), 3)
        elif sp[0] == "int":
            c[k] = min(sp[2], max(sp[1], c[k] + rng.choice((-1, 1))))
        elif sp[0] == "bool":
            c[k] = 1 - c[k]
        else:
            c[k] = rng.choice(sp[1])
    return c


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


# ----------------------- параллельная оценка -----------------------
_BENCH = None


def _eval(genes):
    m, _ = _BENCH.evaluate(genes)
    return m


def evolve(bench, confirm_bench=None, spec=SPEC, base=None, pop=16, gens=20, workers=2, seed=42,
           confirm_every=4, log_path=None, stop_margin=0.0, log=print):
    """Возвращает (лучший геном, история). Оценка особей — параллельно в `workers` процессах."""
    global _BENCH
    _BENCH = bench
    rng = random.Random(seed)
    population = [dict(base)] + [random_genome(spec, rng, base) for _ in range(pop - 1)]
    history, best, best_fit = [], None, -math.inf
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for gen in range(gens):
            t0 = time.time()
            metrics = pool.map(_eval, population)
            fits = [fitness(m) for m in metrics]
            order = sorted(range(len(population)), key=lambda i: -fits[i])
            gbest = order[0]
            rec = {"gen": gen, "best_fit": round(fits[gbest], 4), "mean_fit": round(sum(fits) / len(fits), 4),
                   "best_metrics": metrics[gbest], "best_genes": population[gbest], "seconds": round(time.time() - t0, 1),
                   "population": [{"fit": round(fits[i], 4), "m": metrics[i]} for i in order]}
            if fits[gbest] > best_fit:
                best_fit, best = fits[gbest], dict(population[gbest])
            # перепроверка элиты на большом отложенном срезе (защита от переобучения на golden)
            if confirm_bench is not None and (gen % confirm_every == confirm_every - 1 or gen == gens - 1):
                elite = [population[i] for i in order[:3]]
                cm = [confirm_bench.evaluate(e)[0] for e in elite]
                cf = [fitness(m) for m in cm]
                k = max(range(len(elite)), key=lambda i: cf[i])
                rec["confirm"] = {"metrics": cm[k], "fit": round(cf[k], 4), "genes": elite[k]}
                best = dict(elite[k])
                best_fit = fits[order[k]]
            history.append(rec)
            bm = metrics[gbest]
            log(f"gen {gen:2d} best={fits[gbest]:+.4f} mean={rec['mean_fit']:+.4f} T={bm['temporal_recall']} "
                f"S={bm['state_accuracy']} P={bm['fact_precision']} R={bm['fact_recall']} ({rec['seconds']}s)"
                + (f" | dev T={rec['confirm']['metrics']['temporal_recall']} S={rec['confirm']['metrics']['state_accuracy']}"
                   if "confirm" in rec else ""))
            if log_path:
                json.dump(history, open(log_path, "w"), ensure_ascii=False, indent=1)
            # условие остановки: цель достигнута на golden и подтверждена на dev
            if "confirm" in rec:
                cmm = rec["confirm"]["metrics"]
                if (cmm["temporal_recall"] or 0) >= TARGET_T + stop_margin and (cmm["state_accuracy"] or 0) >= TARGET_S + stop_margin:
                    log(f"  цель достигнута на dev (T≥{TARGET_T + stop_margin:.3f}, S≥{TARGET_S + stop_margin:.3f}) — стоп")
                    break
            # следующее поколение: элитизм + турнир + скрещивание + мутация
            elite_n = max(2, pop // 4)
            nxt = [dict(population[i]) for i in order[:elite_n]]
            while len(nxt) < pop:
                a = population[max(rng.sample(range(len(population)), 3), key=lambda i: fits[i])]
                b = population[max(rng.sample(range(len(population)), 3), key=lambda i: fits[i])]
                child = mutate(crossover(a, b, rng), spec, rng)
                nxt.append(child)
            population = nxt
    return best, history
