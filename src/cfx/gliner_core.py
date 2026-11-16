"""GLiNER-core: GLiNER — единственный источник фактов (бустинг-ранжировщик не участвует).

Конвейер:
  реплика → [префильтр реплик: длина + char-TF-IDF-близость к банку значений]  ← главный рычаг задержки
          → GLiNER: спаны 5 типов фактов (+ объекты и даты)
          → нормализация спана к значению каталога (char-TF-IDF, top-k)
          → p = score^a · sim^b
          → далее как раньше: ловушки §4 → объект → даты → текущее значение.

Спаны считаются один раз на конфигурацию инференса (модель × контекст) с низким порогом и
кэшируются; пороги, префильтр и веса — «дешёвые» гены эволюции, модель и контекст — «дорогие».
Оценка задержки: lat = base_ms + per_turn_ms[cfg] · (число реплик, прошедших префильтр).
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import normalize

from .text import norm, words

ATTRS = ("service_state", "tariff", "ticket_status", "action", "agreement")

GENES = {
    "gc_on": 1, "gc_cfg": "",
    "gc_thr_service_state": 0.3, "gc_thr_tariff": 0.3, "gc_thr_ticket_status": 0.3,
    "gc_thr_action": 0.3, "gc_thr_agreement": 0.3,
    "gc_topk": 1, "gc_min_sim": 0.3, "gc_a": 1.0, "gc_b": 1.0,
    "gc_pref_words": 2, "gc_pref_sim": 0.0, "gc_shift": 1, "gc_first": 0,
}
SPEC = {
    "gc_thr_service_state": ("float", 0.1, 0.9), "gc_thr_tariff": ("float", 0.1, 0.9),
    "gc_thr_ticket_status": ("float", 0.1, 0.9), "gc_thr_action": ("float", 0.1, 0.9),
    "gc_thr_agreement": ("float", 0.1, 0.9),
    "gc_topk": ("int", 1, 3), "gc_min_sim": ("float", 0.1, 0.7),
    "gc_a": ("float", 0.3, 2.0), "gc_b": ("float", 0.0, 2.0),
    "gc_pref_words": ("int", 1, 6), "gc_pref_sim": ("float", 0.0, 0.5), "gc_shift": ("bool",),
    "gc_first": ("bool",),
}
HYB_SPEC = {"gc_hyb": ("bool",), "gc_hyb_w": ("float", 0.1, 1.5), "gc_hyb_mode": ("int", 0, 1), "gc_hyb_min": ("float", 0.05, 0.6)}


def prepare(ex, view):
    """Дешёвые признаки реплик для префильтра: число слов и макс. близость к значениям банка."""
    texts = [t["text"] for t in view["turns"]]
    x = normalize(ex.vec.transform(texts))
    sim = np.zeros(len(texts), np.float32)
    for a in ATTRS:
        s = (x @ ex.banks[a]["lex"].T).toarray().max(axis=1)
        sim = np.maximum(sim, s)
    return {"n_words": np.array([len(words(t)) for t in texts]), "sim": sim,
            "idx": [t["idx"] for t in view["turns"]]}


def selected_turns(pre, g):
    keep = (pre["n_words"] >= int(g["gc_pref_words"])) & (pre["sim"] >= float(g["gc_pref_sim"]))
    return {ti for ti, k in zip(pre["idx"], keep) if k}


def core_cands(ex, scored, g):
    """Кандидаты (a, turn_idx, value, p) только из спанов GLiNER выбранной конфигурации."""
    from .text import CONFIRM, REJECT, norm as _n
    turns = scored["view"]["turns"]
    nxt = {t["idx"]: (turns[i + 1] if i + 1 < len(turns) else None) for i, t in enumerate(turns)}
    spk = {t["idx"]: t["speaker"] for t in turns}
    spans = scored["gc_spans"].get(g.get("gc_cfg") or next(iter(scored["gc_spans"])), [])
    keep = selected_turns(scored["gc_pre"], g)
    scored["_gc_n"] = len(keep)
    out, seen = [], {}
    for ti, a, text, sc in spans:
        if a not in ex.banks or ti not in keep or sc < float(g[f"gc_thr_{a}"]):
            continue
        bank = ex.banks[a]
        sims = (normalize(ex.vec.transform([text])) @ bank["lex"].T).toarray()[0]
        for j in np.argsort(-sims)[: int(g["gc_topk"])]:
            if sims[j] < float(g["gc_min_sim"]):
                break
            v = bank["values"][j]
            p = float(sc ** float(g["gc_a"]) * sims[j] ** float(g["gc_b"]))
            src = ti
            # предложение оператора + «да, подходит» клиента → источник договорённости — реплика клиента (как в эталоне)
            if int(g.get("gc_shift", 1)) and a in ("agreement", "action") and spk.get(ti) == "operator":
                nt = nxt.get(ti)
                if nt is not None and nt["speaker"] == "client" and CONFIRM.search(_n(nt["text"])) and not REJECT.search(_n(nt["text"])):
                    src = nt["idx"]
            key = (a, src, norm(v))
            if p > seen.get(key, -1):
                seen[key] = p
    if int(g.get("gc_first", 0)):
        # одно и то же значение упомянуто несколько раз → источник = первое упоминание (как в эталоне), p = максимум
        grp = {}
        for (a, ti, nv), p in seen.items():
            k = (a, nv)
            t0, p0 = grp.get(k, (ti, p))
            grp[k] = (min(t0, ti), max(p0, p))
        seen = {(a, ti, nv): p for (a, nv), (ti, p) in grp.items()}
    for (a, ti, nv), p in seen.items():
        out.append((a, ti, ex.banks[a]["values"][ex.banks[a]["index"][nv]], p))
    return out
