"""Стенд быстрой оценки: кандидаты считаются один раз, «дешёвые» гены применяются к кэшу."""
import os, sys, pickle, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from metrics_util import fixed, strict
_sf = fixed()
_ss = strict()
KEYS = ('fact_precision', 'fact_recall', 'temporal_recall', 'n_temporal', 'state_accuracy', 'grounding_turn')


class Bench:
    def __init__(self, extractor, calls, name='set'):
        self.ex, self.name = extractor, name
        self.official = False
        self.gold = {c['call_id']: c for c in calls}
        t = time.time()
        self.scored = {c['call_id']: extractor.score(c) for c in calls}
        self.score_ms = (time.time() - t) * 1000 / max(len(calls), 1)
        self.date_cache = {cid: {} for cid in self.gold}

    def predict(self, genes):
        return {cid: self.ex.assemble(s, genes, self.date_cache[cid]) for cid, s in self.scored.items()}

    def evaluate(self, genes, ids=None):
        pred = self.predict(genes)
        gold = self.gold if ids is None else {k: self.gold[k] for k in ids}
        m = _ss.evaluate(gold, pred)          # фитнес — по строгой версии (без утечки эталона)
        out = {k: m[k] for k in KEYS}
        if self.official:
            mo = _sf.evaluate(gold, pred)
            out["state_accuracy_official"] = mo["state_accuracy"]
        return out, pred


class GCBench(Bench):
    """Стенд для GLiNER-core: бустинг не считается; спаны GLiNER — из кэша по конфигурациям.

    spans: {cfg: {call_id: [(turn_idx, attr, text, score)]}}; per_turn_ms: {cfg: мс на реплику};
    в метрики добавляется оценка задержки lat_ms (base_ms + per_turn_ms · реплик после префильтра).
    """
    def __init__(self, extractor, calls, spans, per_turn_ms, base_ms=60.0, name='set', hybrid=False):
        from .gliner_core import prepare
        from .text import inference_view
        self.ex, self.name, self.official = extractor, name, False
        self.gold = {c['call_id']: c for c in calls}
        self.per_turn_ms, self.base_ms = per_turn_ms, base_ms
        self.scored = {}
        for c in calls:
            v = inference_view(c)
            boost = extractor.score(c)["cands"] if hybrid else []     # кандидаты бустинга для гибрида
            self.scored[c['call_id']] = {"view": v, "cands": boost, "obj": extractor.objects.prepare(v),
                                         "spk": [t["speaker"] for t in v["turns"]], "gc_pre": prepare(extractor, v),
                                         "gc_spans": {k: sp.get(c['call_id'], []) for k, sp in spans.items()}}
        self.date_cache = {cid: {} for cid in self.gold}

    def evaluate(self, genes, ids=None):
        g = {"gc_on": 1, **genes}
        out, pred = Bench.evaluate(self, g, ids)
        cfg = g.get("gc_cfg") or next(iter(self.per_turn_ms))
        n = np.mean([s.get("_gc_n", 0) for s in self.scored.values()])
        out["lat_ms"] = round(self.base_ms + self.per_turn_ms[cfg] * n + (160.0 if int(g.get("gc_hyb", 0)) else 0.0), 1)
        out["turns_run"] = round(float(n), 2)
        return out, pred
