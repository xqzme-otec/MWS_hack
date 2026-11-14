"""Стенд быстрой оценки: кандидаты считаются один раз, «дешёвые» гены применяются к кэшу."""
import os, sys, pickle, time
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
