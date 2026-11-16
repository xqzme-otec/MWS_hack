"""Обёртка над официальным case_metrics.py v2.1 (без изменений логики)."""
import json, sys, random, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data'))
import case_metrics as cm
cm.load_catalog(os.path.join(os.path.dirname(__file__), '..', 'data', 'val.jsonl'))

def load(path):
    return {c['call_id']: c for c in map(json.loads, open(path, encoding='utf-8'))}

def evaluate(gold, pred):
    return cm.evaluate(gold, pred)

def bootstrap(gold, pred, n_boot=300, size=None, seed=0, keys=('temporal_recall','state_accuracy','fact_precision','fact_recall')):
    """Бутстреп по звонкам: распределение метрик при выборке size звонков."""
    rng = random.Random(seed); ids = list(gold); size = size or len(ids)
    out = {k: [] for k in keys}
    for _ in range(n_boot):
        s = [rng.choice(ids) for _ in range(size)]
        g = {f"{i}#{j}": gold[c] for j, c in enumerate(s) for i in [c]}
        p = {f"{i}#{j}": pred.get(c, {'facts': []}) for j, c in enumerate(s) for i in [c]}
        m = cm.evaluate(g, p)
        for k in keys:
            if m[k] is not None: out[k].append(m[k])
    return out

# --- Баг v2.1: recover_dates падает на «позапрошл…» и «через две недели» (set.update(date, date)).
# Для анализа train используем копию scorer_fixed.py (исправлены только эти две строки).
import importlib
def fixed():
    import scorer_fixed as sf
    sf.load_catalog(os.path.join(os.path.dirname(__file__), '..', 'data', 'val.jsonl'))
    return sf


def strict():
    """Оценщик v2.1 с исправленными багами (TypeError) и без утечки эталона в state_accuracy."""
    import scorer_strict as ss
    ss.load_catalog(os.path.join(os.path.dirname(__file__), '..', 'data', 'val.jsonl'))
    return ss
