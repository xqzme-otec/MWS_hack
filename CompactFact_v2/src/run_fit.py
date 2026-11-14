"""Обучить экстрактор с заданными генами, оценить на golden и val официальным оценщиком."""
import sys, os, json, time, pickle, argparse
sys.path.insert(0, os.path.dirname(__file__))
from cfx.data import splits, catalog
from cfx.extractor import Extractor
from cfx.matching import value_match
from metrics_util import fixed, evaluate as official_eval
import metrics_util as mu
ap = argparse.ArgumentParser(); ap.add_argument('--genes', default='{}'); ap.add_argument('--out', default='runs/base.pkl'); ap.add_argument('--val', action='store_true')
a = ap.parse_args()
S = splits(); sf = fixed()
t = time.time()
ex = Extractor(catalog(), json.loads(a.genes)).fit(S['train'], value_match)
print('fit', round(time.time() - t, 1), 's')
def run(calls, use_official):
    t = time.time(); pred = {c['call_id']: ex.predict(c) for c in calls}; dt = (time.time() - t) / len(calls)
    gold = {c['call_id']: c for c in calls}
    m = (official_eval if use_official else sf.evaluate)(gold, pred)
    return m, dt, pred
m, dt, _ = run(S['golden'], False)
print('golden', {k: m[k] for k in ('fact_precision', 'fact_recall', 'temporal_recall', 'n_temporal', 'state_accuracy', 'grounding_turn')}, f'{dt*1000:.0f} ms/call')
if a.val:
    m, dt, pred = run(S['val'], True)
    print('VAL', {k: m[k] for k in ('fact_precision', 'fact_recall', 'temporal_recall', 'n_temporal', 'state_accuracy', 'grounding_turn')}, f'{dt*1000:.0f} ms/call')
    with open(a.out.replace('.pkl', '_val.jsonl'), 'w') as fh:
        for c in pred.values(): fh.write(json.dumps(c, ensure_ascii=False) + '\n')
pickle.dump(ex, open(a.out, 'wb'))
