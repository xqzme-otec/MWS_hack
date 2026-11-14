"""Обучить селектор текущего значения на dev_A и прикрепить к модели."""
import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, pickle, time
sys.path.insert(0, os.path.dirname(__file__))
from cfx.data import splits, catalog
from cfx.current import CurrentSelector
from cfx.matching import value_match, object_match
from cfx.objects import ObjectResolver
src, dst = sys.argv[1], sys.argv[2]
ex = pickle.load(open(src, 'rb')); S = splits()
from cfx.objects import refresh; refresh(ex.objects)
genes = {**json.load(open(os.environ.get('BASE_GENES','runs/evo1_best.json'))), 'current_sel': 0, 'one_open_per_object': 0,
         'thr_service_state': 0.1, 'thr_tariff': 0.05, 'thr_ticket_status': 0.05}
t = time.time(); pairs = []
for c in S['dev_A']:
    pairs.append((c, ex.predict(c, genes)['facts']))
print('predicted dev_A', round(time.time() - t, 1), 's')
cs = CurrentSelector().fit(pairs, value_match, object_match)
print('selector trained on', cs.n_train, 'candidate facts')
ex.current = cs
pickle.dump(ex, open(dst, 'wb'))
