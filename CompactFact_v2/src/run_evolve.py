"""Эволюция дешёвых генов на кэше golden (150) с перепроверкой элиты на dev_rest."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, os, json, pickle, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
from cfx.bench import Bench
from cfx.data import splits
from cfx.evolve import evolve, fitness
from cfx.extractor import CHEAP
ap = argparse.ArgumentParser(); ap.add_argument('--model', default='runs/v2.pkl'); ap.add_argument('--tag', default='evo1')
ap.add_argument('--pop', type=int, default=16); ap.add_argument('--gens', type=int, default=12); ap.add_argument('--workers', type=int, default=2); ap.add_argument('--confirm', default='dev_B'); ap.add_argument('--base', default='')
a = ap.parse_args()
ex = pickle.load(open(a.model, 'rb')); S = splits()
from cfx.objects import refresh; refresh(ex.objects)
t = time.time(); golden = Bench(ex, S['golden'], 'golden'); dev = Bench(ex, S[a.confirm], a.confirm)
print(f'cache: golden {golden.score_ms:.0f} ms/call, dev {dev.score_ms:.0f} ms/call, {time.time()-t:.0f}s total', flush=True)
t = time.time(); m, _ = golden.evaluate(dict(CHEAP)); print('one eval on golden', round(time.time() - t, 2), 's', m, flush=True)
base = {**CHEAP, **(json.load(open(a.base)) if a.base else {})}
best, hist = evolve(golden, dev, base=base, pop=a.pop, gens=a.gens, workers=a.workers, log_path=f'runs/{a.tag}_history.json', log=lambda s: print(s, flush=True))
json.dump(best, open(f'runs/{a.tag}_best.json', 'w'), ensure_ascii=False, indent=1)
val = Bench(ex, S['val'], 'val'); val.official = True; mv, pred = val.evaluate(best)
print('VAL best genes:', mv, 'fit', round(fitness(mv), 4))
with open(f'runs/{a.tag}_val.jsonl', 'w') as fh:
    for c in pred.values(): fh.write(json.dumps(c, ensure_ascii=False) + '\n')
pickle.dump(val, open(f'runs/{a.tag}_valbench.pkl', 'wb'))
