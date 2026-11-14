"""Устойчивость к ASR-шуму: val с дополнительным шумом (+0.15, +0.3) и эффект шумоподавителя."""
import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, pickle, copy
sys.path.insert(0, os.path.dirname(__file__))
from cfx.bench import Bench
from cfx.objects import refresh
model, genes_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
ex = pickle.load(open(model, 'rb')); refresh(ex.objects)
genes = json.load(open(genes_path))
D = pickle.load(open('runs/denoiser.pkl', 'rb'))
res = {}
for name in ('data/val.jsonl', 'runs/val_adv_0.15.jsonl', 'runs/val_adv_0.3.jsonl'):
    calls = [json.loads(l) for l in open(name)]
    b = Bench(ex, calls, name); b.official = True
    m, _ = b.evaluate(genes); res[name] = m
    # шумоподавитель на входе (модель обучена на зашумлённом тексте — проверяем как есть)
    dn = copy.deepcopy(calls)
    for c in dn:
        for t in c['turns']:
            t['text'] = D(t['text'])
    bd = Bench(ex, dn, name + '+denoise'); bd.official = True
    md, _ = bd.evaluate(genes); res[name + ' +denoise'] = md
    print(name, m, '\n   +denoise', md, flush=True)
json.dump(res, open(out, 'w'), ensure_ascii=False, indent=1)
