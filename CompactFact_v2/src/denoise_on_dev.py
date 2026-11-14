"""Решение «шумоподавитель вкл/выкл» по dev (golden + dev_B), не по val."""
import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, pickle, copy
sys.path.insert(0, os.path.dirname(__file__))
from cfx.bench import Bench
from cfx.data import splits
from cfx.objects import refresh
ex = pickle.load(open("runs/v5.pkl", "rb")); refresh(ex.objects); S = splits()
genes = json.load(open(sys.argv[1])); D = pickle.load(open("runs/denoiser.pkl", "rb"))
calls = S["golden"] + S["dev_B"]
dn = copy.deepcopy(calls)
for c in dn:
    for t in c["turns"]: t["text"] = D(t["text"])
res = {}
for name, cs in (("dev без шумоподавителя", calls), ("dev + шумоподавитель", dn)):
    b = Bench(ex, cs, name); m, _ = b.evaluate(genes); res[name] = m; print(name, m, flush=True)
json.dump(res, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
