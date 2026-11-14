import os; os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, sys, json, time, pickle
sys.path.insert(0, os.path.dirname(__file__))
from cfx.data import splits, catalog
from cfx.extractor import Extractor
from cfx.matching import value_match

ap = argparse.ArgumentParser(description="Обучение экстрактора (cross-fit) → runs/v4.pkl")
ap.add_argument("--extra", default="", help="jsonl с дистиллированными звонками (runs/distilled_10k.jsonl)")
ap.add_argument("--extra_n", type=int, default=0, help="взять первые N из --extra (0 = все)")
ap.add_argument("--out", default="runs/v4.pkl")
a = ap.parse_args()

S = splits(); t = time.time()
extra = []
if a.extra:
    extra = [json.loads(l) for l in open(a.extra, encoding="utf-8")]
    extra = extra[: a.extra_n] if a.extra_n else extra
    print("extra calls:", len(extra), flush=True)
ex = Extractor(catalog(), {"crossfit": 1}).fit(S['train'], value_match, log=lambda s: print(s, flush=True), extra=extra)
print('fit', round(time.time() - t, 1), flush=True)
pickle.dump(ex, open(a.out, 'wb'))
print('saved', a.out)
