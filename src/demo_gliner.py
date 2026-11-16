"""Консольное демо: звонок из val (или свой json) → факты таблицей + время.

  python src/demo_gliner.py               # случайный звонок из data/val.jsonl (или examples/calls.jsonl)
  python src/demo_gliner.py --i 5         # звонок №5
  python src/demo_gliner.py --file my_call.json --mode gliner
"""
import argparse, json, os, random, sys

here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
ROOT = os.path.join(here, "..")
ap = argparse.ArgumentParser()
ap.add_argument("--i", type=int, default=-1); ap.add_argument("--file", default="")
ap.add_argument("--model", default="models/g13_ru"); ap.add_argument("--mode", default="hybrid", choices=["hybrid", "gliner", "boosting"])
a = ap.parse_args()
from cfx.pipeline import Pipeline  # noqa: E402

P = Pipeline(a.model, mode=a.mode)
if a.file:
    calls = [json.load(open(a.file, encoding="utf-8"))]
else:
    src = os.path.join(ROOT, "data", "val.jsonl")
    src = src if os.path.exists(src) else os.path.join(ROOT, "examples", "calls.jsonl")
    calls = [json.loads(l) for l in open(src, encoding="utf-8")]
c = calls[a.i] if a.i >= 0 else random.choice(calls)
P.extract(c)                                  # прогрев
out = P.extract(c)
row = lambda f: (f"  [{f['source']['turn_idx']:>2}] {f['attribute']:14s} {('НЕ ' if f.get('polarity') == -1 else '') + f['value'][:40]:40s} | "
                 f"{f['object']['name'][:25]:25s} | c {f['valid_from'][:10]}{' по ' + f['valid_to'][:10] if f.get('valid_to') else ''}")
print(f"\n=== звонок {c['call_id']}  ({c['started_at'][:10]}, {len(c['turns'])} реплик) ===")
for t in c["turns"]:
    print(f"{t['idx']:>3} {'ОП' if t['speaker'] == 'operator' else 'КЛ'}: {t['text']}")
m = out["meta"]
print(f"\n=== {a.mode}: фактов {len(out['facts'])}   время {m['latency_ms']:.0f} мс (GLiNER на {m['turns_gliner']}/{m['turns_total']} репликах) ===")
for f in out["facts"]:
    print(row(f))
if c.get("facts"):
    print(f"\n=== эталон: {len(c['facts'])} фактов ===")
    for f in c["facts"]:
        print(row(f))
