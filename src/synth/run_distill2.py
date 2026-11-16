"""Дистилляция v2: планы генератора v2 → DeepSeek (ASR-транскрипт с цитатами) → проверка → runs/distilled_v2.jsonl.

  export TEACHER_API_KEY=… (или DEEPSEEK_API_KEY в ~/Desktop/MTS/.env)
  python src/synth/run_distill2.py --n 200 --out runs/distill_v2_pilot.jsonl            # пилот
  python src/synth/run_distill2.py --n 8000 --budget 4.5 --out runs/distilled_v2.jsonl  # основной
Продолжает с места остановки; останавливается сам, когда потрачен --budget долларов.
"""
import argparse, json, os, pickle, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
from teacher import llm  # noqa: E402
from teacher2 import Validator, messages, parse  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--plans", default="runs/plans_v2.jsonl")
ap.add_argument("--out", default="runs/distilled_v2.jsonl")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--budget", type=float, default=4.5, help="стоп при достижении, $")
ap.add_argument("--model_pkl", default="models/compactfact_v5.pkl", help="банк значений для проверки")
ap.add_argument("--temperature", type=float, default=0.9)
ap.add_argument("--max_tokens", type=int, default=4000)
a = ap.parse_args()
os.environ.setdefault("TEACHER_BASE_URL", "https://api.deepseek.com")
os.environ.setdefault("TEACHER_MODEL", "deepseek-flash")

# цены deepseek-flash, $ за 1M токенов (off-peak; в пиковые часы будних дней ×2)
RATE = {"hit": 0.003, "miss": 0.15, "out": 0.60}


def peak_mult():
    t = datetime.now(timezone.utc)
    return 2.0 if t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10) else 1.0


plans = [json.loads(l) for l in open(a.plans, encoding="utf-8")][: a.n]
rej_path = a.out.replace(".jsonl", "_rejects.jsonl")
done = set()
for p in (a.out, rej_path):
    if os.path.exists(p):
        done |= {json.loads(l)["call_id"] for l in open(p, encoding="utf-8")}
cost_path = a.out.replace(".jsonl", "_cost.json")
spent = json.load(open(cost_path))["usd"] if os.path.exists(cost_path) else 0.0
todo = [c for c in plans if c["call_id"] not in done]
print(f"планов {len(plans)}, уже обработано {len(done)}, осталось {len(todo)}; потрачено ${spent:.3f} из ${a.budget}", flush=True)

ex = pickle.load(open(a.model_pkl, "rb"))
check = Validator(ex)
model = llm(temperature=a.temperature, max_tokens=a.max_tokens)
lock = threading.Lock()
st = {"ok": 0, "rejected": 0, "error": 0, "tok_in": 0, "tok_hit": 0, "tok_out": 0}
stop = threading.Event()
t0 = time.time()


def usage_cost(res):
    tu = (res.response_metadata or {}).get("token_usage") or {}
    um = res.usage_metadata or {}
    pin = tu.get("prompt_tokens") or um.get("input_tokens", 0)
    hit = tu.get("prompt_cache_hit_tokens") or (um.get("input_token_details") or {}).get("cache_read", 0) or 0
    out = tu.get("completion_tokens") or um.get("output_tokens", 0)
    usd = peak_mult() * (hit * RATE["hit"] + (pin - hit) * RATE["miss"] + out * RATE["out"]) / 1e6
    return pin, hit, out, usd


def work(c):
    if stop.is_set():
        return c, None, "skipped", None
    try:
        res = model.invoke(messages(c))
    except Exception as e:
        return c, None, f"error: {type(e).__name__}: {str(e)[:200]}", None
    u = usage_cost(res)
    try:
        v, why = check(c, parse(res.content))
    except Exception as e:
        return c, None, f"rejected: невалидный JSON ({type(e).__name__})", u
    return c, v, ("ok" if v else f"rejected: {why}"), u


with open(a.out, "a", encoding="utf-8") as fo, open(rej_path, "a", encoding="utf-8") as fr, ThreadPoolExecutor(a.workers) as pool:
    futs = [pool.submit(work, c) for c in todo]
    for i, f in enumerate(as_completed(futs), 1):
        c, v, status, u = f.result()
        if status == "skipped":
            continue
        with lock:
            if u:
                st["tok_in"] += u[0]; st["tok_hit"] += u[1]; st["tok_out"] += u[2]; spent += u[3]
            if v is not None:
                fo.write(json.dumps(v, ensure_ascii=False) + "\n"); fo.flush(); st["ok"] += 1
            else:
                fr.write(json.dumps({"call_id": c["call_id"], "block": c["block"], "status": status}, ensure_ascii=False) + "\n"); fr.flush()
                st["error" if status.startswith("error") else "rejected"] += 1
            json.dump({"usd": round(spent, 4), **st}, open(cost_path, "w"))
            if spent >= a.budget and not stop.is_set():
                stop.set(); print(f"!!! бюджет ${a.budget} исчерпан — останавливаюсь", flush=True)
        n = st["ok"] + st["rejected"] + st["error"]
        if n % 25 == 0 or n == len(todo):
            el = time.time() - t0
            per = spent / max(n, 1) if n else 0
            print(f"{n}/{len(todo)} ok={st['ok']} rejected={st['rejected']} error={st['error']} | ${spent:.3f} "
                  f"(${1000 * spent / max(n + len(done), 1):.3f} за 1000) | {el / max(n, 1):.1f} с/звонок | "
                  f"вых. токенов/звонок {st['tok_out'] / max(n, 1):.0f}, кэш {st['tok_hit'] / max(st['tok_in'], 1):.0%}", flush=True)
print("готово:", {k: v for k, v in st.items()}, f"потрачено ${spent:.3f}", flush=True)
