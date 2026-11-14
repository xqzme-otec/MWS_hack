"""Массовая дистилляция: черновики генератора → учитель-LLM (OpenAI-совместимый сервер) → проверка → ASR-шум.

Пример (локальный vLLM):
  export TEACHER_BASE_URL=http://localhost:8000/v1 TEACHER_MODEL=Qwen/Qwen3-32B-AWQ TEACHER_API_KEY=local
  python src/synth/run_distill.py --drafts runs/drafts_clean_10k.jsonl --out runs/distilled_10k.jsonl --workers 16
Повторный запуск с тем же --out продолжает с места остановки (--resume по умолчанию).
"""
import argparse, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from teacher import distill_chain, plan_of, validate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--drafts", default="runs/drafts_clean_10k.jsonl")
ap.add_argument("--out", default="runs/distilled_10k.jsonl")
ap.add_argument("--rejects", default="")
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--workers", type=int, default=8, help="параллельных запросов к серверу")
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--max_tokens", type=int, default=3000)
ap.add_argument("--noise", type=float, default=0.25, help="ASR-шум поверх ответа учителя (0 = без шума)")
ap.add_argument("--style", default="runs/markers.json")
ap.add_argument("--no_json_mode", action="store_true", help="если сервер не поддерживает response_format")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

drafts = [json.loads(l) for l in open(a.drafts, encoding="utf-8")][: a.n]
done = set()
if os.path.exists(a.out):
    done = {json.loads(l)["call_id"] for l in open(a.out, encoding="utf-8")}
rej_path = a.rejects or a.out.replace(".jsonl", "_rejects.jsonl")
if os.path.exists(rej_path):
    done |= {json.loads(l)["call_id"] for l in open(rej_path, encoding="utf-8")}
todo = [c for c in drafts if c["call_id"] not in done]
style = json.load(open(a.style))["openstt_calls"]["p_first"] if os.path.exists(a.style) else {}
print(f"черновиков {len(drafts)}, уже обработано {len(done)}, осталось {len(todo)}", flush=True)

# Qwen3 и похожие: «/no_think» отключает рассуждение (в 5–10 раз быстрее). TEACHER_NO_THINK=0 — не добавлять.
NO_THINK = "\n/no_think" if os.environ.get("TEACHER_NO_THINK", "1") == "1" else ""
chain = distill_chain(temperature=a.temperature, json_mode=not a.no_json_mode, max_tokens=a.max_tokens)
noiser = None
if a.noise > 0:
    from generator import Generator
    noiser = Generator.__new__(Generator)
    noiser.rng = random.Random(a.seed)
lock = threading.Lock()
stats = {"ok": 0, "rejected": 0, "error": 0}
t0 = time.time()


def work(c):
    payload = {"style": json.dumps(style, ensure_ascii=False), "started_at": c["started_at"],
               "plan": json.dumps(plan_of(c), ensure_ascii=False),
               "draft": "\n".join(f"{t['idx']} {t['speaker']}: {t['text']}" for t in c["turns"]) + NO_THINK}
    try:
        out = chain.invoke(payload)
    except Exception as e:  # сеть, таймаут, невалидный JSON
        return c, None, f"error: {type(e).__name__}: {str(e)[:200]}"
    v = validate(c, out)
    if v is None:
        return c, None, "rejected: меньше 80% фактов прошли проверку"
    v["teacher"] = os.environ.get("TEACHER_MODEL", "teacher")
    if noiser is not None:
        with lock:
            for t in v["turns"]:
                t["text"] = noiser.noise(t["text"], a.noise)
        v["noise_level"] = a.noise
    return c, v, "ok"


with open(a.out, "a", encoding="utf-8") as fo, open(rej_path, "a", encoding="utf-8") as fr, \
        ThreadPoolExecutor(a.workers) as pool:
    futs = [pool.submit(work, c) for c in todo]
    for i, f in enumerate(as_completed(futs), 1):
        c, v, status = f.result()
        with lock:
            if v is not None:
                fo.write(json.dumps(v, ensure_ascii=False) + "\n"); fo.flush(); stats["ok"] += 1
            else:
                fr.write(json.dumps({"call_id": c["call_id"], "status": status}, ensure_ascii=False) + "\n"); fr.flush()
                stats["error" if status.startswith("error") else "rejected"] += 1
        if i % 25 == 0 or i == len(futs):
            el = time.time() - t0
            print(f"{i}/{len(futs)}  ok={stats['ok']} rejected={stats['rejected']} error={stats['error']}  "
                  f"{el / i:.1f} с/диалог, осталось ~{(len(futs) - i) * el / i / 3600:.1f} ч", flush=True)
print("готово:", stats, flush=True)
