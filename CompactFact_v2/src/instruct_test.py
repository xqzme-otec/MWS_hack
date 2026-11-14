"""Проверка instruct-модели (Qwen3-0.6B) как экстрактора: zero/few-shot JSON-вывод на CPU.

Цель — точка на кривой «ресурсы × качество» и замер латентности генеративного подхода на том
же железе. Выход приводится к схеме кейса; даты — тем же детерминированным селектором (из
извлечённого выражения), чтобы сравнение касалось именно извлечения.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys, json, time, re, argparse
sys.path.insert(0, os.path.dirname(__file__))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cfx.data import splits
from metrics_util import strict

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="../hf/qwen3-0.6b"); ap.add_argument("--n", type=int, default=6)
ap.add_argument("--threads", type=int, default=2); ap.add_argument("--max_new", type=int, default=400); ap.add_argument("--out", default="runs/instruct_qwen06.json")
a = ap.parse_args()
torch.set_num_threads(a.threads)
tok = AutoTokenizer.from_pretrained(a.model); model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32); model.eval()
SYS = ("Извлеки факты из звонка в техподдержку. Верни ТОЛЬКО JSON-массив объектов с полями: "
       "attribute (service_state|tariff|ticket_status|action|agreement), value (коротко), object (услуга/устройство/заявка или 'клиент'), "
       "turn (номер реплики), date_phrase (дословное выражение времени или ''). Не включай предложения оператора, от которых клиент отказался.")
EX_IN = "0 operator: Здравствуйте!\n1 client: У меня интернет со вторника пропадает, роутер перезагружала — не помогло.\n2 operator: Инженер приедет в четверг после двух, подходит?\n3 client: Да, подходит."
EX_OUT = ('[{"attribute":"service_state","value":"периодические обрывы","object":"домашний интернет","turn":1,"date_phrase":"со вторника"},'
          '{"attribute":"action","value":"перезагрузка роутера — не помогло","object":"клиент","turn":1,"date_phrase":""},'
          '{"attribute":"agreement","value":"выезд специалиста","object":"клиент","turn":3,"date_phrase":"в четверг после двух"}]')
S = splits(); calls = S["val"][: a.n]
from cfx.dates_bank import extract
from datetime import datetime, timedelta
preds, lat, toks_in, toks_out = {}, [], [], []
for c in calls:
    dialog = "\n".join(f"{t['idx']} {t['speaker']}: {t['text']}" for t in c["turns"])
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": EX_IN}, {"role": "assistant", "content": EX_OUT},
            {"role": "user", "content": dialog}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt")
    t = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=a.max_new, do_sample=False)
    dt = time.perf_counter() - t
    gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    lat.append(dt); toks_in.append(int(ids["input_ids"].shape[1])); toks_out.append(int(out.shape[1] - ids["input_ids"].shape[1]))
    facts = []
    m = re.search(r"\[.*\]", gen, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except Exception:
        items = []
    cd = datetime.fromisoformat(c["started_at"]).date()
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or it.get("attribute") not in ("service_state", "tariff", "ticket_status", "action", "agreement"):
            continue
        dp = str(it.get("date_phrase") or "")
        ex = extract(dp, cd)
        vf = c["started_at"]
        if ex:
            ds = ex[0][3]
            d = ds[-1] if it["attribute"] == "agreement" else ds[0]
            vf = datetime.combine(d, datetime.min.time()).isoformat()
        ti = it.get("turn") if isinstance(it.get("turn"), int) and 0 <= it.get("turn") < len(c["turns"]) else 0
        facts.append({"object": {"name": str(it.get("object") or "")}, "attribute": it["attribute"], "value": str(it.get("value") or ""),
                      "polarity": 1, "speaker": c["turns"][ti]["speaker"], "valid_from": vf, "valid_to": None, "source": {"turn_idx": ti}})
    preds[c["call_id"]] = {**c, "facts": facts}
    print(f"call {c['call_id']}: {dt:.1f}s, in {toks_in[-1]} tok, out {toks_out[-1]} tok, facts {len(facts)}", flush=True)
ss = strict()
m = ss.evaluate({c["call_id"]: c for c in calls}, preds)
res = {"model": "Qwen3-0.6B (instruct, few-shot)", "params": sum(p.numel() for p in model.parameters()), "threads": a.threads,
       "n_calls": len(calls), "latency_s_mean": round(sum(lat) / len(lat), 2), "tokens_in_mean": sum(toks_in) / len(toks_in),
       "tokens_out_mean": sum(toks_out) / len(toks_out), "metrics_small_sample": {k: m[k] for k in ("fact_precision", "fact_recall", "temporal_recall", "state_accuracy")}}
print(json.dumps(res, ensure_ascii=False, indent=1)); json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
