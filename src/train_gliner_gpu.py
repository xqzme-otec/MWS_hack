"""Полное дообучение GLiNER на GPU (Kaggle/Colab): разметка v2 + ASR-шум, несколько эпох, отбор по dev.

  python src/train_gliner_gpu.py --base knowledgator/gliner-x-small --out out/xs --epochs 3
  python src/train_gliner_gpu.py --base urchade/gliner_multi-v2.1 --out out/multi --epochs 2 --bs 8

Разметка: cfx.gliner_data.build_examples_v2 (fit-часть train, разбиение по аккаунтам как у всей системы).
Отбор: span-F1 по 5 меткам фактов на golden (чистый) и golden+шум 0.3 — сохраняется лучшая точка.
"""
import os, sys, json, time, random, argparse, math
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # Apple Silicon: неподдержанные операции → CPU
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="knowledgator/gliner-x-small")
ap.add_argument("--out", default="out/xs")
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--max_minutes", type=float, default=0, help="жёсткий лимит времени обучения (0 = нет)")
ap.add_argument("--total_steps", type=int, default=0, help="фиксированная длина расписания lr в шагах (воспроизводимость: не зависит от скорости машины)")
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-5, help="lr энкодера")
ap.add_argument("--head_lr", type=float, default=1e-4, help="lr голов GLiNER")
ap.add_argument("--warmup", type=float, default=0.06)
ap.add_argument("--neg_ratio", type=float, default=0.2)
ap.add_argument("--noise_copies", type=int, default=1, help="копий примера с ASR-шумом")
ap.add_argument("--noise_level", type=float, default=0.3)
ap.add_argument("--evals_per_epoch", type=int, default=2)
ap.add_argument("--fp16", action="store_true", help="AMP fp16 (не для mT5: переполнение) ")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--n_calls", type=int, default=0, help="для отладки: первые N fit-звонков")
ap.add_argument("--n_dev", type=int, default=150)
# ---- гены архитектуры (эволюция архитектуры: src/evolve_gliner_arch.py)
ap.add_argument("--keep_layers", type=int, default=0, help="оставить первые K слоёв энкодера (0 = все)")
ap.add_argument("--rnn", type=int, default=1, help="0 = убрать LSTM-слой над энкодером")
ap.add_argument("--max_width", type=int, default=0, help="макс. ширина спана в словах (0 = как в модели)")
ap.add_argument("--dropout", type=float, default=-1, help="dropout голов GLiNER (-1 = как в модели)")
ap.add_argument("--freeze_layers", type=int, default=0, help="заморозить эмбеддинги и первые F слоёв энкодера")
ap.add_argument("--ctx", type=int, default=25, help="слов следующей реплики в контексте")
ap.add_argument("--save", type=int, default=1, help="0 = не сохранять веса (кандидаты эволюции)")
ap.add_argument("--final_eval_only", action="store_true")
ap.add_argument("--lat_calls", type=int, default=0, help="замер задержки на CPU (2 потока) на N звонках golden")
ap.add_argument("--result", default="", help="json с итогом кандидата")
ap.add_argument("--extra", default="", help="jsonl дистиллированных звонков (добавляются к fit)")
ap.add_argument("--extra_n", type=int, default=0)
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
from cfx.gliner_hub import device, resolve_base
dev = device()
print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)

from gliner import GLiNER
from gliner.data_processing.tokenizer import WordsSplitter
from cfx.data import splits
from cfx.text import split_of
from cfx.gliner_data import build_examples_v2, LABELS, tokenize

S = splits()
fit = [c for c in S["train"] if split_of(c) == "fit"]
fit = fit[: a.n_calls] if a.n_calls else fit
gold = S["golden"][: a.n_dev]
if a.extra and os.path.exists(a.extra):
    extra = [json.loads(l) for l in open(a.extra, encoding="utf-8")]
    extra = extra[: a.extra_n] if a.extra_n else extra
    print(f"+ дистиллированных звонков: {len(extra)}", flush=True)
    fit = fit + extra
MW = a.max_width or 12
train_ex, st = build_examples_v2(fit, neg_ratio=a.neg_ratio, seed=a.seed, noise_copies=a.noise_copies, noise_level=a.noise_level,
                                 context_words=a.ctx, max_width=MW)
dev_clean, _ = build_examples_v2(gold, neg_ratio=0.2, seed=a.seed, drop_unplaced=False, context_words=a.ctx, max_width=MW)
dev_noisy, _ = build_examples_v2(gold, neg_ratio=0.2, seed=a.seed + 1, drop_unplaced=False, noise_copies=1, noise_level=0.3,
                                 context_words=a.ctx, max_width=MW)
dev_noisy = [e for e in dev_noisy if "noise" in e["meta"]]
print(f"train {len(train_ex)} примеров (разметка: {st}); dev {len(dev_clean)} / шум {len(dev_noisy)}", flush=True)
labels_all = sorted(set(LABELS.values()))
FACT_LABELS = {LABELS[k] for k in ("service_state", "tariff", "ticket_status", "action", "agreement")}
for e in train_ex + dev_clean + dev_noisy:
    e["label"] = labels_all

model = GLiNER.from_pretrained(resolve_base(a.base)).to(dev)
model.data_processor.words_splitter = WordsSplitter("whitespace")
model.config.words_splitter_type = "whitespace"


def apply_arch(model):
    """Гены архитектуры: усечение энкодера, LSTM, ширина спанов, dropout, заморозка."""
    enc = model.model.token_rep_layer.bert_layer.model
    ec = model.config.encoder_config
    setc = (lambda k, v: ec.__setitem__(k, v)) if isinstance(ec, dict) else (lambda k, v: setattr(ec, k, v))
    if hasattr(enc, "encoder") and hasattr(enc.encoder, "block"):          # mT5 / T5
        blocks = enc.encoder.block
        if a.keep_layers and a.keep_layers < len(blocks):
            enc.encoder.block = torch.nn.ModuleList(list(blocks)[: a.keep_layers])
            enc.config.num_layers = a.keep_layers; setc("num_layers", a.keep_layers)
        layers = list(enc.encoder.block)
    else:                                                                  # BERT / DeBERTa
        L = enc.encoder.layer
        if a.keep_layers and a.keep_layers < len(L):
            enc.encoder.layer = torch.nn.ModuleList(list(L)[: a.keep_layers])
            enc.config.num_hidden_layers = a.keep_layers; setc("num_hidden_layers", a.keep_layers)
        layers = list(enc.encoder.layer)
    if not a.rnn and hasattr(model.model, "rnn"):
        del model.model.rnn
        model.config.num_rnn_layers = 0
    if a.max_width:
        model.config.max_width = a.max_width
        if hasattr(model.data_processor, "config"):
            model.data_processor.config.max_width = a.max_width
        for m in model.model.modules():           # слой представления спанов хранит ширину у себя
            if hasattr(m, "max_width") and isinstance(getattr(m, "max_width"), int):
                m.max_width = a.max_width
    if a.dropout >= 0:
        model.config.dropout = a.dropout
        for n, m in model.model.named_modules():
            if isinstance(m, torch.nn.Dropout) and not n.startswith("token_rep_layer"):
                m.p = a.dropout
    if a.freeze_layers:
        for n, p in enc.named_parameters():
            if "embed" in n or "shared" in n:
                p.requires_grad = False
        for blk in layers[: a.freeze_layers]:
            for p in blk.parameters():
                p.requires_grad = False
    return len(layers)


n_layers = apply_arch(model)
if dev.type == "mps":
    # MPS: scaled_dot_product_attention не поддерживает dropout при обучении → обычная (eager) реализация внимания
    enc_ = model.model.token_rep_layer.bert_layer.model
    try:
        enc_.set_attn_implementation("eager")
    except Exception:
        pass
    for m_ in enc_.modules():
        if hasattr(m_, "config") and hasattr(m_.config, "_attn_implementation"):
            m_.config._attn_implementation = "eager"
    print("MPS: внимание переключено на eager", flush=True)
print(f"архитектура: слоёв энкодера {n_layers}, LSTM {'да' if hasattr(model.model, 'rnn') else 'нет'}, "
      f"max_width {model.config.max_width}, ctx {a.ctx}", flush=True)
n_all = sum(p.numel() for p in model.parameters())
collator = model._create_data_collator()
ex_train = [{k: v for k, v in e.items() if k != "meta"} for e in train_ex]
dl = DataLoader(ex_train, batch_size=a.bs, shuffle=True, collate_fn=collator,
                generator=torch.Generator().manual_seed(a.seed), num_workers=2)
enc_p = [p for n, p in model.named_parameters() if "token_rep_layer" in n and p.requires_grad]
head_p = [p for n, p in model.named_parameters() if "token_rep_layer" not in n and p.requires_grad]
opt = torch.optim.AdamW([{"params": enc_p, "lr": a.lr}, {"params": head_p, "lr": a.head_lr}], weight_decay=0.01)
total = int(math.ceil(len(dl) * a.epochs))
warm = max(1, int(a.warmup * total))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)) if s >= warm else (s + 1) / warm)
scaler = torch.amp.GradScaler("cuda", enabled=a.fp16 and dev.type == "cuda")
eval_every = max(50, len(dl) // max(1, a.evals_per_epoch))
if a.total_steps:   # расписание как у эталонного прогона: warmup от полной длины, обрыв на total_steps
    total = a.total_steps; eval_every = max(50, total // max(1, int(a.evals_per_epoch * max(a.epochs, 1))))
    print(f"фиксированное расписание: {total} шагов, warmup {warm}, оценка каждые {eval_every}", flush=True)


@torch.no_grad()
def span_f1(examples, thr=0.5):
    """F1 по точному совпадению (начало, конец, метка) для 5 меток фактов; плюс F1 по пересечению."""
    model.eval()
    tp = fp = fn = 0; otp = 0
    texts = [" ".join(e["tokenized_text"]) for e in examples]
    for k in range(0, len(texts), 16):
        preds = model.batch_predict_entities(texts[k:k + 16], labels_all, threshold=thr, flat_ner=False)
        for e, ents in zip(examples[k:k + 16], preds):
            starts = np.cumsum([0] + [len(t) + 1 for t in e["tokenized_text"]])
            gold = {(s, t, l) for s, t, l in e["ner"] if l in FACT_LABELS}
            pr = set()
            for x in ents:
                if x["label"] not in FACT_LABELS:
                    continue
                s = int(np.searchsorted(starts, x["start"], side="right") - 1)
                t = int(np.searchsorted(starts, x["end"] - 1, side="right") - 1)
                pr.add((s, t, x["label"]))
            tp += len(gold & pr); fp += len(pr - gold); fn += len(gold - pr)
            otp += sum(any(l == gl and s <= gt and gs <= t for gs, gt, gl in gold) for s, t, l in pr)
    model.train()
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    po = otp / max(tp + fp, 1)
    return {"f1": round(2 * p * r / max(p + r, 1e-9), 4), "p": round(p, 4), "r": round(r, 4), "overlap_p": round(po, 4)}


os.makedirs(a.out, exist_ok=True)
log, best, t0, step, losses = [], -1.0, time.time(), 0, []
best_rec = None
model.train()
done = False
while not done:
    for batch in dl:
        batch = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=a.fp16 and dev.type == "cuda"):
            out = model(**{k: v for k, v in batch.items() if isinstance(v, torch.Tensor)})
            loss = out.loss if hasattr(out, "loss") else out["loss"]
        if not torch.isfinite(loss):
            opt.zero_grad(); step += 1; continue
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
        losses.append(float(loss)); step += 1
        if step == 30 and a.max_minutes and not a.total_steps:
            sps = (time.time() - t0) / 30
            fit_steps = int(a.max_minutes * 60 * 0.92 / sps)
            if fit_steps < total:
                total = max(fit_steps, 60); eval_every = max(50, total // max(1, a.evals_per_epoch * max(a.epochs, 1)))
                print(f"бюджет {a.max_minutes} мин → {total} шагов ({sps:.2f} с/шаг)", flush=True)
        if step % 50 == 0:
            el = time.time() - t0
            print(f"step {step}/{total} loss {np.mean(losses[-50:]):.3f}  {el / step:.2f} с/шаг, осталось ~{(total - step) * el / step / 60:.0f} мин", flush=True)
        timeout = a.max_minutes and (time.time() - t0) / 60 > a.max_minutes
        if (not a.final_eval_only and step % eval_every == 0) or step >= total or timeout:
            mc, mn = span_f1(dev_clean), span_f1(dev_noisy)
            score = 0.5 * (mc["f1"] + mn["f1"])
            rec = {"step": step, "epoch": round(step / len(dl), 2), "loss": round(float(np.mean(losses[-200:])), 4),
                   "dev": mc, "dev_noise0.3": mn, "minutes": round((time.time() - t0) / 60, 1)}
            log.append(rec); print("EVAL", json.dumps(rec, ensure_ascii=False), flush=True)
            if score > best:
                best = score; best_rec = rec
                if a.save:
                    model.save_pretrained(a.out)
                    try:
                        model.data_processor.transformer_tokenizer.save_pretrained(a.out)   # токенизатор рядом с весами
                    except Exception as e:
                        print("tokenizer save:", e)
                    json.dump({"ctx": a.ctx, "keep_layers": n_layers, "rnn": a.rnn, "max_width": MW},
                              open(os.path.join(a.out, "arch.json"), "w"))
                print(f"  ↳ сохранена лучшая точка (score {score:.4f})", flush=True)
        if step >= total or timeout:
            done = True
            break
lat = None
if a.lat_calls:
    from cfx.gliner_fusion import GlinerSpans
    model.eval(); model.to("cpu"); torch.set_num_threads(2)
    gl = GlinerSpans(model, context_words=a.ctx, raw_threshold=0.1, batch=16)
    calls = S["golden"][: a.lat_calls]
    with torch.no_grad():
        gl.spans(calls[0]); t1 = time.time()
        for c in calls: gl.spans(c)
    lat = round(1000 * (time.time() - t1) / len(calls), 1)
    print(f"задержка CPU (2 потока, fp32): {lat} мс/звонок", flush=True)
if a.result:
    json.dump({"score": best, "best": best_rec, "lat_ms": lat, "layers": n_layers, "params": n_all,
               "minutes": round((time.time() - t0) / 60, 1), "args": vars(a)}, open(a.result, "w"), ensure_ascii=False, indent=1)
json.dump({"args": vars(a), "params": n_all, "train_examples": len(train_ex), "labeling": st, "log": log,
           "best_score": best, "minutes": round((time.time() - t0) / 60, 1)},
          open(os.path.join(a.out, "train_log.json") if a.save else os.devnull, "w"), ensure_ascii=False, indent=1)
print("готово", a.out, "best", best, flush=True)
