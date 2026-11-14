"""LoRA-дообучение GLiNER на спанах фактов (CPU).

LoRA (peft) внедряется в линейные слои трансформер-энкодера; головы GLiNER (span/prompt/LSTM)
обучаются полностью — это небольшая доля параметров. Сиды фиксированы.
Запуск: python src/train_gliner.py --base ../hf/gliner-x-small --out runs/gliner_lora --steps 600
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys, json, time, random, argparse, math
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from torch.utils.data import DataLoader

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="../hf/gliner-x-small")
ap.add_argument("--out", default="runs/gliner_lora")
ap.add_argument("--steps", type=int, default=600)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--head_lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--targets", default="q,k,v,o,wi_0,wi_1,wo")
ap.add_argument("--n_calls", type=int, default=1200)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--data", default="")
a = ap.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
torch.set_num_threads(a.threads)

from gliner import GLiNER
from peft import LoraConfig, inject_adapter_in_model
from cfx.data import splits
from cfx.text import split_of
from cfx.gliner_data import build_examples, LABELS

S = splits()
if a.data:
    train_ex = [json.loads(l) for l in open(a.data)]
else:
    fit = [c for c in S["train"] if split_of(c) == "fit"]
    random.Random(a.seed).shuffle(fit)
    train_ex = build_examples(fit[: a.n_calls], neg_ratio=0.25, seed=a.seed)
dev_ex = build_examples(S["golden"][:40], neg_ratio=0.25, seed=a.seed)
print(f"train examples {len(train_ex)}, dev {len(dev_ex)}; spans/ex {np.mean([len(e['ner']) for e in train_ex]):.2f}", flush=True)
labels_all = sorted(set(LABELS.values()))
for e in train_ex + dev_ex:
    e.pop("meta", None)

model = GLiNER.from_pretrained(a.base)
enc = model.model.token_rep_layer.bert_layer.model
cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, target_modules=a.targets.split(","))
inject_adapter_in_model(cfg, enc)
for n, p in model.named_parameters():
    if "token_rep_layer.bert_layer" in n:
        p.requires_grad = "lora_" in n
    else:
        p.requires_grad = True
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in model.parameters())
print(f"trainable {n_tr/1e6:.2f}M / {n_all/1e6:.1f}M ({100*n_tr/n_all:.1f}%)", flush=True)

collator = model._create_data_collator()
# все метки в каждом примере: модель учится и на отрицательных типах
for e in train_ex + dev_ex:
    e["label"] = labels_all
dl = DataLoader(train_ex, batch_size=a.bs, shuffle=True, collate_fn=collator, generator=torch.Generator().manual_seed(a.seed))
lora_p = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
head_p = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
opt = torch.optim.AdamW([{"params": lora_p, "lr": a.lr}, {"params": head_p, "lr": a.head_lr}], weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 30) * max(0.05, 1 - s / a.steps))
model.train()
step, t0, losses = 0, time.time(), []
log = []
while step < a.steps:
    for batch in dl:
        batch = {k: v for k, v in batch.items() if k in ("input_ids", "attention_mask", "words_mask", "text_lengths", "span_idx", "span_mask", "labels")} if False else batch
        out = model(**{k: v for k, v in batch.items() if isinstance(v, torch.Tensor)})
        loss = out.loss if hasattr(out, "loss") else out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        losses.append(float(loss)); step += 1
        if step % 20 == 0:
            el = time.time() - t0
            rec = {"step": step, "loss": round(float(np.mean(losses[-20:])), 4), "sec": round(el, 1), "sec_per_step": round(el / step, 2)}
            log.append(rec); print(rec, flush=True)
        if step >= a.steps:
            break
os.makedirs(a.out, exist_ok=True)
model.eval()
model.save_pretrained(a.out)
json.dump({"args": vars(a), "trainable_params": n_tr, "total_params": n_all, "log": log,
           "train_examples": len(train_ex), "seconds": round(time.time() - t0, 1)}, open(os.path.join(a.out, "train_log.json"), "w"), indent=1)
print("saved", a.out, flush=True)
