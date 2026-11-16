"""Дообучение GLiNER2.5 (ru-pii-ner) на фактах звонков: урезанный словарь + K слоёв + LoRA или полное.

  python src/g2/train_g2.py --out out/g2 --keep_layers 6 --lora 1 --lora_r 16 --steps 3000 --save 1
Оценка: span-F1 (точное совпадение границ) на golden чистом и с ASR-шумом 0.3; задержка на CPU (2 потока).
"""
import os, sys, json, time, argparse, random
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="metrika-labs/ru-pii-ner-gliner2.5")
ap.add_argument("--out", default="out/g2")
ap.add_argument("--keep_layers", type=int, default=0, help="слоёв энкодера (0 = все 12)")
ap.add_argument("--prune_vocab", type=int, default=1, help="урезать словарь до русских токенов корпуса")
ap.add_argument("--lora", type=int, default=0); ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--lr", type=float, default=2e-5, help="lr энкодера"); ap.add_argument("--task_lr", type=float, default=3e-4)
ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--neg_ratio", type=float, default=0.2); ap.add_argument("--noise_copies", type=int, default=1)
ap.add_argument("--noise_level", type=float, default=0.3); ap.add_argument("--ctx", type=int, default=0)
ap.add_argument("--neg_labels", type=int, default=2, help="пустых типов фактов на пример (контраст для счётчика)")
ap.add_argument("--n_calls", type=int, default=0); ap.add_argument("--n_dev", type=int, default=150)
ap.add_argument("--extra", default="", help="jsonl дистиллированных звонков")
ap.add_argument("--lat_calls", type=int, default=4); ap.add_argument("--save", type=int, default=1)
ap.add_argument("--result", default=""); ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

import common as C  # noqa: E402
from cfx.data import splits  # noqa: E402
from cfx.text import split_of  # noqa: E402
from gliner2.training.trainer import ExtractorTrainer, TrainingConfig  # noqa: E402

dev = C.device()
print("device", dev, flush=True)
S = splits()
fit = [c for c in S["train"] if split_of(c) == "fit"]
fit = fit[: a.n_calls] if a.n_calls else fit
if a.extra and os.path.exists(a.extra):
    fit += [json.loads(l) for l in open(a.extra, encoding="utf-8")]
gold = S["golden"][: a.n_dev]
tr, st = C.examples(fit, ctx=a.ctx, neg_ratio=a.neg_ratio, noise_copies=a.noise_copies, noise_level=a.noise_level, seed=a.seed)
dv, _ = C.examples(gold, ctx=a.ctx, drop_unplaced=False)
dn, _ = C.examples(gold, ctx=a.ctx, drop_unplaced=False, noise_copies=1, noise_level=0.3, seed=a.seed + 1)
dn = [x for x in dn if x[2]]
print(f"train {len(tr)} примеров (разметка {st}); dev {len(dv)} / шум {len(dn)}", flush=True)

model = C.load_base(a.base)
vocab = None
if a.prune_vocab:
    corpus = [t for c in S["train"] for t, _, _ in C.call_texts(c, a.ctx)] + [t for t, _, _ in tr] + C.FACT_LABELS
    vocab = C.vocab_ids(model, corpus)
n0 = sum(p.numel() for p in model.parameters())
layers = C.apply_arch(model, a.keep_layers, vocab)
n1 = sum(p.numel() for p in model.parameters())
print(f"архитектура: слоёв {layers}, словарь {len(vocab) if vocab else 'полный'}, параметров {n0/1e6:.0f}M → {n1/1e6:.1f}M, "
      f"{'LoRA r=' + str(a.lora_r) if a.lora else 'полное дообучение'}", flush=True)
if dev.type == "mps":   # MPS: SDPA не поддерживает dropout при обучении → обычное внимание
    for m_ in C.encoder_of(model).modules():
        if hasattr(m_, "config") and hasattr(m_.config, "_attn_implementation"):
            m_.config._attn_implementation = "eager"


class Trainer(ExtractorTrainer):
    def _setup_device(self):          # + Apple MPS (в библиотеке только CUDA/CPU)
        if dev.type == "mps":
            self.device, self.is_distributed = dev, False
            self.config.fp16 = self.config.bf16 = False
            self.model.to(dev); self.model.float()
        else:
            super()._setup_device()

    def _save_checkpoint(self, name):  # сохраняем сами (урезанный словарь)
        return


cfg = TrainingConfig(output_dir=os.path.join(a.out, "_trainer"), max_steps=a.steps, num_epochs=100, batch_size=a.bs,
                     encoder_lr=a.lr, task_lr=a.task_lr, warmup_ratio=0.06, scheduler_type="linear",
                     eval_strategy="no", save_best=False, num_workers=0, pin_memory=False, logging_steps=50,
                     seed=a.seed, use_lora=bool(a.lora), lora_r=a.lora_r, lora_alpha=2 * a.lora_r,
                     save_adapter_only=False, fp16=False, bf16=False)
t0 = time.time()
trainer = Trainer(model=model, config=cfg)
trainer.train(train_data=C.input_examples(tr, a.neg_labels, a.seed))
model = trainer.model
if hasattr(model, "merge_and_unload"):
    model = model.merge_and_unload()
minutes = (time.time() - t0) / 60
model.to(dev).eval()
mc, mn = C.span_f1(model, dv, 0.5), C.span_f1(model, dn, 0.5)
print("EVAL", json.dumps({"dev": mc, "dev_noise0.3": mn, "minutes": round(minutes, 1)}, ensure_ascii=False), flush=True)

lat = None
if a.lat_calls:
    model.to("cpu"); torch.set_num_threads(2)
    calls = S["golden"][: a.lat_calls]
    C.call_spans(model, calls[0], a.ctx); t1 = time.time()
    for c in calls:
        C.call_spans(model, c, a.ctx)
    lat = round(1000 * (time.time() - t1) / len(calls), 1)
    print(f"задержка CPU (2 потока): {lat} мс/звонок", flush=True)
arch = {"base": a.base, "keep_layers": layers, "ctx": a.ctx, "lora": a.lora, "lora_r": a.lora_r, "neg_labels": a.neg_labels}
if a.save:
    C.save(model, a.out, arch)
    if vocab:
        json.dump(vocab, open(os.path.join(a.out, "vocab.json"), "w"))
    json.dump({"dev": mc, "dev_noise0.3": mn, "minutes": round(minutes, 1), "lat_ms": lat, "args": vars(a)},
              open(os.path.join(a.out, "train_log.json"), "w"), ensure_ascii=False, indent=1)
if a.result:
    json.dump({"score": 0.5 * (mc["f1"] + mn["f1"]), "dev": mc, "dev_noise0.3": mn, "lat_ms": lat, "layers": layers,
               "params": n1, "minutes": round(minutes, 1), "args": vars(a)}, open(a.result, "w"), ensure_ascii=False, indent=1)
print("готово", flush=True)
