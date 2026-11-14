"""Слияние спанов GLiNER с кандидатами экстрактора (позднее слияние).

GLiNER (дообученный LoRA) размечает в каждой реплике спаны типов фактов. Спаны:
  (1) усиливают кандидатов «реплика × значение», если в той же реплике есть спан того же
      атрибута, покрывающий слова значения: p' = p + w_boost · score;
  (2) порождают новых кандидатов: текст спана → ближайшее значение банка (символьный TF-IDF),
      p = w_new · score.
Пороги по меткам, веса и размер контекста — гены эволюции («параметры движка GLiNER»).
Сырые спаны считаются один раз с низким порогом и кэшируются: эволюция гоняет гены по кэшу.
"""
from __future__ import annotations

import time

import numpy as np
from sklearn.preprocessing import normalize

from .gliner_data import LABELS, SPK
from .text import norm, stem, words

ATTR_OF = {v: k for k, v in LABELS.items()}
GENES = {"gl_on": 1, "gl_thr_service_state": 0.4, "gl_thr_tariff": 0.4, "gl_thr_ticket_status": 0.4,
         "gl_thr_action": 0.4, "gl_thr_agreement": 0.4, "gl_w_boost": 0.15, "gl_w_new": 0.5, "gl_min_cov": 0.5}


class GlinerSpans:
    def __init__(self, model, context_words=25, raw_threshold=0.15, batch=12):
        self.model, self.ctx, self.raw_thr, self.batch = model, context_words, raw_threshold, batch
        self.labels = list(LABELS.values())

    def texts(self, call):
        out = []
        turns = call["turns"]
        for i, t in enumerate(turns):
            prefix = SPK.get(t["speaker"], "") + " "
            nxt = turns[i + 1] if i + 1 < len(turns) else None
            suffix = (" || " + SPK.get(nxt["speaker"], "") + " " + " ".join(nxt["text"].split()[: self.ctx])) if nxt else ""
            out.append((prefix + t["text"] + suffix, len(prefix), len(prefix) + len(t["text"])))
        return out

    def spans(self, call):
        """[(turn_idx, attribute|'object'|'date', text, score)] — только внутри самой реплики."""
        tx = self.texts(call)
        res = []
        for k in range(0, len(tx), self.batch):
            chunk = tx[k:k + self.batch]
            preds = self.model.batch_predict_entities([c[0] for c in chunk], self.labels, threshold=self.raw_thr, flat_ner=False)
            for q, ((text, a, b), ents) in enumerate(zip(chunk, preds)):
                ti = call["turns"][k + q]["idx"]
                for e in ents:
                    if e["start"] >= a and e["end"] <= b + 1:
                        res.append((ti, ATTR_OF.get(e["label"], e["label"]), e["text"], float(e["score"])))
        return res


def fuse(ex, scored, spans, g):
    """Вернуть новый список кандидатов (a, turn_idx, value, p) с учётом спанов GLiNER."""
    if not int(g.get("gl_on", 1)) or spans is None:
        return scored["cands"]
    by = {}
    for ti, a, text, sc in spans:
        if a in ex.banks and sc >= g.get(f"gl_thr_{a}", 0.4):
            by.setdefault((ti, a), []).append((text, sc))
    out = []
    seen = set()
    for a, ti, v, p in scored["cands"]:
        sp = by.get((ti, a))
        if sp:
            vs = [stem(w) for w in words(v) if len(w) >= 3 and w != "не"]
            best = 0.0
            for text, sc in sp:
                ts = {stem(w) for w in words(text)}
                cov = sum(1 for s in vs if s in ts) / max(len(vs), 1)
                if cov >= g["gl_min_cov"]:
                    best = max(best, sc)
            p = min(1.0, p + g["gl_w_boost"] * best)
        out.append((a, ti, v, p))
        seen.add((a, ti, norm(v)))
    # новые кандидаты из спанов: ближайшее значение банка
    for (ti, a), sp in by.items():
        bank = ex.banks[a]
        for text, sc in sp:
            x = normalize(ex.vec.transform([text]))
            sims = (x @ bank["lex"].T).toarray()[0]
            j = int(np.argmax(sims))
            if sims[j] < 0.35:
                continue
            v = bank["values"][j]
            key = (a, ti, norm(v))
            if key in seen:
                continue
            seen.add(key)
            out.append((a, ti, v, float(g["gl_w_new"] * sc * sims[j])))
    return out


def load_finetuned(base_dir, adapter_path, rank=16, alpha=32, targets="q,k,v,o,wi_0,wi_1,wo", merge=True):
    """База GLiNER + LoRA-слои + веса адаптера (LoRA и головы); merge=True вливает LoRA в веса энкодера."""
    import torch
    from gliner import GLiNER
    from peft import LoraConfig, inject_adapter_in_model
    from safetensors.torch import load_file
    model = GLiNER.from_pretrained(base_dir)
    enc = model.model.token_rep_layer.bert_layer.model
    inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets.split(",")), enc)
    sd = load_file(adapter_path)
    missing, unexpected = model.model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected[:5]
    if merge:
        for name, mod in list(enc.named_modules()):
            if hasattr(mod, "merge") and hasattr(mod, "lora_A"):
                mod.merge()
    from gliner.data_processing.tokenizer import WordsSplitter
    model.data_processor.words_splitter = WordsSplitter("whitespace")   # как при обучении: \w+ | знак
    model.config.words_splitter_type = "whitespace"
    model.eval()
    return model
