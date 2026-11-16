"""GLiNER2.5 (metrika-labs/ru-pii-ner-gliner2.5) для CompactFact: урезание архитектуры, данные, оценка, спаны.

Модель: энкодер mDeBERTa-v3-base (12 слоёв) + граничная голова GLiNER2. Из 287 млн параметров 192 млн — словарь
на 250 тыс. токенов ~100 языков. «Убираем мусор» двумя способами:
  1) словарь урезается до токенов, реально встречающихся в русских звонках (train + дистилляция + метки);
     неизвестный токен → [UNK]. Скорость почти не меняется, память и размер файла — в разы меньше;
  2) число слоёв энкодера — ген эволюции (3…12): это и есть задержка.
Сохранение своё (torch.save состояния + архитектура): стандартный save_pretrained урезанный словарь не понимает.
"""
from __future__ import annotations

import json, os, re, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from cfx.gliner_data import LABELS, build_examples_v2, SPK  # noqa: E402

BASE = "metrika-labs/ru-pii-ner-gliner2.5"
FACTS = ("service_state", "tariff", "ticket_status", "action", "agreement")
FACT_LABELS = [LABELS[a] for a in FACTS]
ATTR_OF = {LABELS[a]: a for a in FACTS}


def device():
    if os.environ.get("CFX_DEVICE"):
        return torch.device(os.environ["CFX_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------ архитектура
class PrunedEmbedding(nn.Module):
    """Словарь только из нужных токенов; старый id → новый через таблицу, остальное → [UNK]."""

    def __init__(self, emb: nn.Embedding, keep_ids, unk_id):
        super().__init__()
        keep = torch.tensor(sorted(set(int(i) for i in keep_ids) | {int(unk_id)}), dtype=torch.long)
        self.weight = nn.Parameter(emb.weight.detach()[keep].clone())
        remap = torch.full((emb.num_embeddings,), int((keep == unk_id).nonzero()[0, 0]), dtype=torch.long)
        remap[keep] = torch.arange(len(keep))
        self.register_buffer("remap", remap, persistent=True)
        self.num_embeddings, self.embedding_dim, self.padding_idx = emb.num_embeddings, emb.embedding_dim, emb.padding_idx

    def forward(self, ids):
        return F.embedding(self.remap[ids], self.weight)


def encoder_of(model):
    return model.encoder


def apply_arch(model, keep_layers=0, keep_ids=None):
    enc = encoder_of(model)
    L = enc.encoder.layer
    if keep_layers and keep_layers < len(L):
        enc.encoder.layer = nn.ModuleList(list(L)[:keep_layers])
        enc.config.num_hidden_layers = keep_layers
    if keep_ids is not None and not isinstance(enc.embeddings.word_embeddings, PrunedEmbedding):
        tok = model.processor.tokenizer if hasattr(model, "processor") else None
        unk = getattr(tok, "unk_token_id", None) if tok is not None else None
        enc.embeddings.word_embeddings = PrunedEmbedding(enc.embeddings.word_embeddings, keep_ids, unk if unk is not None else 3)
    return len(enc.encoder.layer)


def tokenizer_of(model):
    for attr in ("processor", "data_processor"):
        p = getattr(model, attr, None)
        if p is not None:
            for t in ("tokenizer", "transformer_tokenizer"):
                if getattr(p, t, None) is not None:
                    return getattr(p, t)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")


def vocab_ids(model, texts):
    """id токенов, которые модель реально увидит на этом корпусе. GLiNER2 режет текст своим сплиттером,
    приводит КАЖДОЕ слово к нижнему регистру и токенизирует слова по отдельности — повторяем ровно это
    (+ служебные токены схемы и все добавленные токены)."""
    from gliner2.processing.word_splitter import WhitespaceTokenSplitter
    tok = tokenizer_of(model)
    split = WhitespaceTokenSplitter()
    ids = set(tok.all_special_ids) | set(tok.get_added_vocab().values())
    proc = getattr(model, "processor", None)
    extra = list(getattr(proc, "SPECIAL_TOKENS", [])) + ["(", ")", ",", "|"]
    words = set(extra)
    for t in texts:
        for w, _, _ in split(t, lower=True):
            words.add(w)
        for w, _, _ in split(t, lower=False):
            words.add(w)
    for w in words:
        ids.update(tok.convert_tokens_to_ids(tok.tokenize(w)))
    return sorted(i for i in ids if i is not None)


def load_base(base=BASE):
    from gliner2 import AutoExtractor
    return AutoExtractor.from_pretrained(base)


def save(model, out_dir, arch):
    os.makedirs(out_dir, exist_ok=True)
    m = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model
    torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, os.path.join(out_dir, "weights.pt"))
    json.dump(arch, open(os.path.join(out_dir, "arch.json"), "w"), ensure_ascii=False)


def load(out_dir):
    arch = json.load(open(os.path.join(out_dir, "arch.json")))
    m = load_base(arch.get("base", BASE))
    keep = json.load(open(os.path.join(out_dir, "vocab.json"))) if os.path.exists(os.path.join(out_dir, "vocab.json")) else None
    apply_arch(m, arch.get("keep_layers", 0), keep)
    sd = torch.load(os.path.join(out_dir, "weights.pt"), map_location="cpu")
    missing, unexpected = m.load_state_dict(sd, strict=False)
    bad = [k for k in missing if "lora" not in k]
    if bad or unexpected:
        print("load_state_dict: missing", bad[:5], "unexpected", unexpected[:5])
    return m.eval(), arch


# ------------------------------------------------------------------ данные
def examples(calls, ctx=0, neg_ratio=0.2, noise_copies=0, noise_level=0.3, seed=42, drop_unplaced=True):
    """Реплики → (text, [(start, end, label)]) только с 5 метками фактов (сырой текст, как при поиске)."""
    E, st = build_examples_v2(calls, neg_ratio=neg_ratio, seed=seed, context_words=ctx, max_width=40,
                              drop_unplaced=drop_unplaced, noise_copies=noise_copies, noise_level=noise_level)
    out = [(e["text"], [(a, b, l) for a, b, l in e["char_ner"] if l in ATTR_OF], "noise" in e["meta"]) for e in E]
    return out, st


def input_examples(exs, neg_labels=2, seed=42):
    """GLiNER2 учит «сколько сущностей каждого типа». Если в каждом примере все 5 типов, а 4–5 из них пустые,
    счётчик учится отвечать 0 и модель не находит ничего. Поэтому: в положительном примере — его типы
    + neg_labels случайных пустых; в отрицательном — neg_labels пустых."""
    import random as _r
    from gliner2.training.data import InputExample
    rng = _r.Random(seed)
    res = []
    for text, spans, _ in exs:
        ents = {}
        for a, b, l in spans:
            ents.setdefault(l, [])
            if text[a:b] not in ents[l]:
                ents[l].append(text[a:b])
        empty = [l for l in FACT_LABELS if l not in ents]
        for l in rng.sample(empty, min(neg_labels, len(empty))):
            ents[l] = []
        res.append(InputExample(text=text, entities=ents))
    return res


# ------------------------------------------------------------------ поиск спанов
def _ents(res):
    """Ответ batch_extract_entities → [(label, start, end, score, text)]."""
    out = []
    ent = res.get("entities", res) if isinstance(res, dict) else {}
    for lab, items in (ent or {}).items():
        for it in items or []:
            if isinstance(it, dict):
                out.append((lab, it.get("start"), it.get("end"), float(it.get("confidence", it.get("score", 1.0))), it.get("text", "")))
    return out


@torch.no_grad()
def predict(model, texts, thr=0.3, batch=16):
    model.eval()
    res = []
    for k in range(0, len(texts), batch):
        r = model.batch_extract_entities(texts[k:k + batch], FACT_LABELS, threshold=thr,
                                         include_confidence=True, include_spans=True)
        res.extend(_ents(x) for x in r)
    return res


def span_f1(model, exs, thr=0.5):
    preds = predict(model, [t for t, _, _ in exs], thr)
    tp = fp = fn = ov = 0
    for (text, gold, _), pr in zip(exs, preds):
        g = {(a, b, l) for a, b, l in gold}
        p = {(s, e, l) for l, s, e, _, _ in pr if s is not None}
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        ov += sum(any(l == gl and s < gb and ga < e for ga, gb, gl in g) for s, e, l in p)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return {"f1": round(2 * P * R / max(P + R, 1e-9), 4), "p": round(P, 4), "r": round(R, 4), "overlap_p": round(ov / max(tp + fp, 1), 4)}


def call_texts(call, ctx=0):
    out, turns = [], call["turns"]
    for i, t in enumerate(turns):
        prefix = SPK.get(t["speaker"], "") + " "
        nxt = turns[i + 1] if i + 1 < len(turns) else None
        suffix = (" || " + SPK.get(nxt["speaker"], "") + " " + " ".join(nxt["text"].split()[:ctx])) if nxt else ""
        out.append((prefix + t["text"] + suffix, len(prefix), len(prefix) + len(t["text"])))
    return out


def call_spans(model, call, ctx=0, thr=0.1):
    """[(turn_idx, attribute, text, score)] — формат кэша спанов для run_evolve_gcore.py (GLiNER-core)."""
    tx = call_texts(call, ctx)
    preds = predict(model, [t for t, _, _ in tx], thr)
    out = []
    for (text, a, b), pr, turn in zip(tx, preds, call["turns"]):
        for lab, s, e, sc, _ in pr:
            if s is not None and s >= a and e <= b:
                out.append((turn["idx"], ATTR_OF[lab], text[s:e], sc))
    return out
