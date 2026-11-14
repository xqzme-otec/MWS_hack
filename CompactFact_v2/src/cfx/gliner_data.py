"""Разметка спанов для дообучения GLiNER из эталонных фактов.

Для каждой реплики-источника: спан значения факта (эталонный span, если есть,
иначе — окно реплики с наибольшим покрытием слов значения), спаны дат (банк выражений)
и упоминания объектов. Вход модели — реплика + короткий контекст следующей реплики
(нужен для ловушек: «предложение оператора → отказ клиента»); метки ставятся только
внутри самой реплики.
"""
from __future__ import annotations

import random
import re
from datetime import datetime

from .dates_bank import extract
from .objects import word_close
from .text import words

LABELS = {
    "service_state": "состояние или неисправность услуги",
    "tariff": "тариф клиента",
    "ticket_status": "статус заявки",
    "action": "действие, которое выполнил клиент",
    "agreement": "договорённость с оператором",
    "object": "услуга или устройство",
    "date": "дата или срок",
}
TOK = re.compile(r"[\w+\-]+|[^\w\s]", re.UNICODE)
SPK = {"client": "Клиент:", "operator": "Оператор:"}


def tokenize(text):
    return [(m.group(0), m.start(), m.end()) for m in TOK.finditer(text)]


def char_to_tok(toks, a, b):
    s = next((i for i, (_, x, y) in enumerate(toks) if y > a), None)
    e = next((i for i in range(len(toks) - 1, -1, -1) if toks[i][1] < b), None)
    if s is None or e is None or e < s:
        return None
    return s, e


def best_window(toks, value, max_extra=3):
    """Окно токенов реплики с максимальным покрытием слов значения (нечётко)."""
    vw = [w for w in words(value) if len(w) >= 3 and w != "не"]
    if not vw:
        return None
    low = [t.lower().replace("ё", "е") for t, _, _ in toks]
    hits = [i for i, t in enumerate(low) if any(word_close(w, t) for w in vw)]
    if not hits:
        return None
    best, best_sc = None, 0
    for i in hits:
        for j in hits:
            if j < i or j - i > len(vw) + max_extra:
                continue
            cov = len({w for w in vw if any(word_close(w, low[k]) for k in range(i, j + 1))}) / len(vw)
            sc = cov - 0.02 * (j - i)
            if sc > best_sc:
                best, best_sc = (i, j), sc
    if best is None:
        return None
    cov = len({w for w in vw if any(word_close(w, low[k]) for k in range(best[0], best[1] + 1))}) / len(vw)
    return best if cov >= 0.6 else None


def build_examples(calls, resolver=None, neg_ratio=0.3, seed=42, context_words=25, max_width=12):
    rng = random.Random(seed)
    out = []
    for c in calls:
        cd = datetime.fromisoformat(c["started_at"]).date()
        by_turn = {}
        for f in c["facts"]:
            if f["source"]["turn_idx"] is not None:
                by_turn.setdefault(f["source"]["turn_idx"], []).append(f)
        for i, t in enumerate(c["turns"]):
            facts = by_turn.get(t["idx"], [])
            if not facts and rng.random() > neg_ratio:
                continue
            prefix = SPK.get(t["speaker"], "") + " "
            body = t["text"]
            nxt = c["turns"][i + 1] if i + 1 < len(c["turns"]) else None
            suffix = (" || " + SPK.get(nxt["speaker"], "") + " " + " ".join(nxt["text"].split()[:context_words])) if nxt else ""
            text = prefix + body + suffix
            off = len(prefix)
            toks = tokenize(text)
            body_end = off + len(body)
            spans = []
            for f in facts:
                sp = f["source"].get("span")
                ab = None
                if sp and 0 <= sp[0] < sp[1] <= len(body):
                    # эталонный спан обычно длиннее значения — сужаем до окна значения внутри него
                    sub = [x for x in toks if x[1] >= off + sp[0] and x[2] <= off + sp[1]]
                    w = best_window(sub, f["value"]) if sub else None
                    if w:
                        a0 = toks.index(sub[w[0]]); b0 = toks.index(sub[w[1]])
                        ab = (a0, b0)
                if ab is None:
                    body_toks = [x for x in toks if x[2] <= body_end and x[1] >= off]
                    w = best_window(body_toks, f["value"])
                    if w:
                        ab = (toks.index(body_toks[w[0]]), toks.index(body_toks[w[1]]))
                if ab and ab[1] - ab[0] < max_width:
                    spans.append([ab[0], ab[1], LABELS[f["attribute"]]])
                name = f["object"]["name"]
                if f["object"]["type"] in ("service", "device") and name:
                    body_toks = [x for x in toks if x[2] <= body_end and x[1] >= off]
                    w = best_window(body_toks, name, max_extra=1)
                    if w:
                        spans.append([toks.index(body_toks[w[0]]), toks.index(body_toks[w[1]]), LABELS["object"]])
            for typ, a, b, ds in extract(body, cd):
                ab = char_to_tok(toks, off + a, off + b)
                if ab and ab[1] - ab[0] < max_width:
                    spans.append([ab[0], ab[1], LABELS["date"]])
            uniq = {(s, e, l) for s, e, l in spans}
            out.append({"tokenized_text": [x[0] for x in toks], "ner": [list(u) for u in sorted(uniq)],
                        "meta": {"call_id": c["call_id"], "turn_idx": t["idx"], "body": [off, body_end]}})
    return out
