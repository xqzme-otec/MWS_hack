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


# ============================================================== разметка v2
# v1 размечала только окна с нечётким совпадением слов в реплике-источнике: 24–60% фактов (по атрибутам)
# оставались без спана, а такие реплики учили модель «здесь факта нет». v2:
#   • совпадение и по словам, и по символьным триграммам (перефразы, ASR-ошибки);
#   • договорённость/действие ищутся и в предыдущей реплике (предложение оператора → «да, подходит»);
#   • примеры с фактом, которому не нашлось спана, отбрасываются (не учим ложным отрицаниям);
#   • опционально — копии примеров с ASR-шумом (устойчивость, adversarial-аугментация).
from .text import norm as _norm


def _tri(s):
    s = " " + _norm(s) + " "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def tri_window(toks, value, max_extra=4, thr=0.5):
    tv = _tri(value)
    n = len(value.split())
    ws = [t for t, _, _ in toks]
    best, bs = None, 0.0
    for i in range(len(ws)):
        for j in range(i, min(len(ws), i + n + max_extra)):
            tw = _tri(" ".join(ws[i:j + 1]))
            inter = len(tv & tw)
            sc = 0.7 * inter / max(len(tv), 1) + 0.3 * inter / max(len(tw), 1)
            if sc > bs:
                bs, best = sc, (i, j)
    return best if bs >= thr else None


def find_window(toks, value, thr=0.5):
    return best_window(toks, value) or tri_window(toks, value, thr=thr)


def _noise_token(tok, rng, level, subs, tot):
    if len(tok) < 4 or rng.random() > level:
        return tok
    ch = list(tok)
    for _ in range(1 + (rng.random() < level)):
        x = rng.random() * tot
        for a, b, n in subs:
            x -= n
            if x <= 0:
                idx = [i for i, c in enumerate(ch) if c == a]
                if idx:
                    ch[rng.choice(idx)] = b
                break
    return "".join(ch)


def build_examples_v2(calls, neg_ratio=0.3, seed=42, context_words=25, max_width=12, tri_thr=0.5,
                      prev_attrs=("agreement", "action"), drop_unplaced=True, noise_copies=0, noise_level=0.3):
    rng = random.Random(seed)
    subs = None
    if noise_copies:
        from synth.generator import SUBS as subs  # модель ASR-шума: замены букв, выученные по парам цитата↔реплика
        tot = sum(n for _, _, n in subs)
    out, stats = [], {"facts": 0, "placed": 0, "dropped_examples": 0}
    for c in calls:
        cd = datetime.fromisoformat(c["started_at"]).date()
        turns = c["turns"]
        pos = {t["idx"]: i for i, t in enumerate(turns)}
        # где размечать каждый факт: (позиция реплики, факт)
        place, unplaced_at = {}, set()
        for f in c["facts"]:
            ti = f["source"]["turn_idx"]
            if ti not in pos:
                continue
            stats["facts"] += 1
            i = pos[ti]
            gq = f.get("gl_quote")                       # дистиллированные звонки: точная цитата значения
            if gq and gq.get("turn_idx") in pos:
                place.setdefault(pos[gq["turn_idx"]], []).append(f); stats["placed"] += 1
                continue
            if find_window(tokenize(turns[i]["text"]), f["value"], tri_thr):
                place.setdefault(i, []).append(f); stats["placed"] += 1
            elif f["attribute"] in prev_attrs and i > 0 and find_window(tokenize(turns[i - 1]["text"]), f["value"], tri_thr):
                place.setdefault(i - 1, []).append(f); stats["placed"] += 1
            else:
                unplaced_at.add(i)
                if i > 0 and f["attribute"] in prev_attrs:
                    unplaced_at.add(i - 1)
        for i, t in enumerate(turns):
            facts = place.get(i, [])
            if drop_unplaced and i in unplaced_at:
                stats["dropped_examples"] += 1
                continue
            if not facts and rng.random() > neg_ratio:
                continue
            prefix = SPK.get(t["speaker"], "") + " "
            body = t["text"]
            nxt = turns[i + 1] if i + 1 < len(turns) else None
            suffix = (" || " + SPK.get(nxt["speaker"], "") + " " + " ".join(nxt["text"].split()[:context_words])) if nxt else ""
            text = prefix + body + suffix
            off, body_end = len(prefix), len(prefix) + len(body)
            toks = tokenize(text)
            body_toks = [x for x in toks if x[2] <= body_end and x[1] >= off]
            spans = []
            for f in facts:
                gq = f.get("gl_quote")
                if gq:
                    m = re.search(r"\s+".join(map(re.escape, gq["text"].split())), body, flags=re.I)
                    ab = char_to_tok(toks, off + m.start(), off + m.end()) if m else None
                    if ab and ab[1] - ab[0] < max_width:
                        spans.append([ab[0], ab[1], LABELS[f["attribute"]]])
                    continue
                w = find_window(body_toks, f["value"], tri_thr)
                if w and w[1] - w[0] < max_width:
                    spans.append([toks.index(body_toks[w[0]]), toks.index(body_toks[w[1]]), LABELS[f["attribute"]]])
                name = f["object"]["name"]
                if f["object"]["type"] in ("service", "device") and name:
                    w = best_window(body_toks, name, max_extra=1)
                    if w:
                        spans.append([toks.index(body_toks[w[0]]), toks.index(body_toks[w[1]]), LABELS["object"]])
            for typ, a, b, ds in extract(body, cd):
                ab = char_to_tok(toks, off + a, off + b)
                if ab and ab[1] - ab[0] < max_width:
                    spans.append([ab[0], ab[1], LABELS["date"]])
            uniq = sorted({(s, e, l) for s, e, l in spans})
            ex = {"tokenized_text": [x[0] for x in toks], "ner": [list(u) for u in uniq],
                  "text": text, "char_ner": [[toks[s][1], toks[e][2], l] for s, e, l in uniq],   # для gliner2
                  "meta": {"call_id": c["call_id"], "turn_idx": t["idx"]}}
            out.append(ex)
            for _ in range(noise_copies):
                nt = [_noise_token(x, rng, noise_level, subs, tot) for x in ex["tokenized_text"]]
                chars = list(text)                     # замены букв 1:1 → позиции спанов не сдвигаются
                for (tok, a_, b_), new_tok in zip(toks, nt):
                    if len(new_tok) == b_ - a_ and a_ >= off and b_ <= body_end:   # шум только в самой реплике
                        chars[a_:b_] = list(new_tok)
                out.append({"tokenized_text": nt, "ner": ex["ner"], "text": "".join(chars), "char_ner": ex["char_ner"],
                            "meta": {**ex["meta"], "noise": noise_level}})
    return out, stats
