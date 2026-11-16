"""Привязка фактов к объектам: услуги/устройства (каталог + имена из train), заявки, клиент.

Каталог используется ПОСЛЕ извлечения (кейс §1.6): упоминание в реплике сопоставляется с
именами по нечёткому совпадению слов (устойчиво к ASR: «раутер»≈«роутер»), без склейки
текста до извлечения. Специфичность имени (число совпавших слов) решает при конфликте.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from .text import words

TICKET_RX = re.compile(r"заявк\w*\s*(?:номер\s*|№\s*)?(\d{3,6})")
STOP = {"для", "при", "без", "над", "под", "про", "на", "по", "в", "с", "и", "к", "от", "до", "за", "из"}
_wcache: dict = {}


def word_close(a: str, b: str) -> bool:
    """Нечёткое совпадение слов, терпимое к шуму распознавания."""
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    if abs(len(a) - len(b)) > 3:
        return False
    k = (a, b)
    r = _wcache.get(k)
    if r is None:
        r = a[:5] == b[:5] if min(len(a), len(b)) >= 6 and a[:5] == b[:5] else SequenceMatcher(None, a, b).ratio() >= 0.75
        _wcache[k] = r
    return r


def name_words(name):
    return [w for w in words(name) if w not in STOP and len(w) >= 2]


class Matcher:
    """Поиск упоминаний имён из словаря в тексте (по словам, нечётко, с кэшем «токен → слова имён»)."""

    def __init__(self, names, min_cov=0.99):
        self.items = []
        vocab = set()
        for n in names:
            ws = name_words(n)
            if ws:
                self.items.append((n, ws))
                vocab.update(ws)
        self.vocab = sorted(vocab)
        self.by_first = defaultdict(list)
        for w in self.vocab:
            self.by_first[w[:1]].append(w)
        self.min_cov = min_cov
        self._tok = {}

    def _match_token(self, t):
        r = self._tok.get(t)
        if r is None:
            pool = self.vocab if len(t) >= 6 else self.by_first.get(t[:1], [])
            r = frozenset(w for w in pool if word_close(w, t))
            if len(self._tok) < 300_000:
                self._tok[t] = r
        return r

    def find(self, text):
        present = set()
        for t in set(words(text)):
            present |= self._match_token(t)
        if not present:
            return []
        found = []
        for n, ws in self.items:
            hit = sum(1 for w in ws if w in present)
            if not hit:
                continue
            cov = hit / len(ws)
            if cov >= self.min_cov or (len(ws) >= 3 and cov >= 0.66):
                found.append((n, cov, hit))
        found.sort(key=lambda x: (-x[2], -x[1]))
        return found


class ObjectResolver:
    def __init__(self, catalog, train_calls=(), min_count=2):
        names = list(catalog.get("services", [])) + list(catalog.get("devices", []))
        cnt = Counter(f["object"]["name"] for c in train_calls for f in c["facts"]
                      if f["object"]["type"] in ("service", "device") and f["object"]["name"])
        names += [n for n, k in cnt.items() if k >= min_count]
        uniq = {}
        for n in names:
            uniq.setdefault(n.lower(), n)
        self.matcher = Matcher(list(uniq.values()))
        self.tariffs = Matcher(list(catalog.get("tariffs", [])))

    def prepare(self, call):
        per_turn = [self.matcher.find(t["text"]) for t in call["turns"]]
        freq = Counter()
        for m in per_turn:
            for n, cov, hit in m[:1]:
                freq[n] += 1
        tickets = []
        for i, t in enumerate(call["turns"]):
            for m in TICKET_RX.finditer(t["text"].lower()):
                tickets.append((i, m.group(1)))
        return {"per_turn": per_turn, "freq": freq, "tickets": tickets}

    def resolve(self, call, ctx, turn_i, attribute, back=6):
        if attribute in ("tariff", "action", "agreement"):
            return {"type": "client", "name": "клиент"}
        if attribute == "ticket_status":
            near = [n for i, n in ctx["tickets"] if i <= turn_i + 1]
            num = near[-1] if near else (ctx["tickets"][0][1] if ctx["tickets"] else None)
            return {"type": "ticket", "name": f"заявка {num}" if num else "заявка"}
        for j in range(turn_i, max(-1, turn_i - back), -1):
            if ctx["per_turn"][j]:
                return {"type": "service", "name": ctx["per_turn"][j][0][0]}
        if turn_i + 1 < len(ctx["per_turn"]) and ctx["per_turn"][turn_i + 1]:
            return {"type": "service", "name": ctx["per_turn"][turn_i + 1][0][0]}
        if ctx["freq"]:
            return {"type": "service", "name": ctx["freq"].most_common(1)[0][0]}
        return {"type": "service", "name": ""}


# ---------------- обучаемый выбор объекта для service_state ----------------
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def object_candidates(ctx, turn_i, speakers):
    first, cnt, cli, opr, near_before, near_after, in_turn, best_cov = {}, Counter(), Counter(), Counter(), {}, {}, {}, {}
    order = []
    for j, ms in enumerate(ctx["per_turn"]):
        for rank, (n, cov, hit) in enumerate(ms[:4]):
            if n not in first:
                first[n] = j
                order.append(n)
            cnt[n] += 1
            (cli if speakers[j] == "client" else opr)[n] += 1
            best_cov[n] = max(best_cov.get(n, 0), hit)
            if j <= turn_i:
                near_before[n] = turn_i - j
            elif n not in near_after:
                near_after[n] = j - turn_i
            if j == turn_i:
                in_turn[n] = min(in_turn.get(n, 9), rank)
    names = order
    X = []
    for k, n in enumerate(names):
        X.append([near_before.get(n, 99), near_after.get(n, 99), float(n in in_turn), in_turn.get(n, 9), cnt[n], cli[n],
                  opr[n], first[n] / max(len(speakers), 1), float(k == 0), k, best_cov[n], len(n.split()),
                  float(near_before.get(n, 99) == min(near_before.values(), default=99))])
    return names, np.asarray(X, np.float32)


class ObjectChooser:
    def __init__(self, resolver, seed=42):
        self.r, self.seed, self.m = resolver, seed, None

    def fit(self, calls, same):
        Xs, ys = [], []
        for c in calls:
            facts = [f for f in c["facts"] if f["attribute"] == "service_state" and f["object"]["name"]]
            if not facts:
                continue
            ctx = self.r.prepare(c)
            spk = [t["speaker"] for t in c["turns"]]
            for f in facts:
                names, X = object_candidates(ctx, f["source"]["turn_idx"] or 0, spk)
                if not names:
                    continue
                Xs.append(X)
                ys.append(np.array([same(f, n) for n in names], np.int8))
        X, y = np.vstack(Xs), np.concatenate(ys)
        self.m = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, learning_rate=0.08,
                                                random_state=self.seed).fit(X, y)
        return self

    def choose(self, ctx, turn_i, speakers):
        names, X = object_candidates(ctx, turn_i, speakers)
        if not names:
            return ""
        p = self.m.predict_proba(X)[:, 1]
        return names[int(np.argmax(p))]


def refresh(resolver):
    """Пересобрать матчеры после обновления кода (совместимость со старыми pickle)."""
    resolver.matcher = Matcher([n for n, _ in resolver.matcher.items])
    resolver.tariffs = Matcher([n for n, _ in resolver.tariffs.items])
    return resolver
