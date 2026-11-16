"""Шумоподавитель ASR: исправление искажённых слов по модели зашумлённого канала.

Словарь чистых слов: (1) цитаты из ранней версии корпуса, записанные ДО наложения ASR-шума,
(2) каталог, (3) частотные слова обучающего корпуса (шумовые варианты редки, настоящие слова
частотны). Незнакомый токен заменяется ближайшим словарным словом той же длины ±2 по
похожести символов, с приоритетом частоты. Пунктуация и регистр сохраняются.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher

WORD = re.compile(r"[А-Яа-яЁё]+")
# частые замены первой буквы в шуме корпуса (оглушение/озвончение, редукция гласных)
SWAP = {"о": "а", "а": "о", "е": "и", "и": "е", "д": "т", "т": "д", "в": "ф", "ф": "в", "б": "п", "п": "б", "з": "с", "с": "з", "г": "к", "к": "г"}


class Denoiser:
    def __init__(self, min_freq=15, cutoff=0.78, min_len=4):
        self.min_freq, self.cutoff, self.min_len = min_freq, cutoff, min_len
        self.vocab = Counter()
        self.by_key = {}
        self.cache = {}

    def fit(self, train_calls, clean_texts=(), catalog=None):
        freq = Counter(w for c in train_calls for t in c["turns"] for w in WORD.findall(t["text"].lower().replace("ё", "е")))
        clean = Counter(w for s in clean_texts for w in WORD.findall(s.lower().replace("ё", "е")))
        for w, n in freq.items():
            if n >= self.min_freq:
                self.vocab[w] += n
        for w, n in clean.items():
            self.vocab[w] += max(n, self.min_freq)
        if catalog:
            for vals in catalog.values():
                for v in vals:
                    for w in WORD.findall(v.lower().replace("ё", "е")):
                        self.vocab[w] += self.min_freq
        for w in self.vocab:
            self.by_key.setdefault((w[0], len(w)), []).append(w)
        return self

    def correct_word(self, w):
        lw = w.lower().replace("ё", "е")
        if len(lw) < self.min_len or lw in self.vocab:
            return w
        if lw in self.cache:
            r = self.cache[lw]
        else:
            best, bs = None, 0.0
            firsts = {lw[0]} | set(SWAP.get(lw[0], ""))
            for f in firsts:
                for L in range(len(lw) - 2, len(lw) + 3):
                    for cand in self.by_key.get((f, L), ()):
                        s = SequenceMatcher(None, lw, cand).ratio()
                        if s > bs or (s == bs and best and self.vocab[cand] > self.vocab[best]):
                            best, bs = cand, s
            r = best if bs >= self.cutoff else None
            self.cache[lw] = r
        if not r:
            return w
        return r.capitalize() if w[:1].isupper() else r

    def __call__(self, text):
        return WORD.sub(lambda m: self.correct_word(m.group(0)), text)


def clean_quotes(old_calls):
    """Цитаты ранней версии корпуса (до ASR-шума)."""
    return [f["source"]["quote"] for c in old_calls for f in c["facts"] if f["source"].get("quote")]
