"""Обучаемый выбор дат факта.

Кандидаты: относительные выражения в окне реплик вокруг факта (банк повторяет
восстановитель оценщика v2.1), дата звонка и «скрытое прошлое» — для состояний,
начало которых в тексте не названо. Бустинг выбирает кандидата; valid_to для
одно-значных атрибутов дополнительно закрывается цепочкой состояний.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .dates_bank import extract
from .text import ATTRIBUTES, norm, words, stem

TYPES = ["days_ago", "from_wd", "until_wd", "on_wd_past", "on_wd_next", "in_days", "fixed_позавчера", "fixed_вчера",
         "fixed_сегодня", "fixed_завтра", "fixed_послезавтра", "week_ago", "week_ahead", "last_week", "month_ago",
         "last_month", "next_month", "month_ahead", "two_weeks", "next_week", "pozaprosh", "these_days", "callday", "hidden"]
# представитель многодневного выражения: самый частый сдвиг генератора на train-fit
REP_OFFSET = {"last_week": -6, "month_ago": -28, "last_month": -21, "next_month": 33, "month_ahead": 30,
              "two_weeks": 14, "next_week": 7, "pozaprosh": -3, "these_days": -3}
HIDDEN_DAYS = 400
VALUE_FLAGS = ["работает нормально", "больше не используется", "принята", "в работе", "назначен выезд специалиста", "выполнена"]


from difflib import get_close_matches
import re as _re
DATE_WORDS = ["позавчера", "вчера", "сегодня", "завтра", "послезавтра", "назад", "через", "неделю", "неделе", "недели",
              "месяц", "месяце", "прошлой", "прошлом", "следующей", "следующем", "понедельник", "вторник", "среду",
              "четверг", "пятницу", "субботу", "воскресенье", "понедельника", "вторника", "среды", "четверга",
              "пятницы", "субботы", "воскресенья", "дней", "дня", "день", "днях"]
_DW = set(DATE_WORDS)
_fix_cache = {}


def fix_date_words(text):
    """Исправляет искажённые ASR слова дат («автра»→«завтра», «медяц»→«месяц»)."""
    def rep(m):
        w = m.group(0)
        lw = w.lower()
        if len(lw) < 4 or lw in _DW:
            return w
        if lw not in _fix_cache:
            c = get_close_matches(lw, DATE_WORDS, n=1, cutoff=0.8)
            _fix_cache[lw] = c[0] if c else None
        return _fix_cache[lw] or w
    return _re.sub(r"[А-Яа-яЁё]+", rep, text)


def call_day(call):
    return datetime.fromisoformat(call["started_at"]).date()


def expressions(text, cd):
    """[(type, start, end, date, all_dates)] — по одному представителю на выражение (on_wd — два)."""
    out = []
    for t, a, b, ds in extract(text, cd):
        if t == "on_wd":
            out.append(("on_wd_past", a, b, ds[0], {ds[0]}))
            out.append(("on_wd_next", a, b, ds[1], {ds[1]}))
        elif len(ds) == 1:
            out.append((t, a, b, ds[0], {ds[0]}))
        else:
            off = REP_OFFSET.get(t)
            d = cd + timedelta(days=off) if off is not None and (cd + timedelta(days=off)) in ds else ds[len(ds) // 2]
            out.append((t, a, b, d, set(ds)))
    return out


def value_position(text, value):
    """Позиция первого слова значения в тексте реплики (по стему) — 0..1 или -1."""
    low = text.lower().replace("ё", "е")
    for w in words(value):
        if len(w) < 4:
            continue
        k = low.find(stem(w))
        if k >= 0:
            return k / max(len(low), 1)
    return -1.0


class DateSelector:
    def __init__(self, window=(-3, 2), seed=42, iters=200):
        self.window, self.seed, self.iters = window, seed, iters
        self.models = {}

    # ---------- кандидаты и признаки ----------
    def candidates(self, call, turn_i, attribute, value, speaker):
        cd = call_day(call)
        turns = call["turns"]
        vpos = value_position(turns[turn_i]["text"], value) if 0 <= turn_i < len(turns) else -1
        vwords = [w for w in words(value) if len(w) >= 4]
        cands = []
        n_in_turn = 0
        for j in range(turn_i + self.window[0], turn_i + self.window[1] + 1):
            if not 0 <= j < len(turns):
                continue
            text = turns[j]["text"]
            low = text.lower()
            exprs = expressions(fix_date_words(text), cd)
            if j == turn_i:
                n_in_turn = len(exprs)
            for rank, (t, a, b, d, dset) in enumerate(exprs):
                before = low[max(0, a - 4):a]
                rel = a / max(len(text), 1)
                # локальная связь выражения со значением: слова значения рядом, граница клаузы между ними
                win = low[max(0, a - 60):b + 60]
                vhit = sum(1 for w in vwords if stem(w) in win) / max(len(vwords), 1)
                if vpos >= 0 and j == turn_i:
                    vp = int(vpos * len(text))
                    seg = low[min(vp, a):max(vp, a)]
                    punct = sum(seg.count(ch) for ch in ".!?;—")
                    chars = abs(vp - a) / 200
                else:
                    punct, chars = 5, 2.0
                after = low[b:b + 40]
                local = [vhit, punct, chars, float(before.strip().endswith(("с", "со"))), float(before.strip().endswith(("в", "во"))),
                         float(bool(_re.search(r"принят|в работ|выполн|назнач", win))), float(bool(_re.search(r"ещ[её]|уже|только", low[max(0, a - 15):a]))),
                         float(bool(_re.search(r"нормальн|работал", win))), float(bool(_re.search(r"переш|смен|подключ", win))),
                         float(bool(_re.search(r"начал|пропал|перестал|сломал|появ", win))), float(bool(_re.search(r"приед|выезд|перезвон|верн|пересч|сделаем|придет", win)))]
                cands.append({"type": t, "date": d, "dates": dset, "off": j - turn_i, "rel": rel,
                              "dist": abs(rel - vpos) if (vpos >= 0 and j == turn_i) else 1.5,
                              "after_value": float(vpos >= 0 and j == turn_i and rel > vpos),
                              "before_do": float("до " in before), "same_speaker": float(turns[j]["speaker"] == speaker),
                              "rank": rank, "future": float(d > cd), "past": float(d < cd), "local": local})
        base = {"off": 0, "rel": 0, "dist": 2, "after_value": 0, "before_do": 0, "same_speaker": 1, "rank": 0, "local": [0.0] * 11}
        cands.append({**base, "type": "callday", "date": cd, "dates": {cd}, "future": 0, "past": 0})
        cands.append({**base, "type": "hidden", "date": cd - timedelta(days=HIDDEN_DAYS), "future": 0, "past": 1})
        n_future = sum(c["future"] for c in cands)
        n_past = sum(c["past"] for c in cands)
        ctx = norm(turns[turn_i]["text"]) if 0 <= turn_i < len(turns) else ""
        common = [float(attribute == a) for a in ATTRIBUTES] + [float(norm(value) == v) for v in VALUE_FLAGS]
        common += [float(any(k in ctx for k in ("раньше", "до этого", "был", "прежн", "стар"))),
                   float(any(k in ctx for k in ("сейчас", "теперь", "уже"))), float(speaker == "operator"), n_in_turn, n_future, n_past,
                   float(any(k in ctx for k in ("переш", "смен", "поменя")))]
        X = []
        for c in cands:
            X.append([float(c["type"] == t) for t in TYPES] +
                     [c["off"], c["rel"], c["dist"], c["after_value"], c["before_do"], c["same_speaker"], c["rank"],
                      c["future"], c["past"]] + c["local"] + common)
        return cands, np.asarray(X, dtype=np.float32)

    # ---------- обучение на эталонных фактах ----------
    def fit(self, calls):
        for field in ("valid_from", "valid_to"):
            Xs, ys = [], []
            for call in calls:
                cd = call_day(call)
                for f in call["facts"]:
                    if field == "valid_to" and f["attribute"] not in ("service_state", "tariff", "ticket_status"):
                        continue
                    ti = f["source"]["turn_idx"]
                    if ti is None or ti >= len(call["turns"]):
                        continue
                    cands, X = self.candidates(call, ti, f["attribute"], f["value"], f["speaker"])
                    if field == "valid_to":  # кандидат «нет окончания»
                        X = np.vstack([X, np.zeros((1, X.shape[1]), np.float32)])
                        cands = cands + [{"type": "none", "date": None}]
                    gold = f[field]
                    gd = date.fromisoformat(gold[:10]) if gold else None
                    y = np.zeros(len(cands), np.int8)
                    hit = False
                    for k, c in enumerate(cands):
                        if c["type"] in ("hidden", "none"):
                            continue
                        if gd is not None and gd in c["dates"]:
                            y[k] = 1
                            hit = True
                    if not hit:
                        for k, c in enumerate(cands):
                            if (gd is None and c["type"] == "none") or (gd is not None and c["type"] == "hidden" and gd < cd):
                                y[k] = 1
                    Xs.append(X)
                    ys.append(y)
            X, y = np.vstack(Xs), np.concatenate(ys)
            m = HistGradientBoostingClassifier(max_iter=self.iters, max_leaf_nodes=31, learning_rate=0.08,
                                               l2_regularization=1.0, random_state=self.seed)
            m.fit(X, y)
            self.models[field] = m
        return self

    def choose(self, call, turn_i, attribute, value, speaker, field="valid_from"):
        cands, X = self.candidates(call, turn_i, attribute, value, speaker)
        if field == "valid_to":
            X = np.vstack([X, np.zeros((1, X.shape[1]), np.float32)])
            cands = cands + [{"type": "none", "date": None}]
        p = self.models[field].predict_proba(X)[:, 1]
        k = int(np.argmax(p))
        return cands[k], float(p[k])


def iso(call, cand, with_time=True):
    """Кандидат → ISO-строка. Для даты звонка — точное время звонка."""
    if cand is None or cand.get("date") is None:
        return None
    if cand["type"] == "callday":
        return call["started_at"]
    return datetime.combine(cand["date"], datetime.min.time()).isoformat()
