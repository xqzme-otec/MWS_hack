"""Обучаемый выбор актуального значения одно-значного атрибута (честная замена правилу
«последний по дате»): для каждой группы (объект, атрибут) классификатор оценивает каждого
кандидата как «текущее значение»; остальные закрываются датой начала выбранного.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .text import SINGLE_VALUED, norm

NEW = re.compile(r"сейчас|теперь|уже|стоит|подключ|переш[её]л|перешл|сменил|поменял|на данный момент|актуальн|сегодня")
OLD = re.compile(r"раньше|до этого|был[аои]?\b|прежн|стар[ыо]|до\s+(вчера|понедел|вторн|сред|четвер|пятниц|суббот|воскрес)|ушл|закрыл")
VFLAGS = ["работает нормально", "больше не используется", "принята", "в работе", "назначен выезд специалиста", "выполнена"]
STATUS_ORDER = {"принята": 0, "в работе": 1, "назначен выезд специалиста": 2, "выполнена": 3}
ATTR = ["service_state", "tariff", "ticket_status"]


def group_key(f):
    return (norm(f["object"]["name"]), f["attribute"])


def features(call, group):
    cd = datetime.fromisoformat(call["started_at"]).date()
    n_turns = max(len(call["turns"]), 1)
    probs = sorted((f["confidence"] for f in group), reverse=True)
    last_turn = max(f["source"]["turn_idx"] for f in group)
    vals = defaultdict(int)
    for f in group:
        vals[norm(f["value"])] += 1
    X = []
    for f in group:
        ti = f["source"]["turn_idx"]
        text = norm(call["turns"][ti]["text"]) if 0 <= ti < len(call["turns"]) else ""
        try:
            d = (datetime.fromisoformat(f["valid_from"]).date() - cd).days if f["valid_from"] else -999
        except ValueError:
            d = -999
        v = norm(f["value"])
        X.append([f["confidence"], probs.index(f["confidence"]), len(group), float(d), float(d == 0), float(d < -300),
                  ti / n_turns, float(ti == last_turn), last_turn - ti, vals[v], float(f["speaker"] == "client"),
                  float(bool(NEW.search(text))), float(bool(OLD.search(text))), float("?" in call["turns"][ti]["text"]) if 0 <= ti < len(call["turns"]) else 0.0,
                  STATUS_ORDER.get(v, -1), len(v.split())] + [float(v == x) for x in VFLAGS] + [float(f["attribute"] == a) for a in ATTR])
    return np.asarray(X, np.float32)


class CurrentSelector:
    def __init__(self, seed=42):
        self.seed, self.m = seed, None

    def fit(self, pairs, value_match, object_match):
        """pairs: [(gold_call, predicted_facts)]; метка — значение совпадает с эталонным текущим."""
        Xs, ys = [], []
        for gold, facts in pairs:
            cur = {}
            for f in gold["facts"]:
                if f["attribute"] in SINGLE_VALUED and f.get("valid_to") is None:
                    k = (f["object"]["id"], f["attribute"])
                    if k not in cur or f["valid_from"] > cur[k]["valid_from"]:
                        cur[k] = f
            groups = defaultdict(list)
            for f in facts:
                if f["attribute"] in SINGLE_VALUED and f["polarity"] == 1:
                    groups[group_key(f)].append(f)
            for g in groups.values():
                y = []
                for f in g:
                    ok = any(gf["attribute"] == f["attribute"] and object_match(gf, f["object"]["name"])
                             and value_match(gf["value"], f["value"], f["attribute"]) for gf in cur.values())
                    y.append(int(ok))
                if len(g) >= 1:
                    Xs.append(features(gold, g))
                    ys.append(np.asarray(y, np.int8))
        X, y = np.vstack(Xs), np.concatenate(ys)
        self.m = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, learning_rate=0.06,
                                                l2_regularization=1.0, random_state=self.seed).fit(X, y)
        self.n_train = int(len(y))
        return self

    def apply(self, call, facts, mode=0, min_p=0.0):
        """mode 0: закрыть неактуальные датой начала выбранного; mode 1: удалить неактуальные с p<min_p."""
        groups = defaultdict(list)
        for f in facts:
            if f["attribute"] in SINGLE_VALUED and f["polarity"] == 1:
                groups[group_key(f)].append(f)
        drop = set()
        for g in groups.values():
            if len(g) < 2 and mode == 0:
                continue
            p = self.m.predict_proba(features(call, g))[:, 1]
            k = int(np.argmax(p))
            chosen = g[k]
            chosen["valid_to"] = None
            for i, f in enumerate(g):
                if i == k:
                    continue
                if norm(f["value"]) == norm(chosen["value"]):
                    drop.add(id(f))
                    continue
                if mode == 1 and p[i] < min_p:
                    drop.add(id(f))
                    continue
                if f["valid_to"] is None:
                    vt = chosen["valid_from"]
                    f["valid_to"] = vt if (vt and f["valid_from"] and vt[:10] >= f["valid_from"][:10]) else (vt or None)
        return [f for f in facts if id(f) not in drop]
