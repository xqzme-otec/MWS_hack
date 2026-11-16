"""CompactFact v2: поиск кандидатов (символьные n-граммы) → ранжирование (бустинг) → сборка фактов.

Разделение на «дорогие» гены (словарь, банк значений, ранжировщик — требуют переобучения)
и «дешёвые» гены (пороги, правила ловушек, режимы дат/цепочек — применяются к уже
посчитанным вероятностям). Эволюция гоняет дешёвые гены параллельно и быстро.
"""
from __future__ import annotations

import copy
import time
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .dates import DateSelector, iso
from .objects import ObjectChooser, ObjectResolver
from .state import close_chains
from .text import (ATTRIBUTES, CATALOG_KEYS, CONFIRM, REJECT, RX, SINGLE_VALUED, flags, inference_view, norm,
                   split_of, stem, turn_rows, words)

EXPENSIVE = {"ngram_lo": 2, "ngram_hi": 5, "max_features": 30000, "min_count": 2, "ctx_w": 0.3, "proto_mix": 0.8,
             "topk": 6, "gbm_iter": 150, "gbm_leaves": 15, "gbm_lr": 0.08, "pos_window": 0, "crossfit": 1}
CHEAP = {"thr_service_state": 0.35, "thr_tariff": 0.35, "thr_ticket_status": 0.4, "thr_action": 0.35,
         "thr_agreement": 0.3, "max_per_turn": 2, "trap_offer": 1, "trap_question": 1, "trap_imperative": 1,
         "chain_mode": 0, "date_conf_agreement": 0.0, "date_conf_state": 0.0, "obj_back": 6,
         "one_open_per_object": 0, "drop_dup_window": 0, "obj_learned": 1, "drop_hidden_closed": 0,
         "current_sel": 1, "cur_mode": 0, "cur_min_p": 0.1,
         "gl_on": 0, "gl_thr_service_state": 0.4, "gl_thr_tariff": 0.4, "gl_thr_ticket_status": 0.4,
         "gl_thr_action": 0.4, "gl_thr_agreement": 0.4, "gl_w_boost": 0.15, "gl_w_new": 0.5, "gl_min_cov": 0.5}


def hash_half(c):
    import hashlib
    return int(hashlib.md5(("half:" + c["call_id"]).encode()).hexdigest(), 16) % 2 == 0


def matcher_obj(f, name):
    from .matching import object_match
    return object_match(f, name)


def _cov(value_stems, text_stems):
    if not value_stems:
        return 0.0
    return sum(1 for s in value_stems if s in text_stems) / len(value_stems)


class Extractor:
    def __init__(self, catalog, genes=None, seed=42):
        self.catalog = catalog
        self.g = {**EXPENSIVE, **CHEAP, **(genes or {})}
        self.seed = seed

    # ------------------------------------------------------------------ fit
    def fit(self, calls, matcher, log=print, extra=None):
        """extra — синтетические/дистиллированные звонки: идут только в fit-часть
        (прототипы, банк значений, TF-IDF); ранжировщик калибруется на реальных rank-звонках."""
        t0 = time.time()
        g = self.g
        parts = defaultdict(list)
        for c in calls:
            parts[split_of(c, self.seed)].append(c)
        fit, rank = parts["fit"] + list(extra or []), parts["rank"]
        self.dates = DateSelector(seed=self.seed).fit(fit + rank)
        self.objects = ObjectResolver(self.catalog, fit)
        self.chooser = ObjectChooser(self.objects, self.seed).fit(
            fit[:1500], lambda f, n: matcher_obj(f, n))
        log(f"  dates+objects {time.time() - t0:.1f}s")
        # банк значений: каталог секций атрибута + частые значения train
        self.banks = {}
        for a in ATTRIBUTES:
            cnt = Counter(f["value"] for c in fit for f in c["facts"] if f["attribute"] == a)
            vals = [v for k in CATALOG_KEYS[a] for v in self.catalog.get(k, [])]
            vals += [v for v, k in cnt.items() if k >= g["min_count"]]
            uniq = {}
            for v in vals:
                uniq.setdefault(norm(v), v)
            values = sorted(uniq.values(), key=norm)
            self.banks[a] = {"values": values, "index": {norm(v): i for i, v in enumerate(values)},
                             "counts": np.array([cnt.get(v, 0) for v in values], np.float32),
                             "stems": [[stem(w) for w in words(v) if len(w) >= 3 and w != "не"] for v in values]}
        fit_rows = [(c, r) for c in fit for r in turn_rows(inference_view(c))]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(g["ngram_lo"], g["ngram_hi"]),
                                   max_features=g["max_features"], min_df=2, sublinear_tf=True, dtype=np.float32,
                                   preprocessor=norm)
        self.vec.fit([r["text"] for _, r in fit_rows] + [v for b in self.banks.values() for v in b["values"]])
        def build_protos(src_calls):
            rows = [(c, r) for c in src_calls for r in turn_rows(inference_view(c))]
            X = self._row_matrix([r for _, r in rows])
            row_id = {(c["call_id"], r["turn_idx"]): k for k, (c, r) in enumerate(rows)}
            out = {}
            for a, bank in self.banks.items():
                ri, ci = [], []
                for c in src_calls:
                    for f in c["facts"]:
                        if f["attribute"] != a:
                            continue
                        j = bank["index"].get(norm(f["value"]))
                        k = row_id.get((c["call_id"], f["source"]["turn_idx"]))
                        if j is not None and k is not None:
                            ri.append(j)
                            ci.append(k)
                A = sparse.csr_matrix((np.ones(len(ri), np.float32), (ri, ci)), shape=(len(bank["values"]), X.shape[0]))
                proto = normalize(A @ X)
                out[a] = normalize(g["proto_mix"] * proto + (1 - g["proto_mix"]) * bank["lex"]).astype(np.float32)
            return out

        for a, bank in self.banks.items():
            bank["lex"] = normalize(self.vec.transform(bank["values"]))
        feats = {a: [] for a in ATTRIBUTES}
        ys = {a: [] for a in ATTRIBUTES}

        def collect(target_calls):
            for c in target_calls:
                cand = self._candidates(inference_view(c))
                gold_by_row = defaultdict(list)
                for f in c["facts"]:
                    for d in range(-g["pos_window"], g["pos_window"] + 1):
                        gold_by_row[(f["source"]["turn_idx"] or 0) + d].append(f)
                for a, rows_, vals_, F in cand:
                    for (ti, vj), x in zip(zip(rows_, vals_), F):
                        v = self.banks[a]["values"][vj]
                        feats[a].append(x)
                        ys[a].append(any(f["attribute"] == a and matcher(f["value"], v, a) for f in gold_by_row.get(ti, [])))

        if int(g.get("crossfit", 0)):
            # перекрёстное обучение: прототипы с одной половины fit, признаки — на другой (+ rank)
            h1 = [c for c in fit if hash_half(c)]
            h2 = [c for c in fit if not hash_half(c)]
            for src_calls, tgt in ((h1, h2 + rank), (h2, h1)):
                P = build_protos(src_calls)
                for a, bank in self.banks.items():
                    bank["proto"] = P[a]
                collect(tgt)
                log(f"  crossfit fold done: {sum(len(v) for v in ys.values())} pairs, {time.time() - t0:.0f}s")
            P = build_protos(fit)
            for a, bank in self.banks.items():
                bank["proto"] = P[a]
        else:
            P = build_protos(fit)
            for a, bank in self.banks.items():
                bank["proto"] = P[a]
            collect(rank)
        log(f"  bank+vectors {time.time() - t0:.1f}s; sizes " + ", ".join(f"{a}:{len(b['values'])}" for a, b in self.banks.items()))
        self.rankers = {}
        for a in ATTRIBUTES:
            Xa, ya = np.asarray(feats[a], np.float32), np.asarray(ys[a], np.int8)
            m = HistGradientBoostingClassifier(max_iter=g["gbm_iter"], max_leaf_nodes=g["gbm_leaves"],
                                               learning_rate=g["gbm_lr"], l2_regularization=1.0,
                                               min_samples_leaf=20, random_state=self.seed)
            m.fit(Xa, ya)
            self.rankers[a] = m
            log(f"  ranker {a}: {len(ya)} pairs, {int(ya.sum())} positive")
        self.fit_seconds = time.time() - t0
        return self

    def _row_matrix(self, rows):
        Xt = self.vec.transform([r["text"] for r in rows])
        Xc = self.vec.transform([r["prev"] + " " + r["text"] + " " + r["next"] for r in rows])
        return normalize((1 - self.g["ctx_w"]) * normalize(Xt) + self.g["ctx_w"] * normalize(Xc)).astype(np.float32)

    def _candidates(self, call):
        """[(attribute, row_ids, value_ids, features)] — кандидаты «реплика × значение»."""
        rows = turn_rows(call)
        if not rows:
            return []
        g = self.g
        Xt = normalize(self.vec.transform([r["text"] for r in rows]))
        Xr = self._row_matrix(rows)
        fl = np.stack([flags(r) for r in rows])
        tstems = [set(stem(w) for w in words(r["text"])) for r in rows]
        cstems = [set(stem(w) for w in words(r["prev"] + " " + r["text"] + " " + r["next"])) for r in rows]
        out = []
        for a, bank in self.banks.items():
            n = len(bank["values"])
            k = min(g["topk"], n)
            S = [(Xr @ bank["proto"].T).toarray(), (Xt @ bank["proto"].T).toarray(),
                 (Xt @ bank["lex"].T).toarray(), (Xr @ bank["lex"].T).toarray()]
            pairs = set()
            for M in S:
                top = np.argpartition(-M, k - 1, axis=1)[:, :k]
                for i in range(len(rows)):
                    for j in top[i]:
                        pairs.add((i, int(j)))
            mention = set()
            if a == "tariff":
                for i, r in enumerate(rows):
                    for n, cov, hit in self.objects.tariffs.find(r["text"])[:2]:
                        j = bank["index"].get(norm(n))
                        if j is not None:
                            pairs.add((i, j))
                            mention.add((i, j))
            pairs = sorted(pairs)
            ri = np.array([p[0] for p in pairs])
            vj = np.array([p[1] for p in pairs])
            rowmax = S[0].max(axis=1)
            logc = np.log1p(bank["counts"])
            F = []
            for i, j in pairs:
                vs = bank["stems"][j]
                F.append([S[0][i, j], S[1][i, j], S[2][i, j], S[3][i, j], S[0][i, j] - rowmax[i],
                          logc[j], len(vs), _cov(vs, tstems[i]), _cov(vs, cstems[i]),
                          float("не помог" in bank["values"][j]), float(bank["values"][j] == "работает нормально"),
                          float(bank["values"][j] == "больше не используется"), float((i, j) in mention)] + list(fl[i]))
            out.append((a, [rows[i]["turn_idx"] for i in ri], list(vj), np.asarray(F, np.float32)))
        return out

    # ------------------------------------------------------------- scoring
    def score(self, call):
        """Дорогая часть инференса: кандидаты с вероятностями (кэшируется для эволюции)."""
        view = inference_view(call)
        cands = []
        for a, rows_, vals_, F in self._candidates(view):
            if not len(F):
                continue
            p = self.rankers[a].predict_proba(F)[:, 1]
            for ti, vj, pp, x in zip(rows_, vals_, p, F):
                if pp >= 0.05:
                    cands.append((a, int(ti), self.banks[a]["values"][vj], float(pp)))
        return {"view": view, "cands": cands, "obj": self.objects.prepare(view),
                "spk": [t["speaker"] for t in view["turns"]]}

    # ------------------------------------------------------------ assembly
    def assemble(self, scored, genes=None, date_cache=None):
        g = {**self.g, **(genes or {})}
        view, call_rows = scored["view"], turn_rows(scored["view"])
        by_idx = {r["turn_idx"]: r for r in call_rows}
        date_cache = {} if date_cache is None else date_cache
        per_row = defaultdict(list)
        cands = scored["cands"]
        if scored.get("gc_spans") is not None and int(g.get("gc_on", 0)):
            from .gliner_core import core_cands          # GLiNER — основной источник фактов
            cands = core_cands(self, scored, g)
            if int(g.get("gc_hyb", 0)) and scored.get("cands"):
                # гибрид: бустинг добавляет кандидатов там, где GLiNER промолчал (полнота); вес — ген эволюции
                w, mode = float(g.get("gc_hyb_w", 0.5)), int(g.get("gc_hyb_mode", 0))
                merged = {(a, ti, v): p for a, ti, v, p in cands}
                for a, ti, v, p in scored["cands"]:
                    q = p * w
                    k = (a, ti, v)
                    if k in merged:
                        merged[k] = max(merged[k], q) if mode == 0 else 1 - (1 - merged[k]) * (1 - min(q, 0.999))
                    elif q >= float(g.get("gc_hyb_min", 0.2)):
                        merged[k] = q
                cands = [(a, ti, v, p) for (a, ti, v), p in merged.items()]
        elif scored.get("gl_spans") is not None and int(g.get("gl_on", 0)):
            from .gliner_fusion import fuse
            cands = fuse(self, scored, scored["gl_spans"], g)
        for a, ti, v, p in cands:
            if p < g[f"thr_{a}"]:
                continue
            r = by_idx[ti]
            if not self._trap_ok(a, v, r, g):
                continue
            per_row[(ti, a)].append((p, v))
        best = {}
        for (ti, a), lst in per_row.items():
            lst.sort(reverse=True)
            for p, v in lst[: int(g["max_per_turn"])]:
                r = by_idx[ti]
                if a == "service_state" and int(g["obj_learned"]):
                    obj = {"type": "service", "name": self.chooser.choose(scored["obj"], r["i"], scored["spk"])}
                else:
                    obj = self.objects.resolve(view, scored["obj"], r["i"], a, back=int(g["obj_back"]))
                key = (obj["name"], a, norm(v))
                if key not in best or p > best[key][0]:
                    best[key] = (p, a, v, r, obj)
        facts = []
        for p, a, v, r, obj in best.values():
            k = (r["i"], a, v)
            if k not in date_cache:
                cf, pf = self.dates.choose(view, r["i"], a, v, r["speaker"], "valid_from")
                ct, pt = (self.dates.choose(view, r["i"], a, v, r["speaker"], "valid_to")
                          if a in SINGLE_VALUED else (None, 1.0))
                date_cache[k] = (iso(view, cf), pf, iso(view, ct) if ct else None, cf["type"])
            vf, pf, vt, vtype = date_cache[k]
            if int(g["drop_hidden_closed"]) and vtype == "hidden":
                continue
            if a == "agreement" and pf < g["date_conf_agreement"]:
                continue
            if a in SINGLE_VALUED and pf < g["date_conf_state"]:
                continue
            facts.append({"object": dict(obj), "attribute": a, "value": v,
                          "polarity": -1 if v == "больше не используется" else 1, "speaker": r["speaker"],
                          "valid_from": vf, "valid_to": vt, "_vt_model": vt,
                          "source": {"turn_idx": r["turn_idx"], "span": None, "quote": None}, "confidence": round(p, 4)})
        if int(g["chain_mode"]) == 1:
            close_chains(facts, mode="chain_first")
        elif int(g["chain_mode"]) == 2:
            close_chains(facts, mode="selector_first")
        if int(g.get("current_sel", 0)) and getattr(self, "current", None) is not None:
            facts = self.current.apply(view, facts, mode=int(g.get("cur_mode", 0)), min_p=float(g.get("cur_min_p", 0.0)))
        elif int(g["one_open_per_object"]):
            self._one_open(facts)
        for f in facts:
            f.pop("_vt_model", None)
            if f["valid_to"] and f["valid_from"] and f["valid_to"][:10] < f["valid_from"][:10]:
                f["valid_to"] = None
        facts.sort(key=lambda f: (f["source"]["turn_idx"], f["attribute"]))
        out = copy.deepcopy(view)
        out["facts"] = facts
        return out

    def _trap_ok(self, a, v, r, g):
        """Ловушки схемы §4: неподтверждённое предложение, отвергнутая догадка, вопрос, инструкция."""
        if r["speaker"] != "operator":
            return True
        nxt = norm(r["next"])
        client_next = r["next_speaker"] == "client"
        rejected = client_next and bool(REJECT.search(nxt))
        confirmed = client_next and not rejected and bool(CONFIRM.search(nxt))
        text = norm(r["text"])
        if g["trap_offer"] and a == "agreement" and rejected:
            return False
        if g["trap_offer"] and a in ("tariff", "service_state") and RX["offer"].search(text) and not confirmed:
            return False
        if g["trap_question"] and a in ("tariff", "service_state") and "?" in r["text"] and rejected:
            return False
        if g["trap_imperative"] and a == "action" and RX["imper"].search(text):
            return False
        return True

    @staticmethod
    def _one_open(facts):
        """Одно актуальное значение на (объект, атрибут): более раннее закрывается более поздним."""
        groups = defaultdict(list)
        for f in facts:
            if f["attribute"] in SINGLE_VALUED and f["polarity"] == 1:
                groups[(norm(f["object"]["name"]), f["attribute"])].append(f)
        for g in groups.values():
            open_ = [f for f in g if f["valid_to"] is None]
            if len(open_) <= 1:
                continue
            open_.sort(key=lambda f: ((f["valid_from"] or "")[:10], f["source"]["turn_idx"]))
            last = open_[-1]
            for f in open_[:-1]:
                if norm(f["value"]) != norm(last["value"]):
                    f["valid_to"] = max(last["valid_from"] or "", f["valid_from"] or "") or None  # конец не раньше начала

    def predict(self, call, genes=None):
        return self.assemble(self.score(call), genes)
