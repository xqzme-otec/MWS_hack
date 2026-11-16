"""Финальный конвейер CompactFact: GLiNER (основная модель) + бустинг-кандидаты (гибрид) → сборка фактов.

    звонок ─► GLiNER g13-ru (спаны 5 типов фактов в КАЖДОЙ реплике, + объекты и даты)
          └─► бустинг-ранжировщик (кандидаты «реплика × значение каталога»)
                    ▼  слияние noisy-OR с весом gc_hyb_w (гены эволюции)
          ловушки §4 → объект → даты (относительные → абсолютные) → текущее значение
                    ▼
          факты в схеме кейса + source.turn_idx + source.span (символьные оффсеты спана GLiNER)

Один объект на процесс: модели грузятся один раз. Используется API, веб-интерфейсом, оценкой и замерами.
"""
from __future__ import annotations

import difflib
import json
import os
import pickle
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _p(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def ensure_weights(model_dir):
    """Веса в репозитории лежат частями < 50 МБ (лимит GitHub); склеиваем при первом запуске."""
    target = os.path.join(model_dir, "pytorch_model.bin")
    parts = sorted(f for f in os.listdir(model_dir) if f.startswith("pytorch_model.bin.part"))
    if not os.path.exists(target) and parts:
        tmp = target + ".tmp"
        with open(tmp, "wb") as fo:
            for f in parts:
                with open(os.path.join(model_dir, f), "rb") as fi:
                    fo.write(fi.read())
        md5f = os.path.join(model_dir, "pytorch_model.bin.md5")
        if os.path.exists(md5f):
            import hashlib
            h = hashlib.md5(open(tmp, "rb").read()).hexdigest()
            if h != open(md5f).read().strip():
                raise RuntimeError(f"контрольная сумма весов не совпала: {h}")
        os.replace(tmp, target)
    return target


class Pipeline:
    def __init__(self, gliner_dir="models/g13_ru", boost_pkl="models/compactfact_v5.pkl",
                 genes="models/genes_gliner_hybrid.json", threads=None, mode="hybrid",
                 boost_genes="models/genes_best.json"):
        """mode: hybrid (GLiNER + бустинг, по умолчанию) | gliner (только GLiNER) | boosting (только бустинг)."""
        import torch
        from .objects import refresh
        torch.set_num_threads(int(threads or os.environ.get("CFX_THREADS", "2")))
        self.mode = mode
        self.ex = pickle.load(open(_p(boost_pkl), "rb")); refresh(self.ex.objects)
        self.genes = json.load(open(_p(genes))) if isinstance(genes, str) else dict(genes)
        self.boost_genes = json.load(open(_p(boost_genes))) if os.path.exists(_p(boost_genes)) else {}
        self.gliner_dir = _p(gliner_dir)
        self.glc = None
        if mode != "boosting":
            from gliner import GLiNER
            from gliner.data_processing.tokenizer import WordsSplitter
            from .gliner_fusion import GlinerSpans
            ensure_weights(self.gliner_dir)
            m = GLiNER.from_pretrained(self.gliner_dir).eval().float()
            m.data_processor.words_splitter = WordsSplitter("whitespace")
            arch_p = os.path.join(self.gliner_dir, "arch.json")
            self.arch = json.load(open(arch_p)) if os.path.exists(arch_p) else {"ctx": 0}
            self.glc = GlinerSpans(m, context_words=self.arch.get("ctx", 0), raw_threshold=0.1, batch=16)
            self.model = m

    # ---------------------------------------------------------------- инференс
    def genes_for(self, mode):
        if mode == "gliner":
            return {**self.genes, "gc_hyb": 0}
        if mode == "boosting":
            return self.boost_genes
        return self.genes

    def extract(self, call, with_debug=False, mode=None):
        mode = mode or self.mode
        if self.glc is None:
            mode = "boosting"
        genes = self.genes_for(mode)
        import torch
        from .gliner_core import prepare as gc_prepare, selected_turns
        from .text import inference_view
        from .schema import validate_call
        t0 = time.perf_counter()
        c = {k: call[k] for k in ("call_id", "started_at", "turns") if k in call}
        c["turns"] = [{"idx": t.get("idx", i), "speaker": t["speaker"], "text": t["text"]} for i, t in enumerate(c["turns"])]
        timing = {}
        if mode == "boosting":
            s = self.ex.score(c)
            out = self.ex.assemble(s, genes)
            spans, n_run = [], 0
            timing["boosting_ms"] = (time.perf_counter() - t0) * 1000
        else:
            v = inference_view(c)
            pre = gc_prepare(self.ex, v)
            sel = selected_turns(pre, {**{"gc_pref_words": 0, "gc_pref_sim": 0.0}, **genes})
            sub = {**c, "turns": [t for t in c["turns"] if t["idx"] in sel]}
            t1 = time.perf_counter()
            with torch.no_grad():
                spans = self.glc.spans(sub) if sub["turns"] else []
            timing["gliner_ms"] = (time.perf_counter() - t1) * 1000
            n_run = len(sub["turns"])
            t2 = time.perf_counter()
            boost = self.ex.score(c)["cands"] if int(genes.get("gc_hyb", 0)) else []
            timing["boosting_ms"] = (time.perf_counter() - t2) * 1000
            scored = {"view": v, "cands": boost, "obj": self.ex.objects.prepare(v),
                      "spk": [t["speaker"] for t in v["turns"]], "gc_pre": pre,
                      "gc_spans": {genes.get("gc_cfg", "gl"): spans}}
            out = self.ex.assemble(scored, {**genes, "gc_on": 1}, {})
        self._attach_spans(c, out, spans)
        for f in out["facts"]:   # интервал не может закончиться раньше начала (схема)
            if f.get("valid_to") and f.get("valid_from") and f["valid_to"] < f["valid_from"]:
                f["valid_to"] = f["valid_from"]
        out["turns"] = c["turns"]
        errs = validate_call(out)
        timing["total_ms"] = (time.perf_counter() - t0) * 1000
        out["meta"] = {"engine": mode, "latency_ms": round(timing["total_ms"], 1),
                       "timing_ms": {k: round(v, 1) for k, v in timing.items()},
                       "turns_total": len(c["turns"]), "turns_gliner": n_run, "schema_errors": errs}
        if with_debug:
            out["meta"]["gliner_spans"] = [
                {"turn_idx": ti, "label": a, "text": tx, "score": round(sc, 3),
                 "span": self.glc.offsets.get((ti, a, tx)) if self.glc else None}
                for ti, a, tx, sc in spans]
        return out

    def _attach_spans(self, call, out, spans):
        text_of = {t["idx"]: t["text"] for t in call["turns"]}
        offs = self.glc.offsets if self.glc else {}
        for f in out["facts"]:
            src = f.setdefault("source", {})
            ti = src.get("turn_idx")
            text = text_of.get(ti, "")
            cand = [(tx, sc) for (t, a, tx, sc) in spans if t == ti and a == f["attribute"]]
            cand.sort(key=lambda x: -difflib.SequenceMatcher(None, x[0].lower(), f["value"].lower()).ratio())
            span = None
            for tx, _sc in cand:
                span = offs.get((ti, f["attribute"], tx)) or _find(text, tx)
                if span:
                    break
            src["span"] = span or _find(text, f["value"])
            src["quote"] = text[src["span"][0]:src["span"][1]] if src["span"] else None


def _find(text, s):
    s = (s or "").strip(" ,.")
    if not s:
        return None
    i = text.lower().find(s.lower())
    return [i, i + len(s)] if i >= 0 else None
