#!/usr/bin/env python3
"""Метрики кейса CompactFactExtraction: сравнение предсказаний экстрактора с эталоном.

Самодостаточен (stdlib) — этот скрипт отдаётся участникам вместе с val-срезом, и им же
жюри считает итог на скрытом test-срезе. Форматы входов — вывод scripts/export_case.py.

Метрики (по кейсу):
  fact_precision / fact_recall — совпадение фактов: объект (fuzzy по имени) + атрибут +
      значение (fuzzy ≥ VALUE_SIM) + полярность; жадное взаимно-однозначное сопоставление.
  temporal_recall / temporal_precision — среди сопоставленных: доля с верной датой
      начала (и окончания, если задано) — совпадение по ДНЮ (времена суток в золоте
      присутствуют лишь когда выражение времени озвучено).
  state_accuracy — актуальное значение каждого одно-значного атрибута (state/tariff/
      ticket_status) после всех изменений: последнее по valid_from значение верно.
  grounding — среди сопоставленных фактов: верная реплика-источник (turn_idx) и,
      где у обеих сторон есть спаны, пересечение символьных интервалов.

Запуск: python case_metrics.py gold.jsonl pred.jsonl [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from difflib import SequenceMatcher

VALUE_SIM = 0.85          # порог нечёткого совпадения значения факта
NAME_SIM = 0.80           # порог нечёткого совпадения имени объекта
SINGLE_VALUED = {"service_state", "tariff", "ticket_status"}

_PUNCT = ".,!?;:\"«»()—-"
WORD_RE = re.compile(r"[а-яёa-z0-9]+")


def _tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower().replace("ё", "е"))


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _value_sim(a: str, b: str) -> float:
    """Похожесть значений: точная строка → пословенно-жадная (порядок слов важен)."""
    if a.strip().lower() == b.strip().lower():
        return 1.0
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return _sim(a.lower(), b.lower())
    if ta == tb:
        return 1.0
    # жадное выравнивание слов в порядке следования
    j, hits = 0, 0
    for w in ta:
        best, bj = 0.0, None
        for k in range(j, len(tb)):
            s = _sim(w, tb[k])
            if s > best:
                best, bj = s, k
        if best >= 0.75:
            hits += 1
            j = (bj or j) + 1
    return hits / max(len(ta), len(tb))


def _day(iso: str | None) -> str | None:
    return iso[:10] if iso else None


def match_facts(gold: list[dict], pred: list[dict]) -> list[tuple[dict, dict, float]]:
    """Жадкое взаимно-однозначное сопоставление фактов (лучшие пары первыми)."""
    pairs = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if g["attribute"] != p.get("attribute"):
                continue
            if (g.get("polarity", 1) > 0) != (p.get("polarity", 1) > 0):
                continue
            vsim = _value_sim(g["value"], p.get("value", ""))
            osim = _sim(_tokens(" ".join(g["object"]["name"]))[0] if _tokens(g["object"]["name"]) else "",
                        _tokens(" ".join(p.get("object", {}).get("name", "")))[0] if _tokens(p.get("object", {}).get("name", "")) else "") \
                if _tokens(g["object"]["name"]) and _tokens(p.get("object", {}).get("name", "")) else 1.0
            if vsim >= VALUE_SIM and (osim >= NAME_SIM or not osim):
                pairs.append((vsim * osim, gi, pi, g, p))
    pairs.sort(key=lambda x: -x[0])
    used_g, used_p, out = set(), set(), []
    for _score, gi, pi, g, p in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out.append((g, p, _score))
    return out


def evaluate(gold_calls: dict[str, dict], pred_calls: dict[str, dict]) -> dict:
    """Посчитать все метрики по корпусу (сопоставление звонок к звонку по call_id)."""
    n_gold = n_pred = 0
    matched = []
    state_total = state_ok = 0
    temporal_gold = temporal_ok = 0
    ground_total = ground_turn_ok = ground_span_total = ground_span_ok = 0
    for cid, g in gold_calls.items():
        gf = g["facts"]
        pf = pred_calls.get(cid, {}).get("facts", [])
        n_gold += len(gf)
        n_pred += len(pf)
        m = match_facts(gf, pf)
        matched.extend(m)
        # темпоральная полнота: факты с датой, попавшие в сопоставление
        for gfact, pfact, _s in m:
            temporal_gold += 1
            ok_from = _day(gfact.get("valid_from")) == _day(pfact.get("valid_from"))
            ok_to = _day(gfact.get("valid_to")) == _day(pfact.get("valid_to"))
            if ok_from and ok_to:
                temporal_ok += 1
            # grounding
            ground_total += 1
            gs, ps = gfact["source"], pfact.get("source") or {}
            if ps.get("turn_idx") == gs.get("turn_idx"):
                ground_turn_ok += 1
                if gs.get("span") and ps.get("span"):
                    ground_span_total += 1
                    lo = max(gs["span"][0], ps["span"][0])
                    hi = min(gs["span"][1], ps["span"][1])
                    if hi > lo:
                        ground_span_ok += 1
        # точность текущего состояния: последний по valid_from факт одно-значного атрибута
        by_key: dict[tuple, dict] = {}
        for f in gf:
            if f["attribute"] in SINGLE_VALUED and f.get("valid_to") is None:
                k = (f["object"]["id"], f["attribute"])
                if k not in by_key or f["valid_from"] > by_key[k]["valid_from"]:
                    by_key[k] = f
        for (oid, attr), gfact in by_key.items():
            state_total += 1
            cands = [p for p in pf if p.get("attribute") == attr and p.get("valid_to") is None]
            if cands:
                last = max(cands, key=lambda p: p.get("valid_from") or "")
                if _value_sim(gfact["value"], last.get("value", "")) >= VALUE_SIM:
                    state_ok += 1
    n_matched = len(matched)
    return {
        "n_calls": len(gold_calls),
        "n_gold_facts": n_gold,
        "n_pred_facts": n_pred,
        "fact_precision": round(n_matched / n_pred, 4) if n_pred else 0.0,
        "fact_recall": round(n_matched / n_gold, 4) if n_gold else 0.0,
        "temporal_recall": round(temporal_ok / temporal_gold, 4) if temporal_gold else None,
        "n_temporal": temporal_gold,
        "state_accuracy": round(state_ok / state_total, 4) if state_total else None,
        "n_state": state_total,
        "grounding_turn": round(ground_turn_ok / ground_total, 4) if ground_total else None,
        "grounding_span": round(ground_span_ok / ground_span_total, 4) if ground_span_total else None,
        "n_grounded": ground_total,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gold")
    ap.add_argument("pred")
    ap.add_argument("--json", help="дополнительно записать метрики в файл")
    a = ap.parse_args()

    gold_calls = {c["call_id"]: c for c in map(json.loads, open(a.gold, encoding="utf-8"))}
    pred_calls = {c["call_id"]: c for c in map(json.loads, open(a.pred, encoding="utf-8"))}
    m = evaluate(gold_calls, pred_calls)
    for k, v in m.items():
        print(f"{k:18} {v}")
    if a.json:
        json.dump(m, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
