#!/usr/bin/env python3
"""Метрики кейса CompactFactExtraction v2: сравнение предсказаний экстрактора с эталоном.

Самодостаточен (stdlib, Python 3.10+). Этим скриптом участники считают себя на val,
им же жюри считает скрытый test — версия для обеих сторон одна (см. VERSION).

v2 (14.11.2026), по результатам разбора вопросов участников:
  1. state_accuracy сравнивает состояние в привязке к ОБЪЕКТУ (fuzzy по имени);
     раньше предсказание отбиралось только по атрибуту и могло «перекрыть» чужой объект.
  2. Имя объекта сравнивается по полному тексту (короткие имена — по порогу NAME_SIM);
     убран артефакт посимвольного сравнения и лазейка нулевого сходства.
  3. _value_sim: совпадение по покрытию МЕНЬШЕЙ стороны (вложение каноничного значения
     в более длинную формулировку — совпадение, как и нормализация к каталогу).
  4. temporal_recall считается только для фактов, чья дата ВОССТАНАВЛИВАЕТСЯ из
     транскрипта (относительные выражения, ограниченные банком ниже), и не касается
     attribute="action": кейс определяет темпоральность для изменений состояния и
     договорённостей; дата действия без озвученного срока непроверяема.

Запуск: python case_metrics.py gold.jsonl pred.jsonl [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

VERSION = "2.1"
VALUE_SIM = 0.85          # порог нечёткого совпадения значения (короткие, ≤2 слов)
VALUE_SIM_LONG = 0.70     # порог для длинных значений (≥3 слов, покрытие меньшей стороны)
NAME_SIM = 0.80           # порог совпадения имени объекта
SINGLE_VALUED = {"service_state", "tariff", "ticket_status"}
TEMPORAL_ATTRS = {"service_state", "tariff", "ticket_status", "agreement"}

_WORD_RE = re.compile(r"[а-яёa-z0-9]+")
_TRANSLIT = {"wifi": "вай фай", "вайфай": "вай фай", "wi": "вай", "fi": "фай", "tv": "тв", "lan": "лан",
             "gpon": "гпон", "ont": "онт", "ip": "ип", "sim": "сим", "sms": "смс"}


def _norm_word(w: str) -> str:
    w = w.lower().replace("ё", "е")
    return _TRANSLIT.get(w, w)


def _tokens(text: str) -> list[str]:
    out: list[str] = []
    for w in _WORD_RE.findall(text.lower()):
        nw = _norm_word(w)
        out.extend(nw.split(" "))          # 'wifi' → 'вай фай' (два токена, как «вай-фай»)
    return out


def _stem(w: str) -> str:
    return w[:-2] if len(w) >= 7 else w[:-1] if len(w) >= 5 else w


def _words_close(a: str, b: str) -> bool:
    if a == b or _stem(a) == _stem(b):
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.75


def _sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return 1.0 if a == b else SequenceMatcher(None, a, b).ratio()


def _value_sim(a: str, b: str) -> float:
    """Похожесть значений: пословечно, морфологически-терпимо, знаменатель — меньшая
    сторона. Вложение «Орбита-Люкс» в «тариф Орбита-Люкс» — полное совпадение: более
    короткая сторона покрыта целиком."""
    if a.strip().lower() == b.strip().lower():
        return 1.0
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return _sim(a.lower(), b.lower())
    if ta == tb:
        return 1.0
    j, hits = 0, 0
    for w in ta:
        if w == "не":
            continue                          # частицу не считаем ни попаданием, ни длиной
        for k in range(j, len(tb)):
            if _words_close(w, tb[k]):
                hits += 1
                j = k + 1
                break
    na, nb = [w for w in ta if w != "не"], [w for w in tb if w != "не"]
    neg_a, neg_b = "не" in ta, "не" in tb
    if neg_a != neg_b:                        # «работает» против «не работает» — не совпадение
        return 0.0
    denom = min(len(na), len(nb))
    if denom == 1 and max(len(na), len(nb)) >= 3:
        # одно слово внутри длинной фразы засчитывается только при ТОЧНОМ/стем-совпадении:
        # «не работает» ⊂ «мобильный интернет не работает» — да; «Смарт»~«смартфон» — нет
        js, hs = 0, 0
        for w in na:
            for k in range(js, len(nb)):
                if w == nb[k] or _stem(w) == _stem(nb[k]):
                    hs += 1
                    js = k + 1
                    break
        return hs / max(1, denom)
    return hits / max(1, denom)


def _object_sim(g: dict, p: dict) -> float:
    """Сходство объектов. gold-объект типа client (субъект тарифа/действия) пропускается:
    экстрактор вправе назвать связанную сущность. Иначе — полное имя, короткие строго."""
    if g["object"].get("type") == "client":
        return 1.0
    # без типа в gold считаем объект значимым (default-client пропускал чужие объекты)
    gt_raw = (g["object"].get("name") or "").strip()
    pt_raw = ((p.get("object") or {}).get("name") or "").strip()
    if not gt_raw:
        return 1.0                            # gold-объект не назван — не проверяем
    if not pt_raw:
        return 0.0
    if ":" in gt_raw:                         # сырой id — непрозрачная строка: только точное
        return 1.0 if gt_raw == pt_raw else 0.0
    gt, pt = _tokens(gt_raw), _tokens(pt_raw)
    v = _value_sim(" ".join(gt), " ".join(pt))
    need = NAME_SIM if min(len(gt), len(pt)) <= 2 else VALUE_SIM_LONG
    return v if v >= need else 0.0


# ── каталог нормализации ─────────────────────────────────────────────────────
_CATALOG: dict[str, list[str]] | None = None
_ATTR_SECTIONS = {"tariff": ("tariffs",), "service_state": ("symptoms", "services"),
                  "ticket_status": ("statuses",), "agreement": ("agreements",),
                  "action": ("actions",)}


def load_catalog(path_hint: str) -> None:
    """Каталог ищется рядом с gold-файлом (catalog.json, как в пакете); снап выполняется
    ТОЛЬКО внутри секций атрибута — плоский список коллапсировал разные сущности
    («Безлимит»-тариф ↔ «безлимит на соцсети»-услуга)."""
    global _CATALOG, _canon
    p = Path(path_hint)
    for cand in (p.parent / "catalog.json", Path("catalog.json")):
        if cand.exists():
            _CATALOG = json.loads(cand.read_text(encoding="utf-8"))
            break
    else:
        _CATALOG = {}
    _canon.cache_clear()


def _catalog_for(attr: str | None) -> list[str]:
    if _CATALOG is None:
        load_catalog("catalog.json")
    if not attr:
        return [v for values in (_CATALOG or {}).values() for v in values]
    return [v for sec in _ATTR_SECTIONS.get(attr, ()) for v in (_CATALOG or {}).get(sec, ())]


@lru_cache(maxsize=100_000)
def _canon(value: str, attr: str | None = None) -> str:
    """Каноническая форма: ближайшая запись каталога СЕКЦИЙ атрибута (≥0.85)."""
    best, best_sim = value, 0.0
    for entry in _catalog_for(attr):
        s_ = _value_sim(entry, value)
        if s_ > best_sim:
            best, best_sim = entry, s_
    return best if best_sim >= VALUE_SIM else value


# ── восстановление относительных дат из транскрипта ──────────────────────────
_WEEKDAYS = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")
_WD_GEN = {0: "понедельника", 1: "вторника", 2: "среды", 3: "четверга",
           4: "пятницы", 5: "субботы", 6: "воскресенья"}
_WD_ACC = {0: "понедельник", 1: "вторник", 2: "среду", 3: "четверг",
           4: "пятницу", 5: "субботу", 6: "воскресенье"}
_GEN_TO_WD = {v: k for k, v in _WD_GEN.items()}
_ACC_TO_WD = {v: k for k, v in _WD_ACC.items()}
_NUM_REV = {"один": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6,
            "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "пару": 2}
_NUM_ALT = r"(\d{1,2}|" + "|".join(_NUM_REV) + r")"
_GEN_ALT, _ACC_ALT = "|".join(_WD_GEN.values()), "|".join(_WD_ACC.values())

_DAYS_AGO = re.compile(rf"\b{_NUM_ALT}\s+дн(?:ей|я|ю)?\s+(?:тому\s+)?назад\b")
_DAYS_AGO_INV = re.compile(rf"\bдн(?:ей|я|ю)?\s+{_NUM_ALT}\s+назад\b")
_FROM_WD = re.compile(rf"(?:^|\W)(?:с|со)\s+({_GEN_ALT})\b")
_ON_WD = re.compile(rf"(?:^|\W)(?:в|во)\s+({_ACC_ALT})\b")
_IN_DAYS = re.compile(rf"\bчерез\s+{_NUM_ALT}\s+дн(?:я|ей|ю)?\b")
_FIXED = {"позавчера": -2, "вчера": -1, "сегодня": 0, "завтра": 1, "послезавтра": 2}
_WEEK_AGO = re.compile(r"\bнеделю\s+назад\b")
_WEEK_AHEAD = re.compile(r"\bчерез\s+неделю\b")
_LAST_WEEK = re.compile(r"\bна\s+прошлой\s+неделе\b")
_MONTH_AGO = re.compile(r"\bмесяц\s+назад\b")
_LAST_MONTH = re.compile(r"\bв\s+прошлом\s+месяце\b")
_NEXT_MONTH = re.compile(r"\bв\s+следующем\s+месяце\b")
_MONTH_AHEAD = re.compile(r"\bчерез\s+месяц\b")
_TWO_WEEKS = re.compile(r"\bчерез\s+две\s+недели\b")
_NEXT_WEEK = re.compile(r"\bна\s+следующей\s+неделе\b")
_DAY_BEFORE_YESTERDAY = re.compile(r"\bпозапрошл")
_THESE_DAYS = re.compile(r"\bна\s+днях\b")


def _last_past_weekday(base: date, wd: int) -> date:
    delta = (base.weekday() - wd) % 7 or 7
    return base - timedelta(days=delta)


def _next_weekday(base: date, wd: int) -> date:
    delta = (wd - base.weekday()) % 7 or 7
    return base + timedelta(days=delta)


def recover_dates(text: str, call_date: date) -> set[date]:
    """Даты, восстановимые из текста относительно даты звонка (банк пул генератора)."""
    out: set[date] = set()
    low = text.lower()
    for m in _DAYS_AGO.finditer(low):
        n = _NUM_REV.get(m.group(1), int(m.group(1)) if m.group(1).isdigit() else None)
        if n is not None:
            out.add(call_date - timedelta(days=n))
    for m in _DAYS_AGO_INV.finditer(low):
        n = _NUM_REV.get(m.group(1), int(m.group(1)) if m.group(1).isdigit() else None)
        if n is not None:
            out.add(call_date - timedelta(days=n))
    for m in _FROM_WD.finditer(low):
        out.add(_last_past_weekday(call_date, _GEN_TO_WD[m.group(1)]))
    for m in _ON_WD.finditer(low):
        wd = _ACC_TO_WD[m.group(1)]
        out.add(_last_past_weekday(call_date, wd))
        out.add(_next_weekday(call_date, wd))
    for m in _IN_DAYS.finditer(low):
        n = _NUM_REV.get(m.group(1), int(m.group(1)) if m.group(1).isdigit() else None)
        if n is not None:
            out.add(call_date + timedelta(days=n))
    for word, delta in _FIXED.items():
        if re.search(rf"\b{word}\b", low):
            out.add(call_date + timedelta(days=delta))
    if _WEEK_AGO.search(low):
        out.add(call_date - timedelta(days=7))
    if _WEEK_AHEAD.search(low):
        out.add(call_date + timedelta(days=7))
    if _LAST_WEEK.search(low):
        monday_prev = call_date - timedelta(days=call_date.weekday() + 7)
        out.update(monday_prev + timedelta(days=i) for i in range(7))
    if _MONTH_AGO.search(low):
        out.update(call_date - timedelta(days=n) for n in (28, 29, 30, 31))
    if _LAST_MONTH.search(low):
        # первый день ТЕКУЩЕГО месяца минус 1 день = последний день прошлого
        this_month = call_date.replace(day=1)
        prev = this_month - timedelta(days=1)
        prev = prev.replace(day=1)
        n_days = (this_month - prev).days
        out.update(prev + timedelta(days=i) for i in range(n_days))
    if _NEXT_MONTH.search(low):                       # «в следующем месяце»
        nxt = (call_date.replace(day=28) + timedelta(days=7)).replace(day=1)
        n_days = ((nxt + timedelta(days=32)).replace(day=1) - nxt).days
        out.update(nxt + timedelta(days=i) for i in range(n_days))
    if _MONTH_AHEAD.search(low):
        out.update(call_date + timedelta(days=n) for n in range(28, 32))
    if _TWO_WEEKS.search(low):
        out.update(call_date + timedelta(days=13), call_date + timedelta(days=14), call_date + timedelta(days=15))
    if _NEXT_WEEK.search(low):
        monday_next = call_date + timedelta(days=(7 - call_date.weekday()))
        out.update(monday_next + timedelta(days=i) for i in range(7))
    if _DAY_BEFORE_YESTERDAY.search(low):
        out.update(call_date - timedelta(days=3), call_date - timedelta(days=4))
    if _THESE_DAYS.search(low):
        out.update(call_date - timedelta(days=n) for n in (2, 3, 4, 5))
    return out


def _day(iso: str | None) -> str | None:
    return iso[:10] if iso else None


# ── сопоставление фактов ─────────────────────────────────────────────────────
def match_facts(gold: list[dict], pred: list[dict]) -> list[tuple[dict, dict, float]]:
    """Жадкое взаимно-однозначное сопоставление (лучшие пары первыми)."""
    pairs = []
    for gi, g in enumerate(gold):
        for pi, p in enumerate(pred):
            if g["attribute"] != p.get("attribute"):
                continue
            if (g.get("polarity", 1) > 0) != (p.get("polarity", 1) > 0):
                continue
            gv, pv = g["value"], p.get("value", "")
            vsim = _value_sim(gv, pv)
            cg, cp = _canon(gv, g.get("attribute")), _canon(pv, g.get("attribute"))
            if cg == cp and (cg != gv or cp != pv):
                vsim = max(vsim, 1.0)          # обе стороны снапились к одному канону
            need = VALUE_SIM if min(len(_tokens(gv)), len(_tokens(pv))) <= 2 else VALUE_SIM_LONG
            osim = _object_sim(g, p)
            if vsim >= need and osim > 0:
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


# ── оценка корпуса ───────────────────────────────────────────────────────────
def evaluate(gold_calls: dict[str, dict], pred_calls: dict[str, dict]) -> dict:
    n_gold = n_pred = n_matched = 0
    temporal_gold = temporal_ok = 0
    state_total = state_ok = 0
    ground_total = ground_turn_ok = ground_span_total = ground_span_ok = 0
    for cid, g in gold_calls.items():
        gf = g["facts"]
        pf = pred_calls.get(cid, {}).get("facts") or []
        n_gold += len(gf)
        n_pred += len(pf)
        call_day = datetime.fromisoformat(g["started_at"]).date()
        turns = g.get("turns", [])
        matched = match_facts(gf, pf)
        n_matched += len(matched)
        for gfact, pfact, _s in matched:
            # темпоральная полнота: только факты с восстановимой из текста датой
            if gfact["attribute"] in TEMPORAL_ATTRS:
                # дата обязана быть восстановимой из ТРАНСКРИПТА (окно ±1 от якоря
                # оказалось слишком строгим: формулировки судьи ссылаются не на ту реплику)
                recovered: set[date] = set()
                for t in turns:
                    recovered |= recover_dates(t["text"], call_day)
                gold_days = {_day(gfact.get("valid_from")), _day(gfact.get("valid_to"))} - {None}
                rec_days = {d.isoformat() for d in recovered}
                if gold_days and rec_days & gold_days:
                    temporal_gold += 1
                    if _day(gfact.get("valid_from")) == _day(pfact.get("valid_from")) \
                            and _day(gfact.get("valid_to")) == _day(pfact.get("valid_to")):
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
        # точность текущего состояния — по паре (объект, атрибут)
        by_key: dict[tuple, dict] = {}
        for f in gf:
            if f["attribute"] in SINGLE_VALUED and f.get("valid_to") is None:
                k = (f["object"]["id"], f["attribute"])
                if k not in by_key or f["valid_from"] > by_key[k]["valid_from"]:
                    by_key[k] = f
        for (_oid, attr), gfact in by_key.items():
            state_total += 1
            cands = [p for p in pf if p.get("attribute") == attr and p.get("valid_to") is None
                     and _object_sim(gfact, p) > 0]
            if cands:
                # при равных дат приоритет кандидату с похожим значением (безымянные
                # gold-объекты порождают чужих кандидатов с той же датой)
                last = max(cands, key=lambda p: (p.get("valid_from") or "",
                                                 _value_sim(gfact["value"], p.get("value", ""))))
                need = VALUE_SIM if min(len(_tokens(gfact["value"])),
                                        len(_tokens(last.get("value", "")))) <= 2 else VALUE_SIM_LONG
                if _value_sim(gfact["value"], last.get("value", "")) >= need:
                    state_ok += 1
    return {
        "metrics_version": VERSION,
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

    load_catalog(a.gold)
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
