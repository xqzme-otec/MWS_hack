"""Типизированный извлекатель относительных дат.

Банк выражений повторяет восстановитель оценщика v2.1 (case_metrics.recover_dates),
но возвращает каждое выражение отдельно: тип, позицию в тексте и набор дат-кандидатов.
Так выбор конкретной даты для факта можно обучить, а не угадывать.
"""
from __future__ import annotations
import re
from datetime import date, datetime, timedelta

NUM = {"один": 1, "одну": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6,
       "семь": 7, "восемь": 8, "девять": 9, "десять": 10}
NUM_ALT = r"(\d{1,2}|" + "|".join(NUM) + r")"
WD_GEN = {"понедельника": 0, "вторника": 1, "среды": 2, "четверга": 3, "пятницы": 4, "субботы": 5, "воскресенья": 6}
WD_ACC = {"понедельник": 0, "вторник": 1, "среду": 2, "четверг": 3, "пятницу": 4, "субботу": 5, "воскресенье": 6}


def _n(s):
    return NUM.get(s, int(s) if s.isdigit() else None)


def last_wd(base, wd):
    return base - timedelta(days=(base.weekday() - wd) % 7 or 7)


def next_wd(base, wd):
    return base + timedelta(days=(wd - base.weekday()) % 7 or 7)


PATTERNS = []  # (type, regex, fn(match, call_date) -> list[date])


def P(t, rx):
    def deco(fn):
        PATTERNS.append((t, re.compile(rx), fn))
        return fn
    return deco


@P("days_ago", rf"\b{NUM_ALT}\s+дн(?:ей|я|ю)?\s+(?:тому\s+)?назад\b")
def _(m, d): n = _n(m.group(1)); return [d - timedelta(days=n)] if n else []
@P("days_ago", rf"\bдн(?:ей|я|ю)?\s+{NUM_ALT}\s+назад\b")
def _(m, d): n = _n(m.group(1)); return [d - timedelta(days=n)] if n else []
@P("from_wd", r"(?:^|\W)(?:с|со)\s+(" + "|".join(WD_GEN) + r")\b")
def _(m, d): return [last_wd(d, WD_GEN[m.group(1)])]
@P("until_wd", r"\bдо\s+(" + "|".join(WD_GEN) + r")\b")
def _(m, d): return [last_wd(d, WD_GEN[m.group(1)])]
@P("on_wd", r"(?:^|\W)(?:в|во)\s+(" + "|".join(WD_ACC) + r")\b")
def _(m, d): wd = WD_ACC[m.group(1)]; return [last_wd(d, wd), next_wd(d, wd)]
@P("in_days", rf"\bчерез\s+{NUM_ALT}\s+дн(?:я|ей|ю)?\b")
def _(m, d): n = _n(m.group(1)); return [d + timedelta(days=n)] if n else []
for _w, _dl in {"позавчера": -2, "вчера": -1, "сегодня": 0, "завтра": 1, "послезавтра": 2}.items():
    PATTERNS.append((f"fixed_{_w}", re.compile(rf"\b{_w}\b"), (lambda dl: (lambda m, d: [d + timedelta(days=dl)]))(_dl)))
@P("week_ago", r"\bнеделю\s+назад\b")
def _(m, d): return [d - timedelta(days=7)]
@P("week_ahead", r"\bчерез\s+неделю\b")
def _(m, d): return [d + timedelta(days=7)]
@P("last_week", r"\bна\s+прошлой\s+неделе\b")
def _(m, d):
    mon = d - timedelta(days=d.weekday() + 7); return [mon + timedelta(days=i) for i in range(7)]
@P("month_ago", r"\bмесяц\s+назад\b")
def _(m, d): return [d - timedelta(days=n) for n in (28, 29, 30, 31)]
@P("last_month", r"\bв\s+прошлом\s+месяце\b")
def _(m, d):
    this = d.replace(day=1); prev = (this - timedelta(days=1)).replace(day=1)
    return [prev + timedelta(days=i) for i in range((this - prev).days)]
@P("next_month", r"\bв\s+следующем\s+месяце\b")
def _(m, d):
    nxt = (d.replace(day=28) + timedelta(days=7)).replace(day=1)
    return [nxt + timedelta(days=i) for i in range(((nxt + timedelta(days=32)).replace(day=1) - nxt).days)]
@P("month_ahead", r"\bчерез\s+месяц\b")
def _(m, d): return [d + timedelta(days=n) for n in range(28, 32)]
@P("two_weeks", r"\bчерез\s+две\s+недели\b")
def _(m, d): return [d + timedelta(days=n) for n in (13, 14, 15)]
@P("next_week", r"\bна\s+следующей\s+неделе\b")
def _(m, d):
    mon = d + timedelta(days=7 - d.weekday()); return [mon + timedelta(days=i) for i in range(7)]
@P("pozaprosh", r"\bпозапрошл")
def _(m, d): return [d - timedelta(days=3), d - timedelta(days=4)]
@P("these_days", r"\bна\s+днях\b")
def _(m, d): return [d - timedelta(days=n) for n in (2, 3, 4, 5)]

TIME_RX = re.compile(r"\b(?:после|к|в|до)\s+(\d{1,2}|" + "|".join(["одного", "двух", "трёх", "трех", "четырёх", "четырех", "пяти", "шести", "семи", "восьми", "девяти", "десяти", "одиннадцати", "двенадцати"]) + r")(?:[:.](\d{2}))?(?:\s+(утра|дня|вечера|часов|часа))?")


def extract(text: str, call_date: date):
    """Все относительные выражения в тексте: [(type, start, end, [dates])]."""
    low = text.lower()
    out = []
    for t, rx, fn in PATTERNS:
        for m in rx.finditer(low):
            ds = fn(m, call_date)
            if ds:
                out.append((t, m.start(), m.end(), ds))
    out.sort(key=lambda x: x[1])
    return out
