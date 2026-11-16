"""Совпадение значений ровно по правилам оценщика v2.1 (используется для разметки кандидатов)."""
from functools import lru_cache
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from metrics_util import fixed
_sf = fixed()


@lru_cache(maxsize=500_000)
def value_match(gold_value: str, pred_value: str, attribute: str) -> bool:
    if gold_value.strip().lower() == pred_value.strip().lower():
        return True
    v = _sf._value_sim(gold_value, pred_value)
    cg, cp = _sf._canon(gold_value, attribute), _sf._canon(pred_value, attribute)
    if cg == cp and (cg != gold_value or cp != pred_value):
        v = 1.0
    need = _sf.VALUE_SIM if min(len(_sf._tokens(gold_value)), len(_sf._tokens(pred_value))) <= 2 else _sf.VALUE_SIM_LONG
    return v >= need


def object_match(fact, name):
    return _sf._object_sim(fact, {"object": {"name": name}}) > 0
