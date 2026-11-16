"""Проверка выходной схемы кейса (schema.md §2, §5): обязательные поля, типы, даты ISO, интервалы."""
from datetime import datetime

ATTRS = {"service_state", "tariff", "ticket_status", "action", "agreement"}


def _iso(x):
    try:
        datetime.fromisoformat(x)
        return True
    except Exception:
        return False


def validate_call(call):
    errs = []
    n = len(call.get("turns", []))
    for i, f in enumerate(call.get("facts", [])):
        p = f"facts[{i}]"
        if not isinstance(f.get("object"), dict) or not isinstance(f["object"].get("name"), str):
            errs.append(p + ".object.name")
        if f.get("attribute") not in ATTRS:
            errs.append(p + ".attribute")
        if not isinstance(f.get("value"), str) or not f["value"].strip():
            errs.append(p + ".value")
        if f.get("polarity") not in (1, -1):
            errs.append(p + ".polarity")
        if f.get("speaker") not in ("client", "operator"):
            errs.append(p + ".speaker")
        if not (isinstance(f.get("valid_from"), str) and _iso(f["valid_from"])):
            errs.append(p + ".valid_from")
        vt = f.get("valid_to")
        if vt is not None and not (isinstance(vt, str) and _iso(vt) and vt[:10] >= f["valid_from"][:10]):
            errs.append(p + ".valid_to")
        ti = (f.get("source") or {}).get("turn_idx")
        if not isinstance(ti, int) or not 0 <= ti < max(n, 1):
            errs.append(p + ".source.turn_idx")
    return errs
