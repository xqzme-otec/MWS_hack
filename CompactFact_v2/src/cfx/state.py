"""Цепочки одно-значных атрибутов: новое значение закрывает предыдущее."""
from .text import SINGLE_VALUED, norm


def close_chains(facts, key=lambda f: (norm(f["object"]["name"]), f["attribute"]), mode="chain_first"):
    groups = {}
    for f in facts:
        if f["attribute"] in SINGLE_VALUED and f.get("polarity", 1) == 1:
            groups.setdefault(key(f), []).append(f)
    for g in groups.values():
        g.sort(key=lambda f: ((f["valid_from"] or "")[:10], f["source"]["turn_idx"] or 0))
        for a, b in zip(g, g[1:]):
            if norm(a["value"]) == norm(b["value"]):
                continue
            chained = b["valid_from"]
            if mode == "chain_first" or a.get("valid_to") is None:
                a["valid_to"] = chained
        # последнее значение открыто, если модель не нашла явного окончания
        if mode == "chain_first" and g:
            last = g[-1]
            if last.get("_vt_model") is None:
                last["valid_to"] = None
    return facts
