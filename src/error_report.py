"""Разбор ошибок по типам фактов (официальные функции сопоставления case_metrics v2.1).

Для каждого атрибута: P / R / F1, верная реплика, точность дат среди восстановимых, текущее состояние;
причины пропусков и лишних фактов с примерами; сравнение с базовым LLM-экстрактором и с adversarial.

    python src/error_report.py --pred reports/eval/hybrid_val_pred.jsonl \
        --adv reports/eval/hybrid_val_adv_pred.jsonl --baseline data/baseline_pred.jsonl
→ reports/error_analysis.md, reports/error_analysis.json
"""
import argparse, collections, json, os, sys
from datetime import datetime

here = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(here, "..")
sys.path.insert(0, os.path.join(ROOT, "data"))
import case_metrics as cm  # noqa: E402

A = ["service_state", "tariff", "ticket_status", "action", "agreement"]
RU = {"service_state": "состояние услуги", "tariff": "тариф", "ticket_status": "статус заявки",
      "action": "действие", "agreement": "договорённость"}


def load(p):
    return {c["call_id"]: c for c in map(json.loads, open(p, encoding="utf-8"))}


def analyse(gold, pred, examples=True):
    st = {a: collections.Counter() for a in A}
    ex = collections.defaultdict(list)
    for cid, g in gold.items():
        gf, pf = g["facts"], (pred.get(cid) or {}).get("facts") or []
        cd = datetime.fromisoformat(g["started_at"]).date()
        rec = set()
        for t in g["turns"]:
            rec |= cm.recover_dates(t["text"], cd)
        recs = {d.isoformat() for d in rec}
        matched = cm.match_facts(gf, pf)
        mg = {id(x) for x, _, _ in matched}; mp = {id(y) for _, y, _ in matched}
        for f in gf: st[f["attribute"]]["gold"] += 1
        for f in pf: st[f["attribute"]]["pred"] += 1
        for g1, p1, _ in matched:
            a = g1["attribute"]; s = st[a]; s["match"] += 1
            s["turn_ok"] += (p1.get("source") or {}).get("turn_idx") == g1["source"]["turn_idx"]
            if a in cm.TEMPORAL_ATTRS:
                gd = {cm._day(g1.get("valid_from")), cm._day(g1.get("valid_to"))} - {None}
                if gd & recs:
                    s["t_n"] += 1
                    okf = cm._day(g1.get("valid_from")) == cm._day(p1.get("valid_from"))
                    okt = cm._day(g1.get("valid_to")) == cm._day(p1.get("valid_to"))
                    s["t_ok"] += okf and okt; s["t_from_bad"] += not okf; s["t_to_bad"] += not okt
                    if not (okf and okt) and examples and len(ex[(a, "date")]) < 2:
                        ex[(a, "date")].append(_ex(g, g1, p1))
        # причины пропусков
        for g1 in gf:
            if id(g1) in mg:
                continue
            a = g1["attribute"]
            same = [p for p in pf if p.get("attribute") == a]
            near = [p for p in same if abs((p.get("source") or {}).get("turn_idx", -99) - g1["source"]["turn_idx"]) <= 1]
            if not same:
                why = "нет ни одного факта этого типа в звонке"
            elif near and max(cm._value_sim(g1["value"], p["value"]) for p in near) >= 0.7:
                why = "значение верное, но другой объект/полярность"
            elif near:
                why = "в той же реплике выбрано не то значение"
            else:
                why = "факт этого типа найден, но в другом месте звонка"
            s = st[a]; s["miss:" + why] += 1
            if examples and len(ex[(a, why)]) < 2:
                ex[(a, why)].append(_ex(g, g1, (near or same or [None])[0]))
        # лишние
        for p1 in pf:
            if id(p1) in mp:
                continue
            a = p1.get("attribute")
            near = [x for x in gf if x["attribute"] == a and abs(x["source"]["turn_idx"] - (p1.get("source") or {}).get("turn_idx", -99)) <= 1]
            why = "дубль/другая формулировка эталонного факта" if near else "лишний факт (нет в эталоне рядом)"
            st[a]["extra:" + why] += 1
            if examples and len(ex[(a, why)]) < 2:
                ex[(a, why)].append(_ex(g, near[0] if near else None, p1))
        # текущее состояние по атрибуту (как в оценщике)
        by = {}
        for f in gf:
            if f["attribute"] in cm.SINGLE_VALUED and f.get("valid_to") is None:
                k = (f["object"]["id"], f["attribute"])
                if k not in by or f["valid_from"] > by[k]["valid_from"]:
                    by[k] = f
        for (_o, a), g1 in by.items():
            st[a]["s_n"] += 1
            c = [p for p in pf if p.get("attribute") == a and p.get("valid_to") is None and cm._object_sim(g1, p) > 0]
            if c:
                last = max(c, key=lambda p: (p.get("valid_from") or "", cm._value_sim(g1["value"], p.get("value", ""))))
                need = cm.VALUE_SIM if min(len(cm._tokens(g1["value"])), len(cm._tokens(last.get("value", "")))) <= 2 else cm.VALUE_SIM_LONG
                st[a]["s_ok"] += cm._value_sim(g1["value"], last.get("value", "")) >= need
    rows = {}
    for a in A:
        s = st[a]
        P = s["match"] / s["pred"] if s["pred"] else 0.0
        R = s["match"] / s["gold"] if s["gold"] else 0.0
        rows[a] = {"gold": s["gold"], "pred": s["pred"], "precision": round(P, 3), "recall": round(R, 3),
                   "f1": round(2 * P * R / (P + R), 3) if P + R else 0.0,
                   "grounding_turn": round(s["turn_ok"] / s["match"], 3) if s["match"] else None,
                   "temporal_n": s["t_n"], "temporal_acc": round(s["t_ok"] / s["t_n"], 3) if s["t_n"] else None,
                   "date_from_wrong": s["t_from_bad"], "date_to_wrong": s["t_to_bad"],
                   "state_n": s["s_n"], "state_acc": round(s["s_ok"] / s["s_n"], 3) if s["s_n"] else None,
                   "misses": {k[5:]: v for k, v in s.items() if k.startswith("miss:")},
                   "extras": {k[6:]: v for k, v in s.items() if k.startswith("extra:")}}
    return rows, ex


def _ex(call, g, p):
    def short(f):
        if not f:
            return None
        return {"attr": f.get("attribute"), "value": f.get("value"), "object": (f.get("object") or {}).get("name"),
                "from": cm._day(f.get("valid_from")), "to": cm._day(f.get("valid_to")),
                "turn": (f.get("source") or {}).get("turn_idx")}
    ti = (g or p or {}).get("source", {}).get("turn_idx", 0)
    turn = next((t["text"] for t in call["turns"] if t["idx"] == ti), "")
    return {"call_id": call["call_id"], "turn": ti, "text": turn[:220], "gold": short(g), "pred": short(p)}


def fmt(x):
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/val.jsonl")
    ap.add_argument("--pred", default="reports/eval/hybrid_val_pred.jsonl")
    ap.add_argument("--adv_gold", default="data/val_adv_0.3.jsonl")
    ap.add_argument("--adv", default="reports/eval/hybrid_val_adv_pred.jsonl")
    ap.add_argument("--baseline", default="data/baseline_pred.jsonl")
    ap.add_argument("--out", default="reports/error_analysis")
    a = ap.parse_args()
    os.chdir(ROOT)
    cm.load_catalog(a.gold)
    gold = load(a.gold)
    ours, ex = analyse(gold, load(a.pred))
    base = analyse(gold, load(a.baseline), examples=False)[0] if os.path.exists(a.baseline) else None
    adv = analyse(load(a.adv_gold), load(a.adv), examples=False)[0] if os.path.exists(a.adv) else None
    json.dump({"ours": ours, "baseline": base, "adversarial": adv,
               "examples": {f"{k[0]}|{k[1]}": v for k, v in ex.items()}},
              open(a.out + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    L = ["# Разбор ошибок по типам фактов", "",
         f"Набор: `{a.gold}` ({len(gold)} звонков), оценщик — официальный `case_metrics.py` v2.1 (функции сопоставления без изменений).", "",
         "## 1. Сводка по атрибутам (наша модель)", "",
         "| атрибут | эталон | выдано | P | R | F1 | верная реплика | даты верны (n) | текущее сост. (n) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k in A:
        r = ours[k]
        L.append(f"| {RU[k]} | {r['gold']} | {r['pred']} | {fmt(r['precision'])} | {fmt(r['recall'])} | **{fmt(r['f1'])}** | "
                 f"{fmt(r['grounding_turn'])} | {fmt(r['temporal_acc'])} ({r['temporal_n']}) | {fmt(r['state_acc'])} ({r['state_n']}) |")
    if base:
        L += ["", "## 2. Сравнение с базовым LLM-экстрактором (F1 по атрибутам)", "",
              "| атрибут | база F1 | наша F1 | Δ | база: даты | наша: даты | база: текущее | наша: текущее |", "|---|---|---|---|---|---|---|---|"]
        for k in A:
            b, o = base[k], ours[k]
            L.append(f"| {RU[k]} | {fmt(b['f1'])} | {fmt(o['f1'])} | {o['f1'] - b['f1']:+.3f} | {fmt(b['temporal_acc'])} | {fmt(o['temporal_acc'])} | "
                     f"{fmt(b['state_acc'])} | {fmt(o['state_acc'])} |")
    if adv:
        L += ["", "## 3. Устойчивость: val → adversarial (ASR-шум 0.3), F1 по атрибутам", "",
              "| атрибут | val F1 | adversarial F1 | Δ |", "|---|---|---|---|"]
        for k in A:
            L.append(f"| {RU[k]} | {fmt(ours[k]['f1'])} | {fmt(adv[k]['f1'])} | {adv[k]['f1'] - ours[k]['f1']:+.3f} |")
    L += ["", "## 4. Причины ошибок", "",
          "Пропуск — эталонный факт без пары у модели; лишний — факт модели без пары в эталоне.", ""]
    for k in A:
        r = ours[k]
        L.append(f"### {RU[k]} (`{k}`)")
        L.append("")
        tot_m = sum(r["misses"].values()) or 1
        for why, n in sorted(r["misses"].items(), key=lambda x: -x[1]):
            L.append(f"- пропуск: {why} — **{n}** ({100 * n / tot_m:.0f}% пропусков)")
        tot_e = sum(r["extras"].values()) or 1
        for why, n in sorted(r["extras"].items(), key=lambda x: -x[1]):
            L.append(f"- лишний: {why} — **{n}** ({100 * n / tot_e:.0f}% лишних)")
        if r["temporal_n"]:
            L.append(f"- даты: неверное начало {r['date_from_wrong']}, неверный конец {r['date_to_wrong']} из {r['temporal_n']} восстановимых")
        shown, used = 0, set()
        for (kk, why), lst in ex.items():
            if kk != k or shown >= 3:
                continue
            e = next((x for x in lst if (x["call_id"], x["turn"]) not in used), None)
            if e is None:
                continue
            used.add((e["call_id"], e["turn"])); shown += 1
            g, p = e["gold"], e["pred"]
            L.append(f"  - пример ({why}), `{e['call_id']}` реплика {e['turn']}: «{e['text']}»")
            if g:
                L.append(f"    эталон: **{g['value']}** · {g['object']} · {g['from']}{' → ' + g['to'] if g['to'] else ''}")
            if p:
                L.append(f"    модель: **{p['value']}** · {p['object']} · {p['from']}{' → ' + p['to'] if p['to'] else ''} (реплика {p['turn']})")
        L.append("")
    open(a.out + ".md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L[:20]))


if __name__ == "__main__":
    main()
