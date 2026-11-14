"""Разбор ошибок по типам фактов: P/R по атрибуту, даты (from/to) среди совпавших, текущее состояние,
причины пропусков. Изменения состояния во времени (одно-значные атрибуты) — отдельно от статических."""
import sys, os, json, collections
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))
from metrics_util import fixed, strict, load
sf, ss = fixed(), strict()
gold, pred = load(sys.argv[1]), load(sys.argv[2])
A = ['service_state', 'tariff', 'ticket_status', 'action', 'agreement']
st = {a: collections.Counter() for a in A}
for cid, g in gold.items():
    pf = pred.get(cid, {}).get('facts', [])
    for f in g['facts']: st[f['attribute']]['gold'] += 1
    for f in pf: st[f['attribute']]['pred'] += 1
    cd = datetime.fromisoformat(g['started_at']).date(); rec = set()
    for t in g['turns']: rec |= sf.recover_dates(t['text'], cd)
    recs = {d.isoformat() for d in rec}
    for gf, p, _ in sf.match_facts(g['facts'], pf):
        a = gf['attribute']; s = st[a]; s['match'] += 1
        s['turn_ok'] += p['source']['turn_idx'] == gf['source']['turn_idx']
        if a in sf.TEMPORAL_ATTRS:
            gd = {gf['valid_from'][:10]} | ({gf['valid_to'][:10]} if gf['valid_to'] else set())
            if gd & recs:
                s['t_n'] += 1
                okf = (p['valid_from'] or '')[:10] == gf['valid_from'][:10]
                okt = (p.get('valid_to') or '')[:10] == (gf['valid_to'] or '')[:10]
                s['t_ok'] += okf and okt; s['t_from_bad'] += not okf; s['t_to_bad'] += not okt
                s['t_hidden'] += not (gf['valid_from'][:10] in recs or gf['valid_from'][:10] == cd.isoformat())
    # текущее состояние (строго)
    by = {}
    for f in g['facts']:
        if f['attribute'] in ss.SINGLE_VALUED and f.get('valid_to') is None:
            k = (f['object']['id'], f['attribute'])
            if k not in by or f['valid_from'] > by[k]['valid_from']: by[k] = f
    for (oid, a), gf in by.items():
        s = st[a]; s['s_n'] += 1
        att = [p for p in pf if p['attribute'] == a]; obj = [p for p in att if ss._object_sim(gf, p) > 0]
        op = [p for p in obj if p.get('valid_to') is None]
        if not att: s['s_miss_attr'] += 1; continue
        if not obj: s['s_obj'] += 1; continue
        if not op: s['s_closed'] += 1; continue
        last = max(op, key=lambda p: (p.get('valid_from') or '', p['source']['turn_idx']))
        need = ss.VALUE_SIM if min(len(ss._tokens(gf['value'])), len(ss._tokens(last['value']))) <= 2 else ss.VALUE_SIM_LONG
        if ss._value_sim(gf['value'], last['value']) >= need: s['s_ok'] += 1
        elif any(ss._value_sim(gf['value'], p['value']) >= need for p in op): s['s_wrong_pick'] += 1
        else: s['s_value_missing'] += 1
rows = []
for a in A:
    s = st[a]
    r = {'attribute': a, 'kind': 'изменение состояния' if a in ss.SINGLE_VALUED else 'статический/событие',
         'gold': s['gold'], 'pred': s['pred'], 'precision': round(s['match'] / max(1, s['pred']), 3), 'recall': round(s['match'] / max(1, s['gold']), 3),
         'grounding': round(s['turn_ok'] / max(1, s['match']), 3)}
    if a in sf.TEMPORAL_ATTRS:
        r.update({'temporal_n': s['t_n'], 'temporal_acc': round(s['t_ok'] / max(1, s['t_n']), 3), 'from_wrong': s['t_from_bad'], 'to_wrong': s['t_to_bad'], 'hidden_start': s['t_hidden']})
    if a in ss.SINGLE_VALUED:
        r.update({'state_n': s['s_n'], 'state_acc': round(s['s_ok'] / max(1, s['s_n']), 3), 'no_fact': s['s_miss_attr'], 'wrong_object': s['s_obj'],
                  'wrong_value_picked': s['s_wrong_pick'], 'value_missing': s['s_value_missing'], 'all_closed': s['s_closed']})
    rows.append(r)
print(json.dumps(rows, ensure_ascii=False, indent=1))
json.dump(rows, open(sys.argv[3], 'w'), ensure_ascii=False, indent=1)
