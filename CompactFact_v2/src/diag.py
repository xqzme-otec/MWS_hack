"""Диагностика дат среди совпавших фактов (по правилам оценщика): from/to по атрибутам."""
import sys, os, json, collections
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))
from metrics_util import fixed, load
sf = fixed()
def diag(gold, pred, show=0):
    st = collections.defaultdict(collections.Counter); ex = []
    for cid, g in gold.items():
        pf = pred.get(cid, {}).get('facts', [])
        cd = datetime.fromisoformat(g['started_at']).date(); rec = set()
        for t in g['turns']: rec |= sf.recover_dates(t['text'], cd)
        recs = {d.isoformat() for d in rec}
        for gf, p, _ in sf.match_facts(g['facts'], pf):
            a = gf['attribute']
            if a not in sf.TEMPORAL_ATTRS: continue
            gd = {gf['valid_from'][:10]} | ({gf['valid_to'][:10]} if gf['valid_to'] else set())
            if not gd & recs: continue
            okf = (p['valid_from'] or '')[:10] == gf['valid_from'][:10]
            okt = (p.get('valid_to') or '')[:10] == (gf['valid_to'] or '')[:10]
            s = st[a]; s['n'] += 1; s['from_ok'] += okf; s['to_ok'] += okt; s['both'] += okf and okt
            s['same_turn'] += p['source']['turn_idx'] == gf['source']['turn_idx']
            if not okt: s['to_gold_none'] += gf['valid_to'] is None; s['to_pred_none'] += p.get('valid_to') is None
            if not (okf and okt) and len(ex) < show:
                ex.append((a, gf['value'], p['value'], gf['valid_from'][:10], p['valid_from'], gf['valid_to'], p.get('valid_to'), gf['source']['turn_idx'], p['source']['turn_idx'], g['started_at'][:10]))
    for a, s in st.items(): print(a, dict(s))
    for e in ex: print(e)
if __name__ == '__main__':
    diag(load(sys.argv[1]), load(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 0)
