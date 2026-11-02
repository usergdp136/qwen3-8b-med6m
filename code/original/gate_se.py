# Regola v2 (25/10/2026): margine calibrato sul rumore del dev. Bootstrap appaiato sugli item (stessi item base vs candidato).
import os, json, random
from common import *
def dev_ids(sets):
    d = os.path.join(P['eval'], 'dev'); return {s: [r['id'] for r in rjsonl(os.path.join(d, s + '.jsonl'))] for s in sets if os.path.exists(os.path.join(d, s + '.jsonl'))}
def _boot(diffs, B=2000, seed=7):
    n = len(diffs); rg = random.Random(seed); m = []
    for _ in range(B): m.append(sum(diffs[rg.randrange(n)] for _ in range(n)) / n)
    mu = sum(m) / B; return (sum((x - mu) ** 2 for x in m) / (B - 1)) ** 0.5
def gate_se(res, base, k=2.0, need_med_gain=True, B=1000):
    """Rifiuta se un set dev peggiora di > k*SE (SE bootstrap appaiato della differenza);
    se need_med_gain, richiede miglioramento (delta > 0) sul pool medico dev (tutti gli item medici insieme)."""
    why = []; rep = {}; pool = []
    ids = dev_ids(list(res['sets']))
    for s, L in ids.items():
        L = [i for i in L if i in res['items'] and i in base['items']]
        if not L: continue
        d = [res['items'][i] - base['items'][i] for i in L]; se = _boot(d, B)
        delta = 100 * sum(d) / len(d); rep[s] = {'n': len(L), 'delta': round(delta, 2), 'se': round(100 * se, 2)}
        if delta < -k * 100 * se: why.append(f"{s} delta {delta:.2f} < -{k}*SE ({100*se:.2f})")
        if s in MED_SETS: pool += d
    if pool:
        se = _boot(pool, B); delta = 100 * sum(pool) / len(pool)
        rep['MED_POOL'] = {'n': len(pool), 'delta': round(delta, 2), 'se': round(100 * se, 2), 'z': round(delta / (100 * se), 2) if se else None}
        if need_med_gain and not delta > 0: why.append(f"pool medico delta {delta:.2f} <= 0")
    return (not why), why, rep
