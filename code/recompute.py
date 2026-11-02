#!/usr/bin/env python3
"""Recompute all statistics of the med6m article from the paired counts (CPU only, no GPU, no network).
   python recompute.py  ->  results/recomputed.json
Method: for every item d_i = c_i - b_i in {-1,0,+1} (candidate correct minus base correct).
 * Delta_pp = 100/n * sum d_i
 * 95% CI: percentile paired bootstrap implemented as multinomial resampling of the counts
   (n_-1, n_0, n_+1) ~ Mult(n, p_hat), B=20000, seed 20261101 (equivalent to item resampling).
 * exact McNemar test (two-sided, binomial) on the discordant pairs b=only_base, c=only_model.
 * No multiple-testing correction is applied (reported as such)."""
import json, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
B, SEED = 20000, 20261101

def mcnemar_exact(b, c):
    m = b + c
    if m == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(m, i) for i in range(0, k + 1)) / 2 ** m
    return min(1.0, 2 * p)

def stats(c, n=None):
    n = n or sum(c[k] for k in ('both', 'only_model', 'only_base', 'neither'))
    assert sum(c[k] for k in ('both', 'only_model', 'only_base', 'neither')) == n
    pr = np.array([c['only_base'], c['both'] + c['neither'], c['only_model']], float) / n
    draw = np.random.default_rng(SEED).multinomial(n, pr, size=B)
    boot = 100 * (draw[:, 2] - draw[:, 0]) / n
    lo, hi = np.quantile(boot, [.025, .975])
    return {**c, 'n': n,
            'base_acc': 100 * (c['both'] + c['only_base']) / n,
            'cand_acc': 100 * (c['both'] + c['only_model']) / n,
            'delta_pp': 100 * (c['only_model'] - c['only_base']) / n,
            'ci95_pp': [float(lo), float(hi)], 'se_pp': float(boot.std(ddof=1)),
            'mcnemar_p': mcnemar_exact(c['only_base'], c['only_model'])}

if __name__ == '__main__':
    sealed = json.loads((ROOT / 'results/paired_counts.json').read_text())
    out = {'method': 'paired multinomial bootstrap + exact McNemar', 'B': B, 'seed': SEED,
           'multiple_testing_adjusted': False, 'sealed': {}}
    for sc, rows in sealed.items():
        if sc.startswith('_'): continue
        out['sealed'][sc] = {k: stats(v) for k, v in rows.items()}
    dev = json.loads((ROOT / 'results/dev_final886_counts.json').read_text())
    dev = {k: v for k, v in dev.items() if not k.startswith('_')}
    out['dev_final886'] = {k: stats(v) for k, v in dev.items()}
    pool = {k: sum(dev[b][k] for b in ('medmcqa', 'medqa', 'mmlu_med', 'pubmedqa')) for k in ('both', 'only_model', 'only_base', 'neither')}
    out['dev_final886']['medical_pool'] = stats(pool)
    (ROOT / 'results/recomputed.json').write_text(json.dumps(out, indent=2))
    for sc in out['sealed']:
        p = out['sealed'][sc]['medical_pool']
        print(sc, round(p['delta_pp'], 4), [round(x, 4) for x in p['ci95_pp']], 'McNemar p=%.2e' % p['mcnemar_p'])
    p = out['dev_final886']['medical_pool']; print('dev886 pool', round(p['delta_pp'], 3), [round(x, 3) for x in p['ci95_pp']])
