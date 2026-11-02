#!/usr/bin/env python3
"""Figures + LaTeX table fragments from results/recomputed.json (run recompute.py first)."""
import json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / 'results/recomputed.json').read_text())
FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'mathtext.fontset': 'cm'})
names = ['medmcqa', 'medqa', 'mmlu_med', 'pubmedqa', 'medical_pool', 'arc_it']
lab = {'medmcqa': 'MedMCQA', 'medqa': 'MedQA', 'mmlu_med': 'MMLU-med', 'pubmedqa': 'PubMedQA', 'medical_pool': 'Pool medico\n(n=3766)', 'arc_it': 'ARC-it\n(guardia)'}
labE = dict(lab); labE['medical_pool'] = 'Medical pool\n(n=3766)'; labE['arc_it'] = 'ARC-it\n(guard)'
S = R['sealed']

def forest(L, fn, title, xl, l1, l2):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    y = np.arange(len(names))[::-1]
    for off, sc, col, mk in ((0.14, '050', '#1f5fa8', 'o'), (-0.14, '075', '#9aa7b5', 's')):
        d = [S[sc][n]['delta_pp'] for n in names]; lo = [S[sc][n]['ci95_pp'][0] for n in names]; hi = [S[sc][n]['ci95_pp'][1] for n in names]
        ax.errorbar(d, y + off, xerr=[np.array(d) - lo, np.array(hi) - d], fmt=mk, color=col, capsize=3, label=(l1 if sc == '050' else l2), ms=5)
    ax.axvline(0, color='k', lw=.8); ax.set_yticks(y); ax.set_yticklabels([L[n] for n in names]); ax.set_xlabel(xl); ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='lower right'); fig.tight_layout(); fig.savefig(FIG / (fn + '.png'), dpi=200); fig.savefig(FIG / (fn + '.pdf')); plt.close(fig)
forest(lab, 'fig1_forest_it', 'Differenza adattatore $-$ base (test sigillato, checkpoint 443)\nIC 95% bootstrap appaiato', r'$\Delta$ accuratezza (punti percentuali)', 'scala 0.50', 'scala 0.75')
forest(labE, 'fig1_forest_en', 'Adapter $-$ base difference (sealed test, checkpoint 443)\n95% paired-bootstrap CI', r'$\Delta$ accuracy (percentage points)', 'scale 0.50', 'scale 0.75')

def bars(L, fn, bl, cl, yl):
    fig, ax = plt.subplots(figsize=(6.4, 3.4)); x = np.arange(len(names)); w = .38
    b = [S['050'][n]['base_acc'] for n in names]; c = [S['050'][n]['cand_acc'] for n in names]
    ax.bar(x - w / 2, b, w, color='#c9d3de', label=bl); ax.bar(x + w / 2, c, w, color='#1f5fa8', label=cl)
    for i, (u, v) in enumerate(zip(b, c)): ax.text(i - w / 2, u + .5, '%.1f' % u, ha='center', fontsize=7); ax.text(i + w / 2, v + .5, '%.1f' % v, ha='center', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([L[n] for n in names], fontsize=8); ax.set_ylim(40, 95); ax.set_ylabel(yl); ax.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)
    fig.tight_layout(); fig.savefig(FIG / (fn + '.png'), dpi=200); fig.savefig(FIG / (fn + '.pdf')); plt.close(fig)
bars(lab, 'fig2_acc_it', 'Base Qwen3-8B (4 bit)', 'Adattatore, scala 0.50', 'Accuratezza (%)')
bars(labE, 'fig2_acc_en', 'Base Qwen3-8B (4-bit)', 'Adapter, scale 0.50', 'Accuracy (%)')

def disc(L, fn, a, b, yl):
    fig, ax = plt.subplots(figsize=(6.4, 3.2)); x = np.arange(len(names)); w = .38
    g = [100 * S['050'][n]['only_model'] / S['050'][n]['n'] for n in names]; l = [100 * S['050'][n]['only_base'] / S['050'][n]['n'] for n in names]
    ax.bar(x - w / 2, g, w, color='#2e8b57', label=a); ax.bar(x + w / 2, l, w, color='#c0392b', label=b)
    for i, n in enumerate(names): ax.text(i - w / 2, g[i] + .15, str(S['050'][n]['only_model']), ha='center', fontsize=7); ax.text(i + w / 2, l[i] + .15, str(S['050'][n]['only_base']), ha='center', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([L[n] for n in names], fontsize=8); ax.set_ylabel(yl); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / (fn + '.png'), dpi=200); fig.savefig(FIG / (fn + '.pdf')); plt.close(fig)
disc(lab, 'fig3_disc_it', 'solo adattatore corretto', 'solo base corretto', '% degli item (conteggio sopra)')
disc(labE, 'fig3_disc_en', 'only adapter correct', 'only base correct', '% of items (count above)')

# dev final-886 forest (non independent)
D = R['dev_final886']; dn = ['medmcqa', 'medqa', 'mmlu_med', 'pubmedqa', 'medical_pool', 'arc_it']
for tag, L, ttl, xl in (('it', lab, 'DEV, checkpoint finale 886\n(non indipendente: il dev ha guidato le scelte)', r'$\Delta$ accuratezza (pp)'), ('en', labE, 'DEV, final checkpoint 886\n(not independent: dev guided the decisions)', r'$\Delta$ accuracy (pp)')):
    fig, ax = plt.subplots(figsize=(6.4, 3.8)); y = np.arange(len(dn))[::-1]
    d = np.array([D[n]['delta_pp'] for n in dn]); lo = np.array([D[n]['ci95_pp'][0] for n in dn]); hi = np.array([D[n]['ci95_pp'][1] for n in dn])
    ax.errorbar(d, y, xerr=[d - lo, hi - d], fmt='D', color='#7a4f9a', capsize=3, ms=5)
    ax.axvline(0, color='k', lw=.8); ax.set_yticks(y); ax.set_yticklabels([L[n].replace('(n=3766)', '(n=3779)') for n in dn]); ax.set_xlabel(xl); ax.set_title(ttl, fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / ('fig4_dev886_%s.png' % tag), dpi=200); fig.savefig(FIG / ('fig4_dev886_%s.pdf' % tag)); plt.close(fig)

def fp(p): return '$<$0.001' if p < 1e-3 else '%.3f' % p
def sg(x): return ('+%.2f' % x) if x >= 0 else ('$-$%.2f' % abs(x))
def row(n, sc, L):
    r = S[sc][n]; return '%s & %d & %.2f & %.2f & %s & [%s, %s] & %d/%d & %s \\\\' % (L[n].replace('\n', ' '), r['n'], r['base_acc'], r['cand_acc'], sg(r['delta_pp']), sg(r['ci95_pp'][0]), sg(r['ci95_pp'][1]), r['only_model'], r['only_base'], fp(r['mcnemar_p']))
for tag, L in (('it', lab), ('en', labE)):
    for sc in ('050', '075'):
        with open(FIG.parent / 'paper' / ('tab_sealed_%s_%s.tex' % (sc, tag)), 'w') as f:
            f.write('\n'.join(row(n, sc, L) if n != 'medical_pool' else '\\midrule\n' + row(n, sc, L) for n in names if n != 'arc_it') + '\n\\midrule\n' + row('arc_it', sc, L) + '\n')
    with open(FIG.parent / 'paper' / ('tab_dev_%s.tex' % tag), 'w') as f:
        for n in dn:
            r = D[n]; nn = r['n']; f.write('%s & %d & %.2f & %.2f & %s & [%s, %s] & %d/%d \\\\\n' % (L[n].replace('\n', ' ').replace('(n=3766)', ''), nn, r['base_acc'], r['cand_acc'], sg(r['delta_pp']), sg(r['ci95_pp'][0]), sg(r['ci95_pp'][1]), r['only_model'], r['only_base']))
print('ok')
