# S0 v2 (25/10/2026): nuove sorgenti mediche Q&A/istruzioni con licenza commerciale + decontaminazione contro TUTTE le eval (dev+test sigillato).
# medmcqa_train: openlifescienceai/medmcqa split train (Apache-2.0; eval usa validation -> disgiunto)
# pubmedqa_art : qiaojin/PubMedQA pqa_artificial (MIT; eval usa pqa_labeled -> disgiunto, controllo pubid)
# medical_o1   : FreedomIntelligence/medical-o1-reasoning-SFT 'en' (Apache-2.0; risposte generate con GPT-4o)
import os, json, time, re
from common import *
from datasets import load_dataset
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)
L = 'ABCD'
def norm(t): return ' '.join(WRE.findall((t or '').lower()))
def mm(u, a): return {'messages': [{'role': 'user', 'content': u}, {'role': 'assistant', 'content': a}]}
def raw(name, it, cap, fn):
    p = os.path.join(P['raw'], name + '.jsonl')
    if done('raw_' + name): return
    n = 0
    with open(p + '.tmp', 'w', encoding='utf-8') as f:
        for d in it:
            r = fn(d)
            if r is None: continue
            f.write(json.dumps(r, ensure_ascii=False) + '\n'); n += 1
            if n >= cap: break
    os.replace(p + '.tmp', p); mark('raw_' + name, {'docs': n}); log('raw', name, n)
def f_mcqa(d):
    if not d.get('exp') or len(d['exp']) < 40 or d.get('choice_type') != 'single': return None
    ch = [d['opa'], d['opb'], d['opc'], d['opd']]; y = d['cop']
    q = d['question'] + '\n' + '\n'.join(f'{L[i]}. {c}' for i, c in enumerate(ch))
    return {**mm(q, f"{L[y]}. {ch[y]}\n\n{d['exp'].strip()}"), 'qkey': d['question']}
LAB = set()
def f_pqa(d):
    if str(d.get('pubid')) in LAB: return None
    ctx = '\n'.join(d['context']['contexts'])[:3000]
    return {**mm(f"{ctx}\n\nQuestion: {d['question']}", f"{d['long_answer'].strip()}\n\nAnswer: {d['final_decision']}"), 'qkey': d['question']}
def f_o1(d): return {**mm(d['Question'], d['Response'].strip()), 'qkey': d['Question']}
def main():
    global LAB
    LAB = {str(x['pubid']) for x in load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')}
    raw('medmcqa_train', load_dataset('openlifescienceai/medmcqa', split='train'), 60000, f_mcqa)
    raw('pubmedqa_art', load_dataset('qiaojin/PubMedQA', 'pqa_artificial', split='train'), 40000, f_pqa)
    raw('medical_o1', load_dataset('FreedomIntelligence/medical-o1-reasoning-SFT', 'en', split='train'), 30000, f_o1)
    if done('decontam_v2'): return
    G10, G8, EX = set(), set(), set()
    for sub in ['dev', 'sealed_test']:
        for fn in os.listdir(os.path.join(P['eval'], sub)):
            if not fn.endswith('.jsonl'): continue
            for r in rjsonl(os.path.join(P['eval'], sub, fn)):
                G10 |= ngrams(r['q'] + ' ' + ' '.join(map(str, r.get('ch', [])))); G8 |= ngrams(r['q'], 8); EX.add(norm(r['q']))
    for r in rjsonl(os.path.join(P['eval'], 'heldout_med_it_pool.jsonl')): G10 |= ngrams(r['text'][:6000], 13)
    rep = {'eval_10g': len(G10), 'eval_q8g': len(G8), 'eval_q_exact': len(EX)}
    for src in ['medmcqa_train', 'pubmedqa_art', 'medical_o1']:
        kept = drop = 0; p = os.path.join(P['raw'], src + '.clean.jsonl')
        with open(p + '.tmp', 'w', encoding='utf-8') as f, open(os.path.join(P['raw'], src + '.jsonl'), encoding='utf-8') as fi:
            for l in fi:
                r = json.loads(l); t = ' '.join(m['content'] for m in r['messages'])
                if norm(r['qkey']) in EX or (ngrams(r['qkey'], 8) & G8) or (ngrams(t) & G10) or (ngrams(t[:6000], 13) & G10): drop += 1; continue
                r.pop('qkey'); f.write(json.dumps(r, ensure_ascii=False) + '\n'); kept += 1
        os.replace(p + '.tmp', p); rep[src] = {'kept': kept, 'dropped': drop}; log('decontam', src, rep[src])
    mark('decontam_v2', rep)
if __name__ == '__main__': main(); log('S0b DATA DONE')
