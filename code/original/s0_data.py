# S0: scarica eval e sorgenti di training (licenza commerciale) su D:, split dev/test disgiunti, test IT pool, decontaminazione.
import os, sys, json, random, itertools, time
from common import *
from datasets import load_dataset
L = 'ABCDE'
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)
# ---------- A. eval: dev (ablation + gate) / test (sigillato, usato una sola volta) ----------
def eval_sets():
    if done('eval'): return
    out = {}
    ds = load_dataset('openlifescienceai/medmcqa', split='validation')   # Apache-2.0
    out['medmcqa'] = [{'id': f"medmcqa-{d['id']}", 'q': d['question'], 'ch': [d['opa'], d['opb'], d['opc'], d['opd']], 'y': d['cop'], 'lang': 'en'} for d in ds]
    ds = load_dataset('GBaker/MedQA-USMLE-4-options', split='test')     # CC BY 4.0
    out['medqa'] = [{'id': f'medqa-{i}', 'q': d['question'], 'ch': [d['options'][k] for k in 'ABCD'], 'y': 'ABCD'.index(d['answer_idx']), 'lang': 'en'} for i, d in enumerate(ds)]
    rows = []
    for s in ['anatomy', 'clinical_knowledge', 'college_medicine', 'medical_genetics', 'professional_medicine', 'college_biology']:   # MIT
        for i, d in enumerate(load_dataset('cais/mmlu', s, split='test')):
            rows.append({'id': f'mmlu-{s}-{i}', 'q': d['question'], 'ch': d['choices'], 'y': d['answer'], 'lang': 'en', 'sub': s})
    out['mmlu_med'] = rows
    ds = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', split='train')  # MIT; solo domanda, senza contesto (come le eval precedenti)
    out['pubmedqa'] = [{'id': f"pubmedqa-{d['pubid']}", 'q': d['question'], 'ch': ['yes', 'no', 'maybe'], 'y': ['yes', 'no', 'maybe'].index(d['final_decision']), 'lang': 'en'} for d in ds]
    ds = load_dataset('sapienzanlp/arc_italian', 'arc_challenge', split='test')   # CC BY-SA 4.0
    out['arc_it'] = [{'id': f'arcit-{i}', 'q': d['input_translation'], 'ch': d['choices_translation'], 'y': d['label'], 'lang': 'it'} for i, d in enumerate(ds)]
    ds = load_dataset('openai/gsm8k', 'main', split='test')              # MIT
    out['gsm8k'] = [{'id': f'gsm-{i}', 'q': d['question'], 'a': d['answer'].split('####')[-1].strip().replace(',', ''), 'lang': 'en'} for i, d in enumerate(ds)]
    ds = load_dataset('google/IFEval', split='train')                     # Apache-2.0
    out['ifeval'] = [{'id': f"ifeval-{d['key']}", 'q': d['prompt'], 'raw': dict(d), 'lang': 'en'} for d in ds]
    os.makedirs(os.path.join(P['eval'], 'dev'), exist_ok=True); os.makedirs(os.path.join(P['eval'], 'sealed_test'), exist_ok=True)
    info = {}
    for k, rows in out.items():
        dv = [r for r in rows if split_of(r['id']) == 'dev']; ts = [r for r in rows if split_of(r['id']) == 'test']
        wjsonl(os.path.join(P['eval'], 'dev', k + '.jsonl'), dv); wjsonl(os.path.join(P['eval'], 'sealed_test', k + '.jsonl'), ts)
        info[k] = {'dev': len(dv), 'test': len(ts)}; log('eval', k, info[k])
    with open(os.path.join(P['eval'], 'sealed_test', 'README.txt'), 'w') as f:
        f.write('TEST SIGILLATO: da usare UNA SOLA VOLTA sul checkpoint finale. Mai per scelte di dati, iperparametri o gate.\n')
    mark('eval', info)
# ---------- B. sorgenti di training ----------
MED_IT = ['malatti', 'sindrom', 'patolog', 'terapi', 'farmac', 'diagnos', 'sintom', 'chirurg', 'clinic', 'medic', 'tumor', 'carcinom', 'infezion', 'virus', 'batteri',
          'anatomi', 'organo', 'tessut', 'cellul', 'ormon', 'enzim', 'cardi', 'polmon', 'rene', 'renal', 'fegat', 'epat', 'neurolog', 'cerebr', 'psichiatr', 'pediatr',
          'ginecolog', 'ostetric', 'dermat', 'oftalm', 'ortoped', 'oncolog', 'ematolog', 'endocrin', 'diabet', 'vaccin', 'antibiot', 'ospedal', 'paziente', 'dose', 'arteri', 'ven', 'muscol', 'osso', 'ossa']
MED_EN = ['disease', 'syndrome', 'patholog', 'therap', 'drug', 'diagnos', 'symptom', 'surg', 'clinical', 'medic', 'tumor', 'tumour', 'carcinoma', 'infection', 'virus', 'bacteri',
          'anatom', 'organ', 'tissue', 'cell', 'hormone', 'enzyme', 'cardi', 'pulmonary', 'lung', 'kidney', 'renal', 'liver', 'hepat', 'neurolog', 'cerebr', 'psychiatr', 'pediatr',
          'gynec', 'obstetr', 'dermat', 'ophthalm', 'orthop', 'oncolog', 'hematolog', 'endocrin', 'diabet', 'vaccin', 'antibiotic', 'hospital', 'patient', 'dosage', 'arter', 'muscle', 'bone']
def is_med(text, kws, need=6):
    t = text[:2000].lower(); return sum(1 for k in kws if k in t) >= need
def dump(name, it, cap, fn):
    p = os.path.join(P['raw'], name + '.jsonl')
    if done('raw_' + name): return
    n = 0; t0 = time.time()
    with open(p + '.tmp', 'w', encoding='utf-8') as f:
        for d in it:
            r = fn(d)
            if r is None: continue
            f.write(json.dumps(r, ensure_ascii=False) + '\n'); n += 1
            if n % 20000 == 0: log(name, n, round(time.time() - t0))
            if n >= cap: break
    os.replace(p + '.tmp', p); mark('raw_' + name, {'docs': n}); log('raw', name, n)
def wiki_it():
    if done('raw_wiki_it_med') and done('raw_wiki_it_gen'): return
    ds = load_dataset('wikimedia/wikipedia', '20231101.it', split='train')   # CC BY-SA 3.0 / GFDL
    med, gen, held = [], [], []; rnd = random.Random(7)
    for d in ds:
        t = d['text']
        if len(t) < 800: continue
        r = {'text': d['title'] + '\n\n' + t[:20000], 'src': 'wiki_it', 'title': d['title'], 'url': d['url']}
        if is_med(t, MED_IT):
            (held if h('held|' + d['title']) % 10 == 0 else med).append(r)
        elif len(gen) < 300000 and rnd.random() < 0.35: gen.append(r)
    wjsonl(os.path.join(P['raw'], 'wiki_it_med.jsonl'), med); mark('raw_wiki_it_med', {'docs': len(med)})
    wjsonl(os.path.join(P['raw'], 'wiki_it_gen.jsonl'), gen); mark('raw_wiki_it_gen', {'docs': len(gen)})
    wjsonl(os.path.join(P['eval'], 'heldout_med_it_pool.jsonl'), held)   # mai in training: sorgente del test IT multi-specialita
    log('wiki_it med', len(med), 'gen', len(gen), 'heldout pool', len(held))
def sources():
    wiki_it()
    dump('wiki_en_med', load_dataset('wikimedia/wikipedia', '20231101.en', split='train', streaming=True), 120000,
         lambda d: {'text': d['title'] + '\n\n' + d['text'][:20000], 'src': 'wiki_en_med', 'title': d['title'], 'url': d['url']} if len(d['text']) > 1500 and is_med(d['text'], MED_EN, 8) else None)
    dump('fineweb2_it', load_dataset('HuggingFaceFW/fineweb-2', 'ita_Latn', split='train', streaming=True), 600000,   # ODC-By
         lambda d: {'text': d['text'][:20000], 'src': 'fineweb2_it', 'url': d.get('url')} if len(d['text']) > 500 else None)
    try:
        dump('wikisource_it', load_dataset('wikimedia/wikisource', '20231201.it', split='train', streaming=True), 30000,   # CC BY-SA
             lambda d: {'text': d['title'] + '\n\n' + d['text'][:20000], 'src': 'wikisource_it', 'title': d['title'], 'url': d.get('url')} if len(d['text']) > 800 else None)
    except Exception as e: log('wikisource skipped', e)
    dump('openmath2', load_dataset('nvidia/OpenMathInstruct-2', split='train_1M', streaming=True), 150000,   # CC BY 4.0 (Llama 3.1)
         lambda d: {'messages': [{'role': 'user', 'content': d['problem']}, {'role': 'assistant', 'content': d['generated_solution']}], 'src': 'openmath2'})
    dump('finemath', load_dataset('HuggingFaceTB/finemath', 'finemath-4plus', split='train', streaming=True), 80000,   # ODC-By
         lambda d: {'text': d['text'][:20000], 'src': 'finemath'} if len(d['text']) > 500 else None)
# ---------- C. decontaminazione (10-gram) contro TUTTE le eval (dev+test) ----------
def decontam():
    if done('decontam'): return
    G = set()
    for sub in ['dev', 'sealed_test']:
        for fn in os.listdir(os.path.join(P['eval'], sub)):
            if not fn.endswith('.jsonl'): continue
            for r in rjsonl(os.path.join(P['eval'], sub, fn)):
                G |= ngrams(r['q'] + ' ' + ' '.join(map(str, r.get('ch', []))))
    held_titles = {r['title'] for r in rjsonl(os.path.join(P['eval'], 'heldout_med_it_pool.jsonl'))}
    for r in rjsonl(os.path.join(P['eval'], 'heldout_med_it_pool.jsonl')): G |= ngrams(r['text'][:6000], 13)
    rep = {'eval_ngrams': len(G)}
    for fn in sorted(os.listdir(P['raw'])):
        if not fn.endswith('.jsonl') or fn.endswith('.clean.jsonl'): continue
        src = fn[:-6]; kept = drop = 0; p = os.path.join(P['raw'], src + '.clean.jsonl')
        with open(p + '.tmp', 'w', encoding='utf-8') as f, open(os.path.join(P['raw'], fn), encoding='utf-8') as fi:
            for l in fi:
                r = json.loads(l); t = r['text'] if 'text' in r else ' '.join(m['content'] for m in r['messages'])
                if r.get('title') in held_titles or (ngrams(t) & G) or (ngrams(t[:6000], 13) & G): drop += 1; continue
                f.write(l); kept += 1
        os.replace(p + '.tmp', p); rep[src] = {'kept': kept, 'dropped': drop}; log('decontam', src, rep[src])
    mark('decontam', rep)
if __name__ == '__main__':
    eval_sets(); sources(); decontam(); log('S0 DATA DONE')
