# S1 v2r: mix di ripresa = ~78% medico (stesse 4 sorgenti/quote di S1_v2, nuovi seed) + ~22% replay generale (solo licenze commerciali, decontaminato 10-gram vs TUTTE le eval dev+test).
# Uso: python build_s1v2r.py [--probe]  -> scrive runs\S1_v2r\train.jsonl + MANIFEST.json
import os, sys, json, random, time
os.environ['HF_HOME'] = r'D:\hf_cache'
from common import *
from datasets import load_dataset
from transformers import AutoTokenizer
SEQ = 2048; STEPS = 886; GA = 16
TOTAL = int(STEPS * GA * SEQ * 1.015)                    # token effettivi (troncati a 2048) ~ 1 epoca + 1.5% margine
REPLAY = 0.22
RD = os.path.join(P['runs'], 'S1_v2r'); os.makedirs(RD, exist_ok=True)
tok = AutoTokenizer.from_pretrained(BASE)
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)
def rtext(r): return r['text'] if r.get('text') else tok.apply_chat_template(r['messages'], tokenize=False, enable_thinking=False)
def eff(r): return min(SEQ, len(tok(rtext(r), add_special_tokens=False).input_ids))
def plain(r): return r['text'] if r.get('text') else ' '.join(m['content'] for m in r['messages'])
# ---- n-gram eval (dev+test) ----
G = set()
for sub in ['dev', 'sealed_test']:
    for fn in os.listdir(os.path.join(P['eval'], sub)):
        if fn.endswith('.jsonl'):
            for r in rjsonl(os.path.join(P['eval'], sub, fn)): G |= ngrams(r['q'] + ' ' + ' '.join(map(str, r.get('ch', []))))
for r in rjsonl(os.path.join(P['eval'], 'heldout_med_it_pool.jsonl')): G |= ngrams(r['text'][:6000], 13)
EXQ = set()
for sub in ['dev', 'sealed_test']:
    for fn in os.listdir(os.path.join(P['eval'], sub)):
        if fn.endswith('.jsonl'):
            for r in rjsonl(os.path.join(P['eval'], sub, fn)): EXQ.add(' '.join(WRE.findall(r['q'].lower())))
log('eval ngrams', len(G), 'eval q', len(EXQ))
def contaminated(r):
    t = plain(r)
    if (ngrams(t) & G) or (ngrams(t[:6000], 13) & G): return True
    if r.get('messages') and ' '.join(WRE.findall(r['messages'][0]['content'].lower())) in EXQ: return True
    return False
def mm(*turns): return {'messages': [{'role': ('user' if i % 2 == 0 else 'assistant'), 'content': t} for i, t in enumerate(turns)]}
POOL = {}; STAT = {}
def finish(name, rows, lic):
    k = d = 0; out = []
    for r in rows:
        if contaminated(r): d += 1; continue
        out.append(r); k += 1
    STAT[name] = {'license': lic, 'raw_rows': len(rows), 'kept_after_decontam': k, 'dropped_contaminated': d}; POOL[name] = out; log(name, STAT[name])
# ---- oasst2 (Apache-2.0, human written; solo foglie rank 0, review ok, non synthetic) ----
oa = [r for sp in ('train', 'validation') for r in load_dataset('OpenAssistant/oasst2', split=sp)]
byid = {r['message_id']: r for r in oa}
def ok_leaf(r): return r['role'] == 'assistant' and r['rank'] == 0 and r['review_result'] is True and not r['deleted'] and not r.get('synthetic') and r['parent_id'] in byid
Q = [r for r in oa if ok_leaf(r)]
covered = set()
for r in Q:
    p = r['parent_id']
    while p in byid: covered.add(p); p = byid[p]['parent_id']
for lang, name in [('it', 'oasst2_it'), ('en', 'oasst2_en')]:
    rows = []
    for r in Q:
        if r['lang'] != lang or r['message_id'] in covered: continue
        chain = [r]; p = r['parent_id']
        while p in byid: chain.append(byid[p]); p = byid[p]['parent_id']
        chain = chain[::-1]
        if chain[0]['role'] != 'prompter' or any(c['role'] != ('prompter' if i % 2 == 0 else 'assistant') for i, c in enumerate(chain)): continue
        t = [c['text'] for c in chain]
        if len(t[-1]) < 40: continue
        row = mm(*t); row['src'] = name
        if eff(row) >= SEQ and len(tok(rtext(row)).input_ids) > SEQ: continue
        rows.append(row)
    finish(name, rows, 'Apache-2.0')
# ---- aya_dataset (Apache-2.0, annotatori umani) ----
ay = load_dataset('CohereLabs/aya_dataset', split='train')
for lang, name, minlen in [('Italian', 'aya_it', 20), ('English', 'aya_en', 120)]:
    rows = []
    for r in ay.filter(lambda x: x['language'] == lang):
        if len(r['targets']) < minlen or len(r['inputs']) < 10: continue
        row = mm(r['inputs'], r['targets']); row['src'] = name
        if len(tok(rtext(row)).input_ids) > SEQ: continue
        rows.append(row)
    finish(name, rows, 'Apache-2.0')
# ---- dolly-15k (CC BY-SA 3.0, scritto da dipendenti Databricks) ----
rows = []
for r in load_dataset('databricks/databricks-dolly-15k', split='train'):
    if len(r['response']) < 60: continue
    u = r['instruction'] + (('\n\n' + r['context']) if r['context'] else '')
    row = mm(u, r['response']); row['src'] = 'dolly_en'
    if len(tok(rtext(row)).input_ids) > SEQ: continue
    rows.append(row)
finish('dolly_en', rows, 'CC-BY-SA-3.0')
# ---- fineweb-edu (ODC-By 1.0): ~2500 doc di qualita (int_score>=3) in streaming ----
rows = []; fe = load_dataset('HuggingFaceFW/fineweb-edu', 'sample-10BT', split='train', streaming=True)
for r in fe:
    if r['int_score'] >= 3 and 1500 < len(r['text']) < 30000: rows.append({'text': r['text'][:7500], 'src': 'fineweb_edu'})
    if len(rows) >= 3500: break
finish('fineweb_edu', rows, 'ODC-By-1.0')
# ---- locali gia decontaminati (stesso filtro S0): wiki_it_gen (CC BY-SA 3.0/GFDL), fineweb2_it (ODC-By 1.0) ----
for s, lic in [('wiki_it_gen', 'CC-BY-SA-3.0/GFDL'), ('fineweb2_it', 'ODC-By-1.0')]:
    rows = rjsonl(os.path.join(P['raw'], s + '.clean.jsonl')); random.Random(77).shuffle(rows); rows = rows[:12000]
    for r in rows: r.pop('url', None); r.pop('title', None)
    POOL[s] = rows; STAT[s] = {'license': lic, 'raw_rows': len(rows), 'kept_after_decontam': len(rows), 'note': 'file .clean.jsonl gia decontaminato in S0 (10-gram vs eval dev+test)'}
# ---- selezione ----
M = json.load(open(os.path.join(P['code'] if 'code' in P else os.path.join(R, 'code'), 's1v2_mix.json'), encoding='utf-8'))['sources']
z = sum(M.values()); med_q = {s: v / z for s, v in M.items()}
med_tok = TOTAL * (1 - REPLAY); rep_tok = TOTAL * REPLAY
RSHARE = {'it_instruct': ('oasst2_it', 'aya_it'), 'en_instruct': ('oasst2_en', 'aya_en', 'dolly_en'), 'wiki_it_gen': ('wiki_it_gen',), 'fineweb_edu': ('fineweb_edu',), 'fineweb2_it': ('fineweb2_it',)}
RW = {'it_instruct': 0.155, 'en_instruct': 0.18, 'wiki_it_gen': 0.25, 'fineweb_edu': 0.25, 'fineweb2_it': 0.165}
def take(rows_list, target, seed, maxpass=2.0):
    """prende righe finche' target token effettivi; max maxpass passaggi (interi) sul pool; ritorna (righe, token, passaggi)."""
    pool = [r for rs in rows_list for r in rs]; out = []; got = 0; pool_tok = None
    for p in range(int(maxpass)):
        rr = pool[:]; random.Random(seed + 1000 * p).shuffle(rr); pt = 0
        for r in rr:
            if got >= target: break
            e = eff(r); out.append(r); got += e; pt += e
        if p == 0: pool_tok = pt
        if got >= target: break
    return out, got, (got / pool_tok if pool_tok else 0)
allr = []; man = {'_total_eff_tokens_target': int(TOTAL), '_steps_planned': STEPS, '_replay_share_target': REPLAY, 'medical': {}, 'replay': {}, 'sources_stats': STAT}
for i, (s, q) in enumerate(med_q.items()):
    rows = rjsonl(os.path.join(P['raw'], s + '.clean.jsonl')); rr, got, ps = take([rows], med_tok * q, 5000 + i)
    allr += rr; man['medical'][s] = {'quota': round(q, 4), 'docs': len(rr), 'eff_tokens': int(got), 'passes_of_pool_approx': round(ps, 2)}; log('med', s, man['medical'][s])
short = 0
for j, (grp, names) in enumerate(RSHARE.items()):
    tgt = rep_tok * RW[grp] + (short / 2 if grp in ('wiki_it_gen', 'fineweb2_it') else 0)
    rr, got, ps = take([POOL[n] for n in names], tgt, 9000 + j)
    if grp in ('it_instruct', 'en_instruct'): short += max(0, tgt - got)
    allr += rr; man['replay'][grp] = {'members': list(names), 'target_eff': int(tgt), 'docs': len(rr), 'eff_tokens': int(got), 'passes': round(ps, 2)}; log('replay', grp, man['replay'][grp])
for r in allr: r.setdefault('src', 'med')
tot = sum(v['eff_tokens'] for v in man['medical'].values()) + sum(v['eff_tokens'] for v in man['replay'].values())
rep = sum(v['eff_tokens'] for v in man['replay'].values())
man['total_eff_tokens'] = int(tot); man['replay_share_actual'] = round(rep / tot, 4); man['italian_share_note'] = 'wiki_it_med + wiki_it_gen + fineweb2_it + oasst2_it + aya_it'
rg = random.Random(8); rg.shuffle(allr)
for _ in range(10):
    bad = [k for k in range(1, len(allr)) if allr[k] == allr[k - 1]]
    if not bad: break
    for k in bad: j = rg.randrange(len(allr)); allr[k], allr[j] = allr[j], allr[k]
# schema uniforme minimo: solo messages/text/src
out = []
for r in allr: out.append({'messages': r.get('messages'), 'text': r.get('text'), 'src': r.get('src')})
wjsonl(os.path.join(RD, 'train.jsonl'), out)
json.dump(man, open(os.path.join(RD, 'MANIFEST.json'), 'w'), indent=1, ensure_ascii=False)
log('DONE docs', len(out), 'eff tokens', tot, 'replay share', man['replay_share_actual'])
