# Piano Everest 6 mesi - comune. Solo dati con licenza commerciale. Nessuna pubblicazione.
import os, json, hashlib, re
R = r'D:\tuning\med6m'
P = {k: os.path.join(R, k) for k in ['raw', 'eval', 'mix', 'runs', 'logs', 'state']}
for v in P.values(): os.makedirs(v, exist_ok=True)
BASE = 'unsloth/Qwen3-8B-unsloth-bnb-4bit'   # Qwen/Qwen3-8B, Apache-2.0
def h(s): return int(hashlib.sha1(s.encode('utf-8')).hexdigest(), 16)
def split_of(key): return 'test' if h('split|' + key) % 2 else 'dev'   # dev e test disgiunti, deterministici
def wjsonl(p, rows):
    with open(p + '.tmp', 'w', encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(p + '.tmp', p)
def rjsonl(p):
    with open(p, encoding='utf-8') as f: return [json.loads(l) for l in f if l.strip()]
def done(m): return os.path.exists(os.path.join(P['state'], m + '.done'))
def mark(m, info=None):
    with open(os.path.join(P['state'], m + '.done'), 'w', encoding='utf-8') as f: json.dump(info or {}, f, ensure_ascii=False)
WRE = re.compile(r'\w+', re.U)
def ngrams(t, n=10):
    w = WRE.findall(t.lower()); return {hash(' '.join(w[i:i + n])) for i in range(max(0, len(w) - n + 1))}
MED_SETS = ['medmcqa', 'medqa', 'mmlu_med', 'pubmedqa', 'it_med']   # it_med = test IT multi-specialita (quando generato)
GEN_SETS = ['arc_it']
