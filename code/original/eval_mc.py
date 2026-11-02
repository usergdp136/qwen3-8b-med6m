# Eval MC a logprob della lettera (stesso metodo del baseline Qwen3-14B), split dev o test. Usabile in-process (gate) o da CLI.
import os, json, time, random, torch
from common import *
L = 'ABCDE'
PROMPT = {'it': '\nRispondi solo con la lettera della risposta corretta.', 'en': '\nAnswer with only the letter of the correct answer.'}
def load_items(split='dev', cap=300, sets=None):
    d = os.path.join(P['eval'], 'dev' if split == 'dev' else 'sealed_test'); out = {}
    for s in (sets or MED_SETS + GEN_SETS):
        p = os.path.join(d, s + '.jsonl')
        if not os.path.exists(p): continue
        rows = rjsonl(p); random.Random(1234).shuffle(rows); out[s] = rows[:cap] if cap else rows
    return out
@torch.no_grad()
def evaluate(model, tok, split='dev', cap=300, sets=None):
    was = model.training; model.eval(); dev = next(model.parameters()).device
    lid = [[tok.encode(' ' + c, add_special_tokens=False)[-1], tok.encode(c, add_special_tokens=False)[-1]] for c in L]
    res = {'split': split, 'cap': cap, 'sets': {}, 'items': {}}
    for s, rows in load_items(split, cap, sets).items():
        ok = 0; t0 = time.time()
        for r in rows:
            k = len(r['ch']); q = r['q'] + '\n' + '\n'.join(f'{L[i]}. {c}' for i, c in enumerate(r['ch'])) + PROMPT[r.get('lang', 'en')]
            txt = tok.apply_chat_template([{'role': 'user', 'content': q}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
            ids = tok(txt, return_tensors='pt', add_special_tokens=False).input_ids[:, -2048:].to(dev)
            lg = model(input_ids=ids).logits[0, -1].float()
            sc = [max(lg[a].item(), lg[b].item()) for a, b in lid[:k]]; p = max(range(k), key=lambda i: sc[i])
            res['items'][r['id']] = int(p == r['y']); ok += int(p == r['y'])
        res['sets'][s] = {'acc': round(100 * ok / max(1, len(rows)), 2), 'n': len(rows), 's': round(time.time() - t0)}
        print('eval', split, s, res['sets'][s], flush=True)
    med = [res['sets'][s]['acc'] for s in MED_SETS if s in res['sets']]
    res['med_mean'] = round(sum(med) / len(med), 2) if med else None
    if was: model.train()
    return res
def gate(res, base, tol=1.0, need_beat=True):
    """Gate dev: medico medio > base (se need_beat) e NESSUN sottoinsieme sotto base - tol."""
    why = []
    if need_beat and not (res['med_mean'] > base['med_mean']): why.append(f"med_mean {res['med_mean']} <= base {base['med_mean']}")
    for s, v in res['sets'].items():
        b = base['sets'].get(s)
        if b and v['acc'] < b['acc'] - tol: why.append(f"{s} {v['acc']} < base {b['acc']} - {tol}")
    return (not why), why
if __name__ == '__main__':   # python eval_mc.py OUT.json [adapter_dir|none] [dev|test] [cap]
    import sys
    from unsloth import FastLanguageModel
    out, ad = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else 'none'); split = sys.argv[3] if len(sys.argv) > 3 else 'dev'; cap = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    if split == 'test':
        lock = os.path.join(P['state'], 'TEST_USED.json')
        if os.path.exists(lock): raise SystemExit('Test sigillato gia usato: ' + open(lock).read())
    model, tok = FastLanguageModel.from_pretrained(ad if ad != 'none' else BASE, max_seq_length=2048, load_in_4bit=True)
    FastLanguageModel.for_inference(model)
    r = evaluate(model, tok, split, cap); r['adapter'] = ad
    json.dump(r, open(out, 'w'), indent=1)
    if split == 'test': json.dump({'adapter': ad, 'ts': time.strftime('%Y-%m-%d %H:%M')}, open(os.path.join(P['state'], 'TEST_USED.json'), 'w'))
    print('DONE', r['med_mean'], {k: v['acc'] for k, v in r['sets'].items()})
