# [S1_v2r] copia di train_stage.py con: warmup_steps da cfg, stampa parametri addestrabili, init_adapter. Originale train_stage.py NON modificato.
# Training di uno stadio (o di un'ablation) Qwen3-8B QLoRA su T4. Resumabile, guard termico, gate dev a meta run con stop + checkpoint.
import os, sys, json, time, glob, subprocess
from common import *
cfg = json.load(open(sys.argv[1])); out = cfg['out_dir']; os.makedirs(out, exist_ok=True)
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
import eval_mc
model, tok = FastLanguageModel.from_pretrained(cfg.get('init_adapter') or BASE, max_seq_length=cfg['max_seq'], load_in_4bit=True, dtype=None)
if not cfg.get('init_adapter'):
    model = FastLanguageModel.get_peft_model(model, r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'], lora_dropout=0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], use_gradient_checkpointing='unsloth', random_state=cfg['seed'])
print('TRAINABLE_PARAMS', sum(p.numel() for p in model.parameters() if p.requires_grad), 'INIT_ADAPTER', cfg.get('init_adapter'), flush=True)
ds = load_dataset('json', data_files={'train': cfg['train']})['train']
def fmt(b):
    t = []
    for x, m in zip(b['text'], b['messages']):
        t.append(x if x else tok.apply_chat_template(m, tokenize=False, enable_thinking=False))
    return {'text': t}
if 'messages' not in ds.column_names: ds = ds.add_column('messages', [None] * len(ds))
if 'text' not in ds.column_names: ds = ds.add_column('text', [None] * len(ds))
ds = ds.map(fmt, batched=True, remove_columns=[c for c in ds.column_names if c != 'text'])
base = json.load(open(cfg['base_dev'])) if cfg.get('base_dev') else None
def t4():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=name,temperature.gpu,power.draw,power.limit', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20).stdout
        for l in o.splitlines():
            n, t, p, pl = [x.strip() for x in l.split(',')]
            if 'T4' in n and 'T400' not in n: return float(t), p, float(pl)
    except Exception: pass
    return None, None, None
class CB(TrainerCallback):
    def __init__(s): s.gates = set()
    def on_log(s, a, st, c, logs=None, **k):
        t, p, pl = t4()
        with open(os.path.join(out, 'metrics.jsonl'), 'a') as f: f.write(json.dumps({'step': st.global_step, 'max_steps': st.max_steps, 'ts': time.time(), 't4_temp': t, 't4_w': p, 't4_pl': pl, **(logs or {})}) + '\n')
    def on_step_end(s, a, st, c, **k):
        stop = any(os.path.exists(os.path.join(P['state'], m)) for m in ['STOP', 'HEAT_STOP'])
        if st.global_step % 10 == 0:
            t, p, pl = t4()
            if t is not None and t > 85: stop = True; open(os.path.join(P['state'], 'HEAT_STOP'), 'w').write(f'{time.ctime()} T4 {t}C')
            if pl is not None and pl > 61: stop = True; open(os.path.join(P['state'], 'POWER_STOP'), 'w').write(f'{time.ctime()} limit {pl}W')
        if False and cfg.get('deadline_utc') and time.time() > cfg['deadline_utc']:
            stop = True; open(os.path.join(P['state'], 'STOP'), 'w').write(f'{time.ctime()} deadline T4 (abliteration 29/10)')
        if stop: c.should_save = True; c.should_training_stop = True; print('STOP richiesto (heat/power/STOP file): salvo e mi fermo', flush=True); return
        for fr in cfg.get('gate_fracs', []):
            gs = int(st.max_steps * fr)
            if base and st.global_step == gs and gs not in s.gates:
                s.gates.add(gs); r = eval_mc.evaluate(model, tok, 'dev', cfg.get('gate_cap', 200))
                if cfg.get('gate_mode') == 'se':
                    import gate_se; ok, why, _rep = gate_se.gate_se(r, base, cfg.get('k', 2.0), need_med_gain=cfg.get('mid_need_gain', True))
                else: ok, why = eval_mc.gate(r, base, cfg.get('tol', 1.0), need_beat=True)
                rec = {'step': gs, 'ok': ok, 'why': why, 'med_mean': r['med_mean'], 'base_med_mean': base['med_mean'], 'sets': r['sets'], 'ts': time.ctime()}
                with open(os.path.join(out, 'gates.jsonl'), 'a') as f: f.write(json.dumps(rec) + '\n')
                if not ok:
                    json.dump(rec, open(os.path.join(out, 'GATE_FAIL.json'), 'w'), indent=1); c.should_save = True; c.should_training_stop = True
                    print('GATE FALLITO a meta run:', why, flush=True)
fp16 = not torch.cuda.is_bf16_supported()
tr = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, callbacks=[CB()],
    args=SFTConfig(output_dir=out, dataset_text_field='text', max_seq_length=cfg['max_seq'], packing=True,
        per_device_train_batch_size=1, gradient_accumulation_steps=cfg['grad_accum'], max_steps=cfg['max_steps'], learning_rate=cfg['lr'],
        lr_scheduler_type='cosine', **({'warmup_steps': cfg['warmup_steps']} if cfg.get('warmup_steps') else {'warmup_ratio': cfg.get('warmup_ratio', 0.03)}), optim='adamw_8bit', weight_decay=0.0, fp16=fp16, bf16=not fp16, logging_steps=cfg.get('logging_steps', 5),
        save_steps=cfg.get('save_steps', 240), save_total_limit=cfg.get('save_total_limit', 4), seed=cfg['seed'], report_to='none', dataset_num_proc=4))
c = sorted(glob.glob(os.path.join(out, 'checkpoint-*')), key=lambda p: int(p.rsplit('-', 1)[1])); r = c[-1] if c else None
print('RESUME FROM', r, flush=True)
tr.train(resume_from_checkpoint=r)
model.save_pretrained(os.path.join(out, 'adapter')); tok.save_pretrained(os.path.join(out, 'adapter'))
fin = 'stopped' if (os.path.exists(os.path.join(out, 'GATE_FAIL.json')) or os.path.exists(os.path.join(P['state'], 'HEAT_STOP')) or os.path.exists(os.path.join(P['state'], 'POWER_STOP')) or os.path.exists(os.path.join(P['state'], 'STOP'))) else 'complete'
if cfg.get('final_dev_eval') and fin == 'complete':
    rr = eval_mc.evaluate(model, tok, 'dev', cfg.get('gate_cap', 200))
    if cfg.get('gate_mode') == 'se' and base:   # [31/10/2026] gate finale con gate_se (bootstrap appaiato) sul dev intero (gate_cap=0)
        import gate_se; fok, fwhy, frep = gate_se.gate_se(rr, base, cfg.get('k', 2.0), need_med_gain=cfg.get('mid_need_gain', True))
        rr['gate_se'] = {'ok': fok, 'why': fwhy, 'rep': frep}
        with open(os.path.join(out, 'gates.jsonl'), 'a') as f: f.write(json.dumps({'final': True, 'step': tr.state.global_step, 'ok': fok, 'why': fwhy, 'rep': frep, 'med_mean': rr['med_mean'], 'sets': rr['sets'], 'ts': time.ctime()}) + '\n')
        print('GATE FINALE gate_se:', fok, fwhy, flush=True)
    json.dump(rr, open(os.path.join(out, 'dev_eval.json'), 'w'), indent=1)
open(os.path.join(out, 'FINISHED'), 'w').write(fin); print('FINISHED', fin, flush=True)

