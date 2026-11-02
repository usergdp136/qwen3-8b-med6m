import json,re,pickle,glob,os
import xml.etree.ElementTree as ET
from datasets import load_dataset
E=pickle.load(open('/workspace/med6m_new/evalgrams.pkl','rb')); G,Q=E['G'],E['Q']
WRE=re.compile(r'\w+',re.U)
def grams(t,n=13):
    w=WRE.findall(t.lower()); return {hash(' '.join(w[i:i+n])) for i in range(max(0,len(w)-n+1))}
def mm(u,a): return {'messages':[{'role':'user','content':u},{'role':'assistant','content':a}]}
def write(name,rows):
    st={'docs_in':len(rows),'contam13':0,'exact_q':0,'dup':0,'kept':0,'chars':0}; seen=set()
    with open(f'/workspace/med6m_new/out/{name}.clean.jsonl','w',encoding='utf-8') as f:
        for r,q in rows:
            t=' '.join(m['content'] for m in r['messages']); k=' '.join(WRE.findall(t.lower()))
            if k in seen: st['dup']+=1; continue
            seen.add(k)
            if ' '.join(WRE.findall(q.lower()))[:200] in Q: st['exact_q']+=1; continue
            if grams(t)&G: st['contam13']+=1; continue
            f.write(json.dumps(r,ensure_ascii=False)+'\n'); st['kept']+=1; st['chars']+=len(t)
    st['est_tokens']=int(st['chars']/3.5); json.dump(st,open(f'/workspace/med6m_new/out/{name}.stats.json','w'),indent=1); print(name,st)
# MedQuAD (CC BY 4.0): solo coppie con risposta (sottoinsiemi MedlinePlus 10-12 senza risposte)
rows=[]
for p in sorted(glob.glob('/workspace/med6m_new/MedQuAD-master/*/*.xml')):
    try: x=ET.parse(p).getroot()
    except Exception: continue
    focus=(x.findtext('Focus') or '').strip()
    for qa in x.iter('QAPair'):
        q=(qa.findtext('Question') or '').strip(); a=re.sub(r'\s+\n','\n',(qa.findtext('Answer') or '')).strip()
        if q and len(a)>=80: rows.append((mm(q,a),q))
write('medquad',rows)
# MedQA USMLE (jind11/MedQA, MIT; mirror openlifescienceai/medqa): SOLO train, 4 opzioni, nessun libro
L='ABCD'; rows=[]
for x in load_dataset('openlifescienceai/medqa',split='train'):
    d=x['data']; o=d['Options']; ks=sorted(o)
    q=d['Question']+'\n'+'\n'.join(f'{k}. {o[k]}' for k in ks); y=d['Correct Option']
    rows.append((mm(q,f"{y}. {o[y]}"),d['Question']))
write('medqa_train',rows)
