"""CPU-only recomputation of existing classification outputs and raw U answers.
Never rewrites source experiments or draws new model samples.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,pickle,hashlib,csv
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'reports/research_audit/evidence'
checks=[]
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def check(source,metric,actual,expected):
 ok=bool(np.isclose(actual,expected,atol=1e-10,rtol=1e-10))
 checks.append(dict(source=str(Path(source).relative_to(ROOT)),metric=metric,recomputed=float(actual),stored=float(expected),passed=ok))
def rd(c):
 p=np.asarray(c,dtype=float)/sum(c);return np.abs(p[:,None]-p[None,:]).sum()/(len(p)*(len(p)-1))
for relative in ['u_g_transfer_followup_v1/matched_evaluation_v1','u_g_transfer_followup_v1/adapter_ablation_v1']:
 panel=ROOT/'experiments'/relative; saved=read(panel/'generation_results.json')
 prompts=read(ROOT/'experiments/u_g_transfer_followup_v1/matched_evaluation_v1/generation_prompts.json')
 for state,result in saved['states'].items():
  path=panel/'classifications'/state/'test_results.pkl'
  # Ablation uses borrowed base/full state files registered in the model manifest.
  if not path.exists():
   source={'BASE':'BASE','U_full':'understanding_seed4101_epoch3','G_full':'generation_seed4101_epoch3'}.get(state)
   if source:path=ROOT/'experiments/u_g_transfer_followup_v1/matched_evaluation_v1/classifications'/source/'test_results.pkl'
  if not path.exists():continue
  with path.open('rb') as f:faces,boxes,gender,race,age=pickle.load(f)
  for style in ['portrait','work']:
   inds=[p['index'] for p in prompts if p['style']==style and p['condition']=='neutral']
   for dim,logits,k in [('gender',gender,2),('race',race,4)]:
    counts=[]
    for i in inds:
     mask=faces[i].cpu().numpy().reshape(-1).astype(bool); pred=logits[i].cpu().numpy().argmax(-1)
     counts.append(np.bincount(pred[mask],minlength=k))
    value=float(np.mean([rd(c) for c in counts])); target=result['neutral'][style][dim]
    check(path,state+'/'+style+'/'+dim+'/RD',value,target['rd']);check(path,state+'/'+style+'/'+dim+'/detected',sum(sum(c) for c in counts),target['detected'])
for kind in ['paper_g_evaluation_v3','own_g_evaluation_v3']:
 panel=ROOT/'experiments/paper_sft_replication_v2'/kind
 for dim,k in [('gender',2),('race',4)]:
  saved=read(panel/('results_'+dim+'.json'))
  for state,result in saved['states'].items():
   path=panel/'classifications'/state/'test_results.pkl'
   with path.open('rb') as f:faces,boxes,gender,race,age=pickle.load(f)
   logits=gender if dim=='gender' else race;counts=[]
   for i in sorted(faces):
    mask=faces[i].cpu().numpy().reshape(-1).astype(bool);pred=logits[i].cpu().numpy().argmax(-1)
    counts.append(np.bincount(pred[mask],minlength=k))
   check(path,dim+'/RD',np.mean([rd(c) for c in counts]),result['rd']);check(path,dim+'/detected',sum(sum(c) for c in counts),result['detected_faces'])
panel=ROOT/'experiments/u_g_transfer_followup_v1/matched_evaluation_v1'
for name in ['sispi','paired']:
 plan=rows(panel/(name+'_plan.jsonl'));saved=read(panel/(name+'_results.json'))
 if name=='paired':
  q={r['id'] for r in read(ROOT/'experiments/u_g_transfer_followup_v1/paired_visual_v1/qualification.json') if r['sufficient_evidence']};selected=[r for r in plan if r['pair_id'] in q]
 else:selected=plan
 for state,r in saved['states'].items():
  path=panel/name/state/'observations.jsonl';obs=rows(path);pred={v['id']:v for v in obs}
  check(path,'raw coverage',len(pred),len(plan));check(path,'correct/selected',sum(pred[v['id']].get('answer')==v['expected'] for v in selected)/len(selected),r['accuracy'])
# Validate exported table against JSON, so transcription is independently checked.
with (panel/'全部检查点结果.csv').open(encoding='utf-8-sig') as f:
 for row in csv.DictReader(f):
  state=row['状态'];g=read(panel/'generation_results.json')['states'][state]
  for style,cn in [('portrait','肖像性别RD'),('work','工作性别RD')]:check(panel/'全部检查点结果.csv',state+'/'+cn,float(row[cn]),g['neutral'][style]['gender']['rd'])
(OUT/'recomputed_checks.json').write_text(json.dumps({'checks':checks,'total':len(checks),'failed':sum(not c['passed'] for c in checks),'scope':'Point estimates and raw coverage only; no new bootstrap, training, generation or classifier inference.'},ensure_ascii=False,indent=2))
print('checks',len(checks),'failed',sum(not c['passed'] for c in checks))
