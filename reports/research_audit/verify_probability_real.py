"""Independent raw count check for the real-photo panel; no model calls."""
from pathlib import Path
import json,hashlib,datetime
P=Path('experiments/answer_probability_real_panel_v1');E=Path('reports/research_audit/evidence')
plan={d['id']:d for l in (P/'plan.jsonl').read_text().splitlines() if (d:=json.loads(l))};saved=json.load(open(P/'results.json'));checks=[];hashes={}
def ck(state,key,a,b):checks.append(dict(state=state,metric=key,recomputed=a,stored=b,passed=a==b))
for state,summary in saved['results'].items():
 path=P/'predictions'/state/'observations.jsonl';raw=path.read_bytes();hashes[str(path)]=hashlib.sha256(raw).hexdigest();rr=[json.loads(l) for l in raw.splitlines()];obs={r['id']:r for r in rr}
 ck(state,'unique_condition_coverage',len(obs),len(plan));ck(state,'row_count',len(rr),len(plan))
 for s in summary:
  values=[(item,obs[item['id']]) for item in plan.values() if item['image_id']!='noimage' and item['attribute']==s['attribute'] and item['format']==s['format']]
  correct=semantic=prefix=complete=0
  for item,r in values:
   ref=item[item['attribute']+'_reference']
   correct+=sum(x['answer']==ref for x in r['samples']);semantic+=sum(x['semantic_answer']==ref for x in r['samples'])
   prefix+=max(r['scores'],key=lambda k:r['scores'][k]['prefix_logprob'])==ref
   complete+=max(r['scores'],key=lambda k:r['scores'][k]['complete_logprob'])==ref
  for field,value in [('qualified_conditions',len(values)),('sampled_correct',correct),('sampled_semantic_correct',semantic),('raw_prefix_correct',prefix),('raw_complete_correct',complete)]:ck(state,s['attribute']+'/'+s['format']+'/'+field,value,s[field])
(E/'probability_real_checks.json').write_text(json.dumps(dict(snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,total=len(checks),failed=sum(not c['passed'] for c in checks),source_hashes=hashes,results_sha256=hashlib.sha256((P/'results.json').read_bytes()).hexdigest(),scope='Raw counts only; auxiliary semantics are saved machine labels, not new human review.'),ensure_ascii=False,indent=2));print('real-panel checks',len(checks),'failures',sum(not c['passed'] for c in checks))
