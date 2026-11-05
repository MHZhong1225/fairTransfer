"""Read-only source inventory; writes audit derivatives only. Run from project root."""
import os,json,hashlib,re,datetime
from pathlib import Path
OUT=Path('reports/research_audit/evidence'); OUT.mkdir(parents=True,exist_ok=True)
SKIP={'site-packages','.git','__pycache__','environment','evaluator_environment','evaluator_environment_v2','author_reference_environment_v1','downloads_direct','upstream','frozen_code','code','references','third_party'}
files=[]; texts={}
for base in ['docs','experiments','scripts']:
 for root,ds,fs in os.walk(base,followlinks=False):
  ds[:]=[d for d in ds if d not in SKIP and not d.endswith('.egg-info')]
  for f in fs:
   p=Path(root)/f
   if p.suffix.lower() not in {'.md','.json','.jsonl','.csv','.log','.yaml','.yml','.py','.sh','.pid'}: continue
   s=p.stat(); rec={'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'suffix':p.suffix,'symlink':p.is_symlink()}
   if p.suffix=='.md' or base=='docs' or (p.suffix in {'.json','.yaml','.yml','.py','.sh'} and s.st_size<2000000):
    raw=p.read_bytes(); rec['sha256']=hashlib.sha256(raw).hexdigest()
    txt=raw.decode('utf-8',errors='replace'); texts[str(p)]=txt
    if p.suffix=='.json':
     try:
      obj=json.loads(txt);rec['shape']=list(obj)[:40] if isinstance(obj,dict) else {'length':len(obj)}
     except Exception as e:rec['parse_error']=str(e)
   files.append(rec)
(OUT/'file_inventory.json').write_text(json.dumps({'snapshot':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exclusions':sorted(SKIP),'files':files},ensure_ascii=False,indent=2))
# Full text preserves read scope for later searching; not a new authoritative report.
(OUT/'source_text_index.json').write_text(json.dumps(texts,ensure_ascii=False))
with (OUT/'document_digest.txt').open('w') as o:
 for p,t in texts.items():
  if not p.endswith('.md'):continue
  if '/checkpoint-' in p or '/results/' in p or '/aggregate/' in p:continue
  lines=t.splitlines(); selected=[]
  for i,l in enumerate(lines):
   if i<10 or l.startswith('#') or any(w in l for w in ['结论','不能','尚未','未完成','未执行','失败','不支持','不等于','不足','已完成','待核','暂停','停止']):selected.append(f'{i+1}: {l}')
  o.write('\nFILE '+p+'\n'+'\n'.join(selected)+'\n')
print('indexed',len(files),'files; full text',len(texts),'digest bytes',(OUT/'document_digest.txt').stat().st_size)
