"""Snapshot all assets and parse every JSONL outside bundled dependencies."""
import os,json,csv,hashlib,collections,datetime
from pathlib import Path
OUT=Path('reports/research_audit/evidence')
F=json.load(open(OUT/'file_inventory.json'))['files']
with (OUT/'run_records.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['path','rows','parse_errors','unique_ids','duplicate_ids','statuses','first_fields','sha256','changed_during_read']);w.writeheader()
 for r in F:
  p=Path(r['path'])
  if p.suffix!='.jsonl':continue
  before=p.stat();n=bad=0;ids=[];status=collections.Counter();h=hashlib.sha256();keys=[]
  with p.open('rb') as stream:
   for raw in stream:
    h.update(raw)
    if not raw.strip():continue
    n+=1
    try:
     d=json.loads(raw)
     if isinstance(d,dict):
      if not keys:keys=list(d)[:25]
      if 'status' in d:status[str(d['status'])]+=1
      if isinstance(d.get('id'),(str,int)):ids.append(d['id'])
    except Exception:bad+=1
  after=p.stat();w.writerow(dict(path=str(p),rows=n,parse_errors=bad,unique_ids=len(set(ids)) if ids else '',duplicate_ids=len(ids)-len(set(ids)) if ids else '',statuses=json.dumps(status),first_fields=json.dumps(keys),sha256=h.hexdigest(),changed_during_read=before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns))
# Includes dependency trees as assets, without interpreting their test data as experiments.
count=0;total=0
with (OUT/'filesystem_inventory.jsonl').open('w') as f:
 for base in ['docs','experiments','scripts']:
  for root,ds,fs in os.walk(base,followlinks=False):
   for n in fs:
    p=Path(root)/n;s=p.lstat();r=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,symlink=p.is_symlink())
    if p.is_symlink():r['target']=os.readlink(p)
    f.write(json.dumps(r,ensure_ascii=False)+'\n');count+=1;total+=s.st_size
(OUT/'scan_scope.json').write_text(json.dumps(dict(snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),asset_files=count,logical_bytes=total,warning='Logical sizes include hardlink duplicates; not unique physical storage. No full-image/weight hash audit.'),indent=2))
print('asset files',count,'logical bytes',total)
