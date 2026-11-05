"""Check new report evidence links and source changes, without experiment edits."""
from pathlib import Path
import json,re,hashlib,urllib.parse,datetime
B=Path.cwd();O=B/'reports/research_audit';E=O/'evidence'
def target_path(p,target):
 target=urllib.parse.unquote(target.strip('<>')).split('#')[0]
 target=re.sub(r':\d+(?::\d+)?$','',target)
 return p.parent/target
broken=[];checked=0
for p in O.rglob('*.md'):
 if p.parent==E:continue # raw historical excerpts, source-relative references are literal evidence
 for i,l in enumerate(p.read_text().splitlines(),1):
  for target in re.findall(r'\]\(([^)]+)\)',l):
   if re.match(r'\w+://|#',target):continue
   checked+=1
   if not target_path(p,target).exists():broken.append(dict(source=str(p),line=i,target=target))
legacy=[]
for p in [B/'README.md',*sorted((B/'docs').glob('*.md'))]:
 for i,l in enumerate(p.read_text().splitlines(),1):
  for target in re.findall(r'\]\(([^)]+)\)',l):
   if re.match(r'\w+://|#',target):continue
   if not target_path(p,target).exists():legacy.append(dict(source=str(p.relative_to(B)),line=i,target=target))
(E/'unresolved_document_links.json').write_text(json.dumps(legacy,ensure_ascii=False,indent=2))
F=json.load(open(E/'file_inventory.json'))['files'];changes=[]
for r in F:
 if not r.get('sha256') or not r['path'].startswith(('docs/','experiments/')):continue
 p=B/r['path']
 if not p.exists():changes.append(dict(path=r['path'],change='missing'));continue
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 if h!=r['sha256']:changes.append(dict(path=r['path'],change='hash_changed',before=r['sha256'],after=h))
report=dict(snapshot_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checked_report_links=checked,broken_report_links=broken,legacy_missing_links=legacy,source_changes_since_inventory=changes,scope='Original text hashes only; active experiments may change themselves. New index files were not in the initial inventory.')
(E/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('report links',checked,'broken',len(broken),'legacy missing',len(legacy),'source hash changes',len(changes));print(json.dumps(changes,ensure_ascii=False)[:1600])
