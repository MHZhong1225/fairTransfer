"""Resume the exact COCO selection using its public S3 bucket with valid TLS."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import urllib.request

from PIL import Image
from common import STUDY,now,read,rows,sha,write,write_rows


def main():
    prior = STUDY/'data/scene_supplement_v1'
    output = STUDY/'data/scene_supplement_v2'
    output.mkdir(exist_ok=False);(output/'images').mkdir()
    plan = rows(prior/'download_plan.jsonl')
    for row in plan:
        filename = row['original_metadata']['file_name']
        row['transport_url'] = 'https://s3.amazonaws.com/images.cocodataset.org/train2017/'+filename
        row['image_path'] = str(output/'images'/filename)
    write_rows(output/'download_plan.jsonl',plan)
    write(output/'contract.json',dict(created_utc=now(),code_sha256=sha(__file__),
        prior_selection_sha256=sha(prior/'download_plan.jsonl'),prior_contract_sha256=sha(prior/'contract.json'),
        plan_sha256=sha(output/'download_plan.jsonl'),planned=len(plan),
        change='Same image IDs and splits. COCO HTTPS custom hostname failed certificate validation. Official public bucket path-style HTTPS works with normal certificate validation. Original metadata URL is retained.',
        scientific_selection_changed=False,tls_verification_disabled=False,
        source_url='https://cocodataset.org/#download'))
    status = dict(status='running',started_utc=now(),planned=len(plan),attempted=0,downloaded=0)
    write(output/'status.json',status)
    def download(row):
        try:
            request = urllib.request.Request(row['transport_url'],headers={'User-Agent':'research-image-audit/1.0'})
            with urllib.request.urlopen(request,timeout=45) as response:
                content = response.read()
            path = Path(row['image_path'])
            with path.open('xb') as stream:
                stream.write(content)
            with Image.open(path) as image:
                size = image.size;image.verify()
            expected = (row['original_metadata']['width'],row['original_metadata']['height'])
            if size!=expected:
                raise ValueError('Image dimensions differ from COCO source metadata')
            return dict(row,status='downloaded',image_sha256=sha(path),bytes=len(content),size=size)
        except Exception as exc:
            return dict(row,status='download_failed',error=repr(exc))
    completed=[]
    with (output/'download_results.jsonl').open('x',buffering=1) as stream,ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(download,row) for row in plan]
        for future in as_completed(futures):
            result=future.result();stream.write(json.dumps(result,ensure_ascii=False)+'\n');status['attempted']+=1
            if result['status']=='downloaded':
                completed.append(result);status['downloaded']+=1
            write(output/'status.json',status)
    completed.sort(key=lambda row:row['id'])
    write_rows(output/'manifest.jsonl',completed)
    write_rows(output/'blind_manifest.jsonl',[{key:row[key] for key in ('id','image_path','image_sha256')} for row in completed])
    status.update(status='completed',completed_utc=now(),downloaded_counts=dict(Counter(row['split'] for row in completed)))
    write(output/'status.json',status)


if __name__=='__main__':
    main()
