"""Prepare a fixed COCO person/object supplement; do not infer actions from labels."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import random
import urllib.request

from PIL import Image
from common import STUDY,now,sha,write,write_rows

CATEGORIES = ('bicycle','motorcycle','umbrella','skis','sports ball','baseball bat',
              'skateboard','surfboard','tennis racket','cup','knife','laptop','cell phone','book','scissors')
SOURCE = Path('/home/ubuntu/zmh/dataset/coco_annotations/instances_train2017.json')


def main():
    output = STUDY/'data/scene_supplement_v1'
    output.mkdir(exist_ok=False)
    (output/'images').mkdir()
    source = json.loads(SOURCE.read_text())
    images = {r['id']:r for r in source['images']}
    names = {r['id']:r['name'] for r in source['categories']}
    licenses = {r['id']:r for r in source['licenses']}
    annotations = defaultdict(list)
    for row in source['annotations']:
        annotations[row['image_id']].append(row)
    buckets = defaultdict(list)
    for image_id,boxes in annotations.items():
        image = images[image_id];area = image['width']*image['height']
        people = [r for r in boxes if r['category_id']==1]
        if len(people)!=1 or people[0]['iscrowd'] or people[0]['bbox'][2]*people[0]['bbox'][3]<.08*area:
            continue
        objects = {names[r['category_id']] for r in boxes if not r['iscrowd'] and
                   names[r['category_id']] in CATEGORIES and r['bbox'][2]*r['bbox'][3]>=.001*area}
        for category in objects:
            buckets[category].append(image_id)
    generator = random.Random(2026110302)
    used,selected = set(),[]
    # Rare categories first. No target-model outputs, demographic requests or image judgments enter selection.
    for category in sorted(CATEGORIES,key=lambda name:(len(buckets[name]),name)):
        candidates = sorted(set(buckets[category])-used);generator.shuffle(candidates)
        for index,image_id in enumerate(candidates[:70]):
            image = images[image_id];used.add(image_id)
            split = 'train' if index<50 else 'dev' if index<60 else 'test'
            selected.append(dict(id=f'coco_{image_id:012d}',source='COCO train2017',source_image_id=image_id,
                source_family=f'coco_{image_id}',split=split,source_object_stratum=category,
                source_request_group=None,source_url=image['coco_url'].replace('http:','https:',1),
                original_metadata=image,license=licenses[image['license']],
                image_path=str(output/'images'/image['file_name'])))
    write_rows(output/'download_plan.jsonl',selected)
    write(output/'contract.json',dict(created_utc=now(),code_sha256=sha(__file__),source_annotation_path=str(SOURCE),
        source_annotation_sha256=sha(SOURCE),download_plan_sha256=sha(output/'download_plan.jsonl'),
        candidate_counts=dict(Counter(r['split'] for r in selected)),categories=CATEGORIES,seed=2026110302,
        eligibility='One non-crowd annotated person, person box >=8% of area, and an object box >=0.1%. These source annotations choose candidates, not facts or adult/demographic truth. All visual facts undergo the same blind annotation and verification as BPO.',
        purpose='The user plan permits external image supplementation when actions/objects are absent. This fixed pool supplies candidate visible interactions and contexts without manufacturing new images. No G output influences selection.',
        source_balance='All four training groups will use identical source membership and loss weights. Report source-by-appearance/context coverage; do not attribute a source shift to supervision content.',
        final_use='Candidate acquisition only. Technical BPO coverage audit and scene annotation must be reported before inclusion in a frozen training pool.',
        rights='Per-image license and original URLs preserved from published COCO metadata. Local research analysis; no image redistribution is performed.',
        failure_policy='One attempt per HTTPS source. Preserve every failure and missing planned denominator; no outcome-based replacement.'))
    status = dict(status='running',planned=len(selected),attempted=0,downloaded=0,started_utc=now())
    write(output/'status.json',status)
    def download(row):
        try:
            request = urllib.request.Request(row['source_url'],headers={'User-Agent':'research-image-audit/1.0'})
            with urllib.request.urlopen(request,timeout=45) as response:
                content = response.read()
            destination = Path(row['image_path'])
            with destination.open('xb') as stream:
                stream.write(content)
            with Image.open(destination) as image:
                image.verify()
            return dict(row,status='downloaded',image_sha256=hashlib.sha256(content).hexdigest(),bytes=len(content))
        except Exception as exc:
            return dict(row,status='download_failed',error=repr(exc))
    successful = []
    with (output/'download_results.jsonl').open('x',buffering=1) as stream,ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(download,row) for row in selected]
        for future in as_completed(futures):
            result = future.result();stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            status['attempted'] += 1
            if result['status']=='downloaded':
                successful.append(result);status['downloaded']+=1
            write(output/'status.json',status)
    successful.sort(key=lambda r:r['id'])
    write_rows(output/'manifest.jsonl',successful)
    write_rows(output/'blind_manifest.jsonl',[{k:r[k] for k in ('id','image_path','image_sha256')} for r in successful])
    status.update(status='completed',completed_utc=now(),downloaded_counts=dict(Counter(r['split'] for r in successful)))
    write(output/'status.json',status)


if __name__=='__main__':
    main()
