"""Expand independent source occupations and deduplicate BPO by exact pixels/files."""
from collections import Counter,defaultdict
import hashlib
from pathlib import Path
import random

from common import STUDY,PRIOR,now,read,rows,sha,write,write_rows


def main():
    output = STUDY/'data/bpo_pool_v2'
    output.mkdir(exist_ok=False)
    old = rows(STUDY/'data/candidate_manifest.jsonl')
    original_splits = {r['source_prompt_index']:r['split'] for r in old}
    excluded = set(read(STUDY/'data/source_inventory.json')['technical_prompt_ids'])
    gender_path = PRIOR/'author_gender_data_v1/understanding_gender.json'
    all_rows = read(gender_path)
    captions = {int(Path(r['image']).parent.name.split('_')[1]):r['conversations'][1]['value'] for r in all_rows}
    order = sorted(captions);random.Random(2026110301).shuffle(order)
    remaining = [p for p in order if p not in original_splits and p not in excluded]
    splits = dict(original_splits)
    for index,prompt in enumerate(remaining[:50]):
        splits[prompt] = 'train' if index<40 else 'dev' if index<45 else 'test'
    grouped = defaultdict(list)
    for dimension in ('gender','race'):
        source = PRIOR/f'author_{dimension}_data_v1'
        known = read(source/'image_sha256.json')
        for row in read(source/f'understanding_{dimension}.json'):
            path = Path(row['image']);prompt = int(path.parent.name.split('_')[1])
            if prompt not in splits:
                continue
            digest = known['/'.join(path.parts[-4:])]
            grouped[digest].append(dict(image_path=str(path),source_prompt_index=prompt,
                source_image_index=int(path.stem.split('_')[1]),source_request_group=path.parent.parent.name,
                source_family=f'bpo_occupation_{prompt}',split=splits[prompt],
                original_caption=row['conversations'][1]['value']))
    plan = []
    for digest,aliases in grouped.items():
        aliases.sort(key=lambda r:(r['source_image_index'],r['image_path']))
        row = aliases[0]
        if len({r['split'] for r in aliases})!=1:
            raise ValueError('Exact duplicate crosses family splits; resolve before annotation')
        if sha(row['image_path'])!=digest:
            raise ValueError('Source image hash changed')
        plan.append(dict(row,id='bpo_'+digest[:20],image_sha256=digest,source='pkulium/bpo',
            source_revision='a5b259422cf6ef0602ac810602361e0acec5b122',aliases=aliases))
    plan.sort(key=lambda r:r['id'])
    write_rows(output/'manifest.jsonl',plan)
    write_rows(output/'blind_manifest.jsonl',[{k:r[k] for k in ('id','image_path','image_sha256')} for r in plan])
    write(output/'selection.json',dict(created_utc=now(),code_sha256=sha(__file__),
        initial_plan_sha256=sha(STUDY/'data/candidate_manifest.jsonl'),
        source_family_splits=splits,excluded_technical_prompt_ids=sorted(excluded),
        source_files=sum(len(v) for v in grouped.values()),distinct_images=len(plan),
        distinct_images_by_split=dict(Counter(r['split'] for r in plan)),
        distinct_images_by_requested_group=dict(Counter(r['source_request_group'] for r in plan)),
        duplicate_group_sizes=dict(Counter(len(v) for v in grouped.values())),
        decision='Before formal training or evaluation: initial 2400 training file paths contain only 1220 distinct file hashes. Retain original split assignments; add 40/5/5 seeded unused occupation families to train/dev/test, excluding all 16 technical families. Exact duplicates count once; all aliases retained.',
        limitation='Exact byte uniqueness does not establish unique human identities or original independent generation seeds. Near-duplicate screen and source grouping still required before freeze.',
        annotation_labels='Source request group is never passed to image annotation or treated as observed demographics.'))


if __name__=='__main__':
    main()
