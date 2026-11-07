"""Inventory BPO without exposing source requests to visual annotators."""
from collections import Counter
import hashlib
from pathlib import Path
import random
import shutil
import subprocess

from common import ROOT, STUDY, PRIOR, BAGEL_REPO, BAGEL_WEIGHTS, now, read, sha, write, write_rows


def main():
    data = STUDY / 'data'
    data.mkdir(exist_ok=False)
    candidates = []
    captions = {}
    manifests = [PRIOR / f'author_{dimension}_data_v1/understanding_{dimension}.json' for dimension in ('gender', 'race')]
    for manifest in manifests:
        dimension = 'gender' if 'gender' in manifest.name else 'race'
        source_hashes = read(manifest.parent / 'image_sha256.json')
        for row in read(manifest):
            path = Path(row['image'])
            prompt = int(path.parent.name.split('_')[1])
            index = int(path.stem.split('_')[1])
            group = path.parent.parent.name
            relative = '/'.join(path.parts[-4:])
            caption = row['conversations'][1]['value']
            if dimension == 'gender':
                captions[prompt] = caption
            identity = f'bpo/{group}/{prompt}/{index}'
            candidates.append(dict(id=hashlib.sha256(identity.encode()).hexdigest()[:20],
                source='pkulium/bpo', source_revision='a5b259422cf6ef0602ac810602361e0acec5b122',
                source_request_group=group, source_prompt_index=prompt, source_image_index=index,
                source_family=f'bpo_occupation_{prompt}', image_path=str(path),
                image_sha256=source_hashes[relative], original_caption=caption))
    prompt_ids = sorted(captions)
    random.Random(2026110301).shuffle(prompt_ids)
    formal_ids, technical_ids = prompt_ids[:50], prompt_ids[50:66]
    splits = {p: ('train' if i < 40 else 'dev' if i < 45 else 'test') for i, p in enumerate(formal_ids)}
    formal = [dict(row, split=splits[row['source_prompt_index']]) for row in candidates if row['source_prompt_index'] in splits]
    pilot = [dict(row, split='technical_only') for row in candidates
             if row['source_prompt_index'] in technical_ids and row['source_image_index'] == 0]
    for subset in (formal, pilot):
        subset.sort(key=lambda row: row['id'])
        for row in subset:
            if sha(row['image_path']) != row['image_sha256']:
                raise ValueError('Source image differs from preserved source manifest')
    if len(formal) != 3000 or len(pilot) != 96:
        raise ValueError('Unexpected source membership')
    write_rows(data / 'candidate_manifest.jsonl', formal)
    write_rows(data / 'technical_manifest.jsonl', pilot)
    for label, subset in (('candidate', formal), ('technical', pilot)):
        write_rows(data / f'{label}_blind_manifest.jsonl', [
            {key: row[key] for key in ('id', 'image_path', 'image_sha256')} for row in subset])
    provenance = dict(created_utc=now(), seed=2026110301, source_images=len(candidates),
        candidate_images=len(formal), candidate_splits=dict(Counter(row['split'] for row in formal)),
        candidate_requested_groups=dict(Counter(row['source_request_group'] for row in formal)),
        technical_images=len(pilot), technical_prompt_ids=technical_ids,
        occupation_names={str(p): captions[p] for p in formal_ids + technical_ids},
        sources={str(p):sha(p) for p in manifests},
        plans={str(p):sha(p) for p in data.glob('*manifest.jsonl')},
        selection='Seeded random source occupations before visual annotations or BAGEL outputs; all six source groups and all ten existing images per group retained. No preference for clear faces or apparently useful actions.',
        splitting='Whole source occupation families initially held apart: 40 train, 5 dev, 5 test. Exact and perceptual duplicate connected components will be checked before final data freeze. Source image index is not asserted to be an independent original generation seed.',
        source_groups='Generation-request metadata, never visual ground truth. Race request names remain provenance only, never true ethnicity labels.',
        stage='Candidate pool and excluded technical inspection only; no formal training authorized by this data inventory alone.')
    write(data / 'source_inventory.json', provenance)
    input_path = Path('/home/ubuntu/.codex/attachments/c17dbc15-8c28-4cc1-aaf9-99fe8012ba69/Pasted text.txt')
    shutil.copyfile(input_path, STUDY / 'configs/user_research_plan.txt')
    shutil.copyfile(ROOT / '实验开展须知.md', STUDY / 'configs/experiment_requirements.md')
    write(STUDY / 'configs/intake.json', dict(created_utc=now(),
        user_plan_sha256=sha(input_path), requirements_sha256=sha(ROOT/'实验开展须知.md'),
        model='ByteDance-Seed/BAGEL-7B-MoT', model_revision=BAGEL_WEIGHTS.name,
        upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BAGEL_REPO,text=True).strip(),
        isolation='Workspace is not a Git repository; all new code/configurations/outputs live in this dedicated directory. Historical files and upstream sources are read-only.',
        priority='The attached plan and explicit BAGEL request supersede historical VILA-U-only designs. T0/T1/T2/T3 are the first-round core; correction and set-choice supervision are deferred conditional branches, not automatic jobs.'))


if __name__ == '__main__':
    main()
