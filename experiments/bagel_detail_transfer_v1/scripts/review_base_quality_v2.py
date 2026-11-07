"""Review every attempted quality judgment, retaining failed parses as unknown."""
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from common import STUDY,now,read,rows,sha,write
from metrics_v2 import g_case_metrics,consensus_g
from bagel_backend import GENERATION


def main():
    directory=STUDY/'results/base_quality_v1'
    contract=read(directory/'contract.json')
    if contract['generation']!=GENERATION or read(directory/'status.json')['status']!='completed':
        raise RuntimeError('The fixed official base generation is not complete or has different settings')
    plan=rows(STUDY/'configs/generation_plan_v1/quality_check.jsonl');observations=[]
    for item in plan:
        actual=read(directory/'observations'/(item['id']+'.json'))
        if sha(actual['image_path'])!=actual['image_sha256']:
            raise ValueError('Base quality image changed')
        with Image.open(actual['image_path']) as image:
            if image.size!=(1024,1024):
                raise ValueError('Base image is not at official resolution')
            pixels=np.asarray(image.convert('RGB'),dtype=float)
            variation=float(pixels.std())
        judges={};evidence={}
        for judge in ('qwen','ministral'):
            grades={task:read(directory/'quality_grades'/judge/task/'observations'/(item['id']+'.json'))
                for task in ('depiction','appearance','content')}
            judges[judge]=g_case_metrics(item,grades['depiction'],grades['appearance'],grades['content'])
            evidence[judge]={task:dict(status=grade['status'],parsed=grade.get('parsed'),raw=grade.get('raw'))
                for task,grade in grades.items()}
        common=consensus_g(judges['qwen'],judges['ministral'],item)
        observations.append(dict(id=item['id'],image_path=actual['image_path'],image_sha256=actual['image_sha256'],
            pixel_standard_deviation=variation,judges=judges,common=common,evidence=evidence))
    attempts={}
    expected={item['id'] for item in plan}
    for judge in ('qwen','ministral'):
        attempts[judge]={}
        for task in ('depiction','appearance','content'):
            folder=directory/'quality_grades'/judge/task
            records=[read(path) for path in (folder/'observations').glob('*.json')]
            if {row['id'] for row in records}!=expected or len(records)!=len(expected):
                raise ValueError('Missing, extra or duplicate quality-grading attempts')
            if read(folder/'status.json')['status']!='completed':
                raise ValueError('A quality-grading process has not finished')
            attempts[judge][task]=dict(planned=len(expected),attempted=len(records),
                parsed=sum(row['status']=='completed' for row in records),
                failed=sum(row['status']!='completed' for row in records))
    visual=read(directory/'codex_visual_review_v1.json')
    if {row['id'] for row in visual['observations']}!=expected:
        raise ValueError('All ten original images need an actual visual review')
    for row in visual['observations']:
        if sha(row['image_path'])!=row['image_sha256']:
            raise ValueError('A visually reviewed image changed')
    review=dict(created_utc=now(),status='reviewed_for_formal_evaluation',observations=observations,
        reviewer='Two independent external vision-model interfaces plus image-file/configuration checks. No human or Codex manual image review is claimed by this file.',
        generation=GENERATION,planned=len(plan),confirmed_people=sum(r['common']['valid_person'] for r in observations),
        confirmed_settings=sum(r['common']['content'].get('setting',False) for r in observations),
        all_grades_available=all(r['common']['scored'] for r in observations),
        all_planned_attempts_present=True,attempts=attempts,
        jointly_scored=sum(r['common']['scored'] for r in observations),
        qualification_revision='configs/base_quality_review_revision_v2.json',
        visual_review_sha256=sha(directory/'codex_visual_review_v1.json'),
        sample_policy='All ten prespecified images retained; no replacement, prompt selection or guidance change.',
        limitation='Small automatic preflight checks detect gross pipeline failure; they do not establish overall image quality, subtle demographic accuracy or human gold labels.',
        source_sha256=sha(__file__))
    # Keep the existing remote/mini >=90% structure criterion. An attempted
    # failed parse remains unknown in every metric; it is not an absent job.
    # Also require >=90% jointly scored cases and all ten actual visual reviews.
    structurally_qualified=(all(part['parsed']>=.9*part['planned']
        for judge in attempts.values() for part in judge.values())
        and review['jointly_scored']>=.9*len(plan))
    if not structurally_qualified or not review['confirmed_people'] or any(
            r['pixel_standard_deviation']==0 for r in observations):
        review['status']='failed';write(directory/'quality_review_v2.json',review)
        raise RuntimeError('Gross baseline generation or quality-grading failure; evidence retained')
    review['quality_limitations']=[f'{len(plan)-review["confirmed_people"]}/{len(plan)} images lack two-judge confirmation of one recognizable photo-like person.',
        f'{len(plan)-review["confirmed_settings"]}/{len(plan)} images lack two-judge confirmation of the requested setting.']
    write(directory/'quality_review_v2.json',review)


if __name__=='__main__':
    main()
