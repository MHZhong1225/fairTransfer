"""Visible scene controls and blank-image contrasts, excluded from training."""
import argparse
import json
from pathlib import Path
import traceback
from common import STUDY,now,rows,sha,write
from annotate_facts_v3 import verification_prompt,parse_verification

CLAIMS = [
    ['A large work surface is visible in front of the person.',
     'The person has visible hands near the work surface.',
     'The person is holding a tennis racket.',
     'The person is riding a bicycle.'],
    ['A knife is visible near the person’s hand.',
     'Part of the person’s bare torso is visible.',
     'The person’s full face is visible.',
     'The person is wearing a bright red coat.'],
    ['A laptop is visible in front of the person.',
     'The person is wearing glasses.',
     'The person is reading an open book held in their hands.',
     'The person is standing outdoors.'],
    ['The person is holding a tennis racket.',
     'The person’s arms are raised.',
     'The person is seated behind a laptop on a desk.',
     'The person is riding a bicycle.'],
]


def main(judge):
    import torch
    from PIL import Image
    torch.set_num_threads(4)
    directory=STUDY/'results/scene_judge_controls_v1'/judge;directory.mkdir(parents=True,exist_ok=False)
    plan=rows(STUDY/'data/scene_technical_v1/blind_manifest.jsonl')
    write(directory/'contract.json',dict(created_utc=now(),images=plan,claims=CLAIMS,code_sha256=sha(__file__),
        annotation_worker_sha256=sha(Path(__file__).with_name('annotation_worker.py')),
        acceptance='For each native image, claims 0 and 1 YES; claims 2 and 3 not YES. For blank images no claim YES.',
        scope='Four assistant-inspected technical scenes and synthetic blank controls, excluded from formal training/evaluation. Not independent human gold.',
        judge=judge,transform='Blank negative controls preserve source canvas dimensions and contain uniform gray pixels; real images remain unchanged.'))
    status=dict(status='loading',started_utc=now());write(directory/'status.json',status)
    try:
        if judge=='qwen':
            from scripts.umm_u_training_v1.review import _load_judge
            from annotation_worker import qwen_batch
            model,processor=_load_judge('qwen','cuda:0');processor.tokenizer.padding_side='left';respond=qwen_batch
        else:
            from ministral_judge import load
            from ministral_batch import respond_batch
            model,processor=load();respond=respond_batch
        images=[Image.open(item['image_path']).convert('RGB') for item in plan]
        prompts=[verification_prompt(dict(facts=[dict(field='activity',value='visual claim',statement=s,
            region='whole_image',visibility='clear') for s in claims])) for claims in CLAIMS]
        predictions=[]
        for condition in ('native','blank'):
            inputs=images if condition=='native' else [Image.new('RGB',image.size,(128,128,128)) for image in images]
            output=respond(model,processor,inputs,prompts)
            for item,raw in zip(plan,output,strict=True):
                predictions.append(dict(id=item['id'],condition=condition,raw=raw,parsed=parse_verification(raw,4)))
        with (directory/'observations.jsonl').open('x') as stream:
            for result in predictions:
                stream.write(json.dumps(result,ensure_ascii=False)+'\n')
        positive=sum(r['supported']=='YES' for p in predictions if p['condition']=='native' for r in p['parsed']['reviews'] if r['id']<2)
        false_positive=sum(r['supported']=='YES' for p in predictions for r in p['parsed']['reviews'] if p['condition']=='blank' or r['id']>=2)
        status.update(positive_accepted=positive,positive_planned=8,false_positive=false_positive,negative_planned=24)
        if positive!=8 or false_positive!=0:
            raise ValueError('Scene or blank-image controls did not meet the frozen criterion')
        status.update(status='passed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(directory/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('judge',choices=('qwen','ministral'))
    main(parser.parse_args().judge)
