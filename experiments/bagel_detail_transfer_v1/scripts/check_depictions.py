"""Qualify and run independent image-only depiction screening before data freeze."""
import argparse
import hashlib
from pathlib import Path
import traceback
import torch
from PIL import Image
from common import STUDY,now,rows,sha,write,write_rows
from judge_runtime import load,respond
from depiction import PROMPT,parse


def main(judge):
    from remote_source_handoff import use_remote_if_running
    if use_remote_if_running(judge):
        return
    torch.set_num_threads(4)
    directory=STUDY/'results/depiction_check_v1'/judge;directory.mkdir(parents=True,exist_ok=False)
    (directory/'observations').mkdir()
    sources=[STUDY/'data'/name for name in ('combined_sources_v1/blind_manifest.jsonl',
        'technical_blind_manifest.jsonl','scene_technical_v1/blind_manifest.jsonl')]
    plan=[row for source in sources for row in rows(source)]
    if len({r['id'] for r in plan})!=len(plan):
        raise ValueError('Overlapping depiction source ids')
    technical=rows(sources[1]);scene=rows(sources[2])[2]
    statue=next(r for r in technical if r['id']=='4e819ce2f82fa0fdb3e3')
    controls=[dict(row,expected='PHOTO_LIKE_PERSON') for row in technical[:4]+[scene]]
    controls.append(dict(statue,expected='SCULPTURE_OR_MANNEQUIN'))
    controls.append(dict(id='blank_gray',image_path=None,image_sha256=None,expected='NO_PERSON'))
    write(directory/'contract.json',dict(created_utc=now(),judge=judge,planned=len(plan),prompt=PROMPT,
        input_hashes={str(p):sha(p) for p in sources},controls=controls,batch_size=4,max_new_tokens=256,
        source_hashes={name:sha(Path(__file__).with_name(name)) for name in
            ('check_depictions.py','remote_source_handoff.py','depiction.py','judge_runtime.py','ministral_judge.py','ministral_batch.py')},
        reason='Assistant inspection found a white sculpted bust jointly mislabeled as a person with light skin in technical annotation. No formal BAGEL outcomes have been observed.',
        acceptance='All five inspected photo-like people, the obvious sculpture and the gray blank must receive their expected categories.',
        limitation='Only obvious technical controls; an automatic screen with unknowns retained, not human-gold depiction labeling.'))
    status=dict(status='loading',planned=len(plan),attempted=0,started_utc=now());write(directory/'status.json',status)
    try:
        model,processor=load(judge);predictions=[]
        for offset in range(0,len(controls),4):
            batch=controls[offset:offset+4]
            images=[Image.open(r['image_path']).convert('RGB') if r['image_path'] else Image.new('RGB',(448,448),(128,128,128)) for r in batch]
            outputs=respond(model,processor,judge,images,[PROMPT]*len(batch),limit=256)
            for row,raw in zip(batch,outputs):
                predictions.append(dict(id=row['id'],expected=row['expected'],raw=raw,parsed=parse(raw)))
        write(directory/'controls.json',predictions)
        if any(r['expected']!=r['parsed']['category'] for r in predictions):
            raise ValueError('Depiction controls failed; do not screen the source pool with this interface')
        status.update(status='screening',controls_passed=True);write(directory/'status.json',status)
        for offset in range(0,len(plan),4):
            batch=plan[offset:offset+4];images=[]
            for row in batch:
                if sha(row['image_path'])!=row['image_sha256']:
                    raise ValueError('Depiction source image changed')
                with Image.open(row['image_path']) as image:
                    images.append(image.convert('RGB').copy())
            outputs=respond(model,processor,judge,images,[PROMPT]*len(batch),limit=256)
            for row,raw in zip(batch,outputs):
                result=dict(id=row['id'],raw=raw,image_sha256=row['image_sha256'],completed_utc=now())
                try:
                    result.update(status='completed',parsed=parse(raw))
                except (ValueError,TypeError,KeyError):
                    result.update(status='parse_failed',error=traceback.format_exc())
                write(directory/'observations'/(row['id']+'.json'),result);status['attempted']+=1
            write(directory/'status.json',status)
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(directory/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('judge',choices=('qwen','ministral'))
    main(parser.parse_args().judge)
