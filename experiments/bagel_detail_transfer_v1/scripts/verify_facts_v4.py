"""Independent native-resolution Ministral reviews of each proposed image fact."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
from common import STUDY,now,rows,read,sha,write
from annotate_facts_v3 import (manifest_path,completed_lines,verification_prompt,parse_verification)
from ministral_judge import load,MODEL
from ministral_batch import respond_batch


def run(stage,limit=None):
    import torch
    from PIL import Image
    torch.set_num_threads(4)
    manifest=manifest_path(stage);plan=rows(manifest)
    if limit is not None:
        plan=plan[:limit]
    directory=STUDY/'results'/f'fact_annotation_{stage}_v4'/'verify'
    directory.mkdir(parents=True,exist_ok=False)
    dependencies=('verify_facts_v4.py','ministral_judge.py','ministral_batch.py','annotate_facts_v3.py')
    write(directory/'contract.json',dict(created_utc=now(),manifest_sha256=sha(manifest),stage=stage,
        planned=len(plan),limit=limit,model=str(MODEL),precision='BF16',batch_size=4,
        preprocessing='Official MistralCommonBackend tekken image processing; native pixels, no manual resize. Batch padding is masked using true image_sizes.',
        decoding='greedy',max_new_tokens=2048,
        source_hashes={name:sha(Path(__file__).with_name(name)) for name in dependencies},
        blindness='Image pixels and proposed claims only; no source request, occupation metadata, split label or BAGEL result enters the model.',
        limitation='Verification is conditioned on proposed claims and may share errors with the proposer. Agreement is not human gold.'))
    status=dict(status='loading',planned=len(plan),attempted=0,started_utc=now());write(directory/'status.json',status)
    try:
        model,tokenizer=load();status['status']='running'
        parent=directory.parent/'propose'
        with (directory/'observations.jsonl').open('x',buffering=1) as stream:
            for offset in range(0,len(plan),4):
                batch=plan[offset:offset+4]
                while True:
                    proposals={r['id']:r for r in completed_lines(parent/'observations.jsonl')}
                    if all(item['id'] in proposals for item in batch):
                        break
                    if (parent/'status.json').exists() and read(parent/'status.json')['status'] in ('failed','completed'):
                        raise RuntimeError('Proposal producer ended before the full planned batch')
                    time.sleep(5)
                eligible=[item for item in batch if proposals[item['id']]['status']=='completed']
                results={}
                if eligible:
                    images=[];prompts=[]
                    for item in eligible:
                        if sha(item['image_path'])!=item['image_sha256']:
                            raise ValueError('Source image changed')
                        with Image.open(item['image_path']) as image:
                            images.append(image.convert('RGB').copy())
                        prompts.append(verification_prompt(proposals[item['id']]['parsed']))
                    outputs=respond_batch(model,tokenizer,images,prompts)
                    for item,raw,prompt in zip(eligible,outputs,prompts):
                        result=dict(id=item['id'],raw=raw,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
                        try:
                            parsed=parse_verification(raw,len(proposals[item['id']]['parsed']['facts']))
                            result.update(status='completed',parsed=parsed)
                        except (ValueError,TypeError,KeyError):
                            result.update(status='parse_failed',error=traceback.format_exc())
                        results[item['id']]=result
                for item in batch:
                    result=results.get(item['id'],dict(id=item['id'],status='missing_proposal',proposal_status=proposals[item['id']]['status']))
                    result['completed_utc']=now();stream.write(json.dumps(result,ensure_ascii=False)+'\n')
                    status['attempted']+=1
                write(directory/'status.json',status)
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(directory/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('technical','candidate','scene'))
    parser.add_argument('--limit',type=int)
    args=parser.parse_args();run(args.stage,args.limit)
