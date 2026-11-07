"""Persist every independent model judgment, including invalid or missing cases."""
import argparse
import hashlib
from pathlib import Path
import time
import traceback
from PIL import Image
import torch
from common import now,read,rows,sha,write
from judge_runtime import respond
from remote_judge_runtime import load
from depiction import PROMPT as DEPICTION_PROMPT,parse as parse_depiction
from grading import (understanding_prompt,parse_understanding,APPEARANCE_PROMPT,
    parse_appearance,content_prompt,parse_content)


def observations(root):
    indexed={}
    for path in sorted(root.rglob('observations/*.json')):
        row=read(path)
        if row['id'] in indexed:
            raise ValueError('Duplicate measured case; choose one model state and all its distinct shards')
        indexed[row['id']]=dict(row,observation_path=str(path))
    return indexed


def run(args):
    torch.set_num_threads(4)
    plan=rows(args.manifest);measured=observations(args.measured)
    directory=args.output
    dependencies=('score_outputs_remote_v1.py','remote_judge_runtime.py','depiction.py','grading.py','judge_runtime.py','ministral_judge.py','ministral_batch.py')
    contract=dict(judge=args.judge,task=args.task,manifest=str(args.manifest),manifest_sha256=sha(args.manifest),
        measured=str(args.measured),planned=len(plan),batch_size=4,precision='BF16',decoding='greedy',max_new_tokens=4096,
        measured_files={key:sha(row['observation_path']) for key,row in measured.items()},
        source_hashes={name:sha(Path(__file__).with_name(name)) for name in dependencies},
        blindness='Judges never receive training arm, checkpoint identity or source request metadata. G appearance receives the same image-only prompt for all states and no requested demographic attributes.',
        limitation='Each external judge is reported separately; agreement is not human gold.')
    if args.resume:
        if read(directory/'contract.json')!=contract:
            raise ValueError('Resume cannot change frozen grading or measured outputs')
    else:
        directory.mkdir(parents=True,exist_ok=False);(directory/'observations').mkdir();write(directory/'contract.json',contract)
    status=dict(status='loading',planned=len(plan),attempted=0,started_utc=now());write(directory/'status.json',status)
    try:
        model,processor=load(args.judge);status['status']='grading'
        for offset in range(0,len(plan),4):
            batch=plan[offset:offset+4];pending=[];images=[];prompts=[]
            for item in batch:
                path=directory/'observations'/(item['id']+'.json')
                if path.exists():
                    status['attempted']+=1;continue
                if item['id'] not in measured:
                    write(path,dict(id=item['id'],status='missing_model_output',completed_utc=now()))
                    status['attempted']+=1;continue
                actual=measured[item['id']]
                image_path=item.get('image_path') if args.task=='understanding' else actual['image_path']
                expected_sha=item.get('image_sha256') if args.task=='understanding' else actual['image_sha256']
                if image_path:
                    if sha(image_path)!=expected_sha:
                        raise ValueError('Judged image changed')
                    with Image.open(image_path) as image:
                        pixels=image.convert('RGB').copy()
                else:
                    pixels=None
                if args.task=='understanding':
                    prompt=understanding_prompt(item,actual['response'])
                    if pixels is None:
                        prompt='No image is available in this control. Do not infer unseen facts.\n'+prompt
                elif args.task=='depiction':
                    prompt=DEPICTION_PROMPT
                elif args.task=='appearance':
                    prompt=APPEARANCE_PROMPT
                else:
                    prompt=content_prompt(item)
                pending.append((item,actual,path));images.append(pixels);prompts.append(prompt)
            if pending:
                started=now();output=respond(model,processor,args.judge,images,prompts)
                for (item,actual,path),raw,prompt in zip(pending,output,prompts,strict=True):
                    result=dict(id=item['id'],raw=raw,started_utc=started,
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),measured_sha256=sha(actual['observation_path']))
                    try:
                        if args.task=='understanding':
                            parsed=parse_understanding(raw,item,actual['response'])
                        elif args.task=='depiction':
                            parsed=parse_depiction(raw)
                        elif args.task=='appearance':
                            parsed=parse_appearance(raw)
                        else:
                            parsed=parse_content(raw,item)
                        result.update(status='completed',parsed=parsed)
                    except (ValueError,TypeError,KeyError):
                        result.update(status='parse_failed',error=traceback.format_exc())
                    result['completed_utc']=now();write(path,result);status['attempted']+=1
            status['peak_gpu_memory_bytes']=torch.cuda.max_memory_allocated()
            write(directory/'status.json',status)
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(directory/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judge',choices=('qwen','ministral'),required=True)
    parser.add_argument('--task',choices=('understanding','depiction','appearance','content'),required=True)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--measured',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    run(parser.parse_args())
