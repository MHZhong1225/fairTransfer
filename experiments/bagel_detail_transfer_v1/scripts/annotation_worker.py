"""Blind, manifest-driven image fact production and independent verification."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
from common import STUDY,now,rows,read,sha,write
from annotate_facts_v3 import (PROPOSAL_PROMPT,parse_proposal,verification_prompt,
    parse_verification,completed_lines)


def qwen_batch(model,processor,images,prompts):
    import torch
    messages=[[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
        for image,prompt in zip(images,prompts,strict=True)]
    inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
        return_dict=True,return_tensors='pt',padding=True).to(model.device)
    with torch.inference_mode():
        output=model.generate(**inputs,do_sample=False,max_new_tokens=2048)
    return processor.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,
        clean_up_tokenization_spaces=False)


def run(manifest,output,task):
    import torch
    from PIL import Image
    torch.set_num_threads(4)
    plan=rows(manifest);directory=output/task;directory.mkdir(parents=True,exist_ok=False)
    dependencies=('annotation_worker.py','annotate_facts_v3.py','ministral_judge.py','ministral_batch.py')
    write(directory/'contract.json',dict(created_utc=now(),manifest=str(manifest),manifest_sha256=sha(manifest),
        planned=len(plan),task=task,batch_size=4,precision='BF16',decoding='greedy',max_new_tokens=2048,
        model='Qwen3-VL-8B-Instruct' if task=='propose' else 'Ministral-3-8B-Instruct-2512-BF16',
        source_hashes={name:sha(Path(__file__).with_name(name)) for name in dependencies},
        proposal_prompt=PROPOSAL_PROMPT,preprocessing='Official model processor without manual resizing or lower pixel limits.',
        blindness='Only image pixels and fixed task enter proposer; verifier also sees proposed claims, never source requests or BAGEL outputs.',
        limitation='Model agreement is not human gold; every rejection, uncertainty and parse failure is retained.'))
    status=dict(status='loading',planned=len(plan),attempted=0,started_utc=now());write(directory/'status.json',status)
    try:
        if task=='propose':
            from scripts.umm_u_training_v1.review import _load_judge
            model,processor=_load_judge('qwen','cuda:0');processor.tokenizer.padding_side='left'
            respond=qwen_batch
        else:
            from ministral_judge import load
            from ministral_batch import respond_batch
            model,processor=load();respond=respond_batch
        status['status']='running'
        with (directory/'observations.jsonl').open('x',buffering=1) as stream:
            for offset in range(0,len(plan),4):
                batch=plan[offset:offset+4]
                proposals={}
                if task=='verify':
                    parent=output/'propose'
                    while True:
                        proposals={r['id']:r for r in completed_lines(parent/'observations.jsonl')}
                        if all(item['id'] in proposals for item in batch):
                            break
                        if (parent/'status.json').exists() and read(parent/'status.json')['status'] in ('failed','completed'):
                            raise RuntimeError('Proposal producer ended before the full planned batch')
                        time.sleep(5)
                eligible=[item for item in batch if task=='propose' or proposals[item['id']]['status']=='completed']
                results={}
                if eligible:
                    images=[];prompts=[]
                    for item in eligible:
                        if sha(item['image_path'])!=item['image_sha256']:
                            raise ValueError('Source image changed')
                        with Image.open(item['image_path']) as image:
                            images.append(image.convert('RGB').copy())
                        prompts.append(PROPOSAL_PROMPT if task=='propose' else verification_prompt(proposals[item['id']]['parsed']))
                    started=now();outputs=respond(model,processor,images,prompts)
                    for item,raw,prompt in zip(eligible,outputs,prompts,strict=True):
                        result=dict(id=item['id'],started_utc=started,raw=raw,
                            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
                        try:
                            parsed=parse_proposal(raw) if task=='propose' else parse_verification(raw,len(proposals[item['id']]['parsed']['facts']))
                            result.update(status='completed',parsed=parsed)
                        except (ValueError,TypeError,KeyError):
                            result.update(status='parse_failed',error=traceback.format_exc())
                        results[item['id']]=result
                for item in batch:
                    result=results.get(item['id'],dict(id=item['id'],status='missing_proposal'))
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
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',choices=('propose','verify'),required=True)
    args=parser.parse_args();run(args.manifest,args.output,args.task)
