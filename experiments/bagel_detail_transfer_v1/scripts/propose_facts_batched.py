"""Batch independent image-only proposals with the same Qwen teacher and prompt."""
import argparse
import hashlib
import json
import traceback

from common import STUDY,now,read,rows,sha,write
from annotate_facts_v3 import PROPOSAL_PROMPT,parse_proposal,manifest_path


def run(stage,limit=None):
    from PIL import Image
    import torch
    from scripts.umm_u_training_v1.review import _load_judge
    torch.set_num_threads(4)
    manifest = manifest_path(stage)
    plan = rows(manifest)
    if limit is not None:
        plan = plan[:limit]
    directory = STUDY/'results'/f'fact_annotation_{stage}_v3'/'propose'
    directory.mkdir(parents=True,exist_ok=False)
    write(directory/'contract.json',dict(created_utc=now(),source_sha256=sha(__file__),
        annotation_schema_sha256=sha(__import__('pathlib').Path(__file__).with_name('annotate_facts_v3.py')),
        manifest_sha256=sha(manifest),stage=stage,planned=len(plan),limit=limit,
        model='Qwen3-VL-8B-Instruct',batch_size=4,precision='BF16',decoding='greedy',
        max_new_tokens=2048,tokenizer_padding='left',proposal_prompt=PROPOSAL_PROMPT,
        technical_change='Retain valid facts when a different fact violates the schema; record every rejected fact. Duplicate controlled demographic fields are all rejected. Excluded technical data only informed this format rule; no target-model outputs were used.',
        inference='Each image has an independent conversation. Batch padding is computational only; no image or answer from another sample enters its conversation.'))
    status = dict(status='loading',planned=len(plan),attempted=0,started_utc=now())
    write(directory/'status.json',status)
    try:
        model,processor = _load_judge('qwen','cuda:0')
        processor.tokenizer.padding_side = 'left'
        status['status'] = 'running'
        with (directory/'observations.jsonl').open('x',buffering=1) as stream:
            for offset in range(0,len(plan),4):
                batch = plan[offset:offset+4];messages=[]
                for item in batch:
                    if sha(item['image_path'])!=item['image_sha256']:
                        raise ValueError('Source image changed')
                    with Image.open(item['image_path']) as image:
                        source = image.convert('RGB').copy()
                    messages.append([{'role':'user','content':[{'type':'image','image':source},
                        {'type':'text','text':PROPOSAL_PROMPT}]}])
                inputs = processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                    return_dict=True,return_tensors='pt',padding=True).to(model.device)
                started = now()
                with torch.inference_mode():
                    output = model.generate(**inputs,do_sample=False,max_new_tokens=2048)
                outputs = processor.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True,
                    clean_up_tokenization_spaces=False)
                for item,raw in zip(batch,outputs):
                    result = dict(id=item['id'],started_utc=started,completed_utc=now(),raw=raw,
                        generated_batch_columns=int(output.shape[1]-inputs.input_ids.shape[1]),batch_offset=offset,
                        prompt_sha256=hashlib.sha256(PROPOSAL_PROMPT.encode()).hexdigest())
                    try:
                        result.update(parsed=parse_proposal(raw),status='completed')
                    except (ValueError,TypeError,KeyError):
                        result.update(status='parse_failed',error=traceback.format_exc())
                    stream.write(json.dumps(result,ensure_ascii=False)+'\n')
                    status['attempted']+=1
                write(directory/'status.json',status)
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc(),updated_utc=now());raise
    finally:
        write(directory/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('technical','candidate','scene'))
    parser.add_argument('--limit',type=int)
    args=parser.parse_args();run(args.stage,args.limit)
