"""Identical image-only depiction controls and full source screening on server 2."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback
import torch
from PIL import Image
from common import now,read,rows,sha,write
from remote_judge_runtime import load
from judge_runtime import respond
from depiction import PROMPT,parse


def main(args):
    torch.set_num_threads(4)
    plan=rows(args.manifest);controls=read(args.controls);directory=args.output
    directory.mkdir(parents=True,exist_ok=False);(directory/'observations').mkdir()
    status=dict(status='verifying_assets',pid=os.getpid(),started_utc=now(),planned=len(plan),attempted=0)
    write(directory/'status.json',status)
    try:
        if args.judge=='ministral':
            ready=Path(os.environ['UMM_JUDGE_MODEL_ROOT']).parent/'results/ministral_environment_ready.json'
            while not ready.exists():
                status.update(status='waiting_pinned_environment',updated_utc=now());write(directory/'status.json',status)
                time.sleep(10)
            if read(ready)['status']!='ready':
                raise RuntimeError('Pinned independent-judge environment failed')
        provenance=read(args.provenance)
        model=Path(os.environ['UMM_JUDGE_MODEL_ROOT'])/Path(provenance['path']).name
        actual_hashes={name:sha(model/name) for name in provenance['files']}
        if any(actual_hashes[name]!=details['sha256'] for name,details in provenance['files'].items()):
            raise ValueError('Remote model differs from the verified local source')
        versions={name:importlib.metadata.version(name) for name in
            ('torch','transformers','numpy','pillow','accelerate','safetensors','tokenizers','huggingface-hub')}
        if args.judge=='ministral':
            versions.update({name:importlib.metadata.version(name) for name in
                ('mistral-common','opencv-python-headless','tiktoken','sentencepiece','pydantic','pydantic-extra-types')})
        write(directory/'contract.json',dict(created_utc=now(),judge=args.judge,planned=len(plan),prompt=PROMPT,
            manifest_sha256=sha(args.manifest),controls=controls,batch_size=4,max_new_tokens=256,
            precision='BF16',attn_implementation='sdpa',decoding='greedy',preprocessing='Unmodified official native preprocessing; original image bytes.',
            model_files_sha256=actual_hashes,versions=versions,python=platform.python_version(),cuda=torch.version.cuda,
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
            source_hashes={name:sha(Path(__file__).with_name(name)) for name in
                ('remote_depiction_worker.py','remote_judge_runtime.py','judge_runtime.py','ministral_judge.py','ministral_batch.py','depiction.py','annotate_facts_v3.py')},
            allocation='Dedicated third GPU authorized by the researcher. Only model/data file locations differ from the original source-screen interface.',
            acceptance='All five fixed photo-like people, the obvious sculpture and the gray blank must receive their expected categories.',
            limitation='Automatic obvious controls, not human-gold demographic or depiction labels.'))
        model,processor=load(args.judge);predictions=[]
        for offset in range(0,len(controls),4):
            batch=controls[offset:offset+4]
            images=[Image.open(r['image_path']).convert('RGB') if r['image_path'] else Image.new('RGB',(448,448),(128,128,128)) for r in batch]
            outputs=respond(model,processor,args.judge,images,[PROMPT]*len(batch),limit=256)
            for row,raw in zip(batch,outputs,strict=True):
                predictions.append(dict(id=row['id'],expected=row['expected'],raw=raw,parsed=parse(raw)))
        write(directory/'controls.json',predictions)
        if any(r['expected']!=r['parsed']['category'] for r in predictions):
            raise ValueError('Prespecified remote depiction controls failed')
        status.update(status='screening',controls_passed=True,screening_started_utc=now());write(directory/'status.json',status)
        start=time.perf_counter()
        for offset in range(0,len(plan),4):
            batch=plan[offset:offset+4];images=[]
            for row in batch:
                if sha(row['image_path'])!=row['image_sha256']:
                    raise ValueError('Remote source image bytes changed')
                with Image.open(row['image_path']) as image:
                    images.append(image.convert('RGB').copy())
            outputs=respond(model,processor,args.judge,images,[PROMPT]*len(batch),limit=256)
            for row,raw in zip(batch,outputs,strict=True):
                result=dict(id=row['id'],raw=raw,image_sha256=row['image_sha256'],completed_utc=now())
                try:
                    result.update(status='completed',parsed=parse(raw))
                except (ValueError,TypeError,KeyError):
                    result.update(status='parse_failed',error=traceback.format_exc())
                write(directory/'observations'/(row['id']+'.json'),result);status['attempted']+=1
            elapsed=time.perf_counter()-start
            status.update(updated_utc=now(),images_per_minute=status['attempted']/elapsed*60,
                estimated_remaining_seconds=(len(plan)-status['attempted'])*elapsed/status['attempted'],
                peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
            write(directory/'status.json',status)
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(directory/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judge',choices=('qwen','ministral'),required=True)
    for name in ('manifest','controls','provenance','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
