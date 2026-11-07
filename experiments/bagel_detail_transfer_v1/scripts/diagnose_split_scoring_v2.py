"""Actual fixed controls for separated text reading and visual verification."""
import argparse
import os
import subprocess
import traceback
from common import STUDY,now,read,rows,sha,write


def run(judge):
    from PIL import Image
    import torch
    from remote_judge_runtime import load
    from understanding_grading_v3 import score_batch
    torch.set_num_threads(4)
    directory=STUDY/'results/scoring_split_diagnostic_v2'/judge
    directory.mkdir(parents=True,exist_ok=False)
    source=read(STUDY/'results/scoring_quote_diagnostic_v1'/judge/'contract.json')
    item=rows(STUDY/'data/scene_technical_v1/manifest.jsonl')[2]
    assert sha(item['image_path'])==source['image_sha256']
    image=Image.open(item['image_path']).convert('RGB')
    write(directory/'contract.json',dict(source,source_sha256=sha(__file__),
        split_scoring_sha256=sha(STUDY/'scripts/understanding_grading_v3.py'),
        scope='Prespecified correct, contradictory, empty and mixed technical answers; reference matching and claim extraction see text only; only extracted claims are verified against the image. No formal outcomes exist.'))
    status=dict(status='running',pid=os.getpid(),started_utc=now());write(directory/'status.json',status)
    try:
        model,processor=load(judge)
        values=score_batch(model,processor,judge,[image]*4,[source['case']]*4,source['responses'],
            lambda i,stage,raw:write(directory/f'raw/{i}/{stage}.json',raw))
        for i,value in enumerate(values):
            write(directory/f'case_{i}.json',value)
        status.update(status='completed',parse_errors=[i for i,v in enumerate(values) if 'error' in v],completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(directory/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('judge',choices=('qwen','ministral','both'))
    judge=parser.parse_args().judge
    if judge=='both':
        for name in ('qwen','ministral'):
            subprocess.run([str(STUDY/'envs'/name/'bin/python'),__file__,name],check=True)
    else:
        run(judge)
