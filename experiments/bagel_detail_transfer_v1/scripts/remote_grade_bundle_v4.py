"""Sequential full-precision judge processes on the third GPU, with durable status."""
import argparse
import os
from pathlib import Path
import subprocess
import traceback
from common import now,read,sha,write


def main(job):
    request=read(job/'request.json');root=Path(request['remote_root'])
    status=dict(status='running',pid=os.getpid(),started_utc=now(),request_sha256=sha(job/'request.json'),
        source_sha256=sha(__file__))
    write(job/'status.json',status)
    try:
        for relative,digest in request['source_hashes'].items():
            if sha(root/'scripts'/relative)!=digest:
                raise ValueError('Remote scoring implementation differs from the staged contract')
        if request['mode']=='qualify':
            verified={}
            for judge in ('qwen','ministral'):
                provenance=read(root/'data'/f'{judge}_model_provenance.json')
                model=root/'models'/Path(provenance['path']).name
                verified[judge]={name:sha(model/name) for name in provenance['files']}
                if any(verified[judge][name]!=value['sha256'] for name,value in provenance['files'].items()):
                    raise ValueError('Remote scoring model weights/configuration do not match the source')
            write(job/'model_verification.json',dict(status='passed',files=verified,completed_utc=now()))
        for judge in ('qwen','ministral'):
            python=root/'envs'/judge/'bin/python'
            tasks=['qualification'] if request['mode']=='qualify' else request['tasks']
            for task in tasks:
                status.update(judge=judge,task=task,updated_utc=now());write(job/'status.json',status)
                if task=='qualification':
                    output=root/'results/scoring_qualification_remote_v6'/judge
                    command=[str(python),str(root/'scripts/qualify_scoring_remote_v6.py'),judge]
                    expected='passed'
                else:
                    output=job/'grades'/judge/task
                    measured='understanding' if task=='understanding' else 'generation'
                    manifest='u_cases.jsonl' if task=='understanding' else 'g_cases.jsonl'
                    command=[str(python),str(root/'scripts/score_outputs_remote_v4.py'),
                        '--judge',judge,'--task',task,'--manifest',str(job/'inputs'/manifest),
                        '--measured',str(job/'inputs'/measured),'--output',str(output)]
                    expected='completed'
                if (output/'status.json').exists() and read(output/'status.json')['status']==expected:
                    contract=read(output/'contract.json')
                    if any(sha(root/'scripts'/name)!=digest for name,digest in contract['source_hashes'].items()):
                        raise ValueError('Cannot reuse completed scoring with changed implementation')
                    continue
                if output.exists():
                    raise RuntimeError('An incomplete previous scoring attempt needs explicit reconciliation')
                with (job/f'{judge}_{task}.log').open('x') as log:
                    result=subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',
                        UMM_JUDGE_MODEL_ROOT=str(root/'models'),HF_HUB_DISABLE_PROGRESS_BARS='1'),
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode or read(output/'status.json')['status']!=expected:
                    raise RuntimeError(f'Remote scoring failed: {judge}/{task}; evidence retained')
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(job/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    main(parser.parse_args().job)
