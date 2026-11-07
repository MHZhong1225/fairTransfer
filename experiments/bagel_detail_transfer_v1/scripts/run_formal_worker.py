"""Run one GPU's frozen formal workload; preserve failures and stop on error."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback
from common import STUDY,PYTHON,now,read,rows,sha,write


def completed(path):
    return path.exists() and read(path).get('status')=='completed'


def execute(script,arguments,output,gpu,label,python=PYTHON):
    status_path=Path(output)/'status.json'
    if completed(status_path):
        return
    arguments=list(map(str,arguments))
    if Path(output).exists():
        if not (Path(output)/'contract.json').exists():
            raise RuntimeError(f'Partial output needs reconciliation: {output}')
        arguments.append('--resume')
    logdir=STUDY/'logs/formal_v1';logdir.mkdir(exist_ok=True)
    attempt=len(list(logdir.glob(label+'_attempt_*.log')))
    with (logdir/f'{label}_attempt_{attempt:02d}.log').open('x') as log:
        process=subprocess.Popen([str(python),str(STUDY/'scripts'/script),*arguments],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_DISABLE_PROGRESS_BARS='1'),
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    launch=logdir/f'{label}_attempt_{attempt:02d}.json'
    evidence=dict(pid=process.pid,started_utc=now(),script=script,arguments=arguments,
        output=str(output),gpu=gpu,python=str(python),source_sha256=sha(STUDY/'scripts'/script))
    write(launch,evidence)
    code=process.wait();evidence.update(exit_code=code,completed_utc=now());write(launch,evidence)
    if code or not completed(status_path):
        raise RuntimeError(f'Formal task failed: {label}. Original outputs retained; inspect before resuming.')


def validate_freeze(freeze):
    for path,digest in freeze['file_hashes'].items():
        if sha(path)!=digest:
            raise ValueError(f'Frozen input or implementation changed: {path}')
    for path,expected in freeze['required_statuses'].items():
        if read(path)['status']!=expected:
            raise ValueError(f'Qualification no longer meets frozen requirement: {path}')


def main(args):
    freeze=read(args.freeze);validate_freeze(freeze)
    states=rows(freeze['states_manifest']);gpu=args.gpu
    state_path=STUDY/'results'/f'formal_worker_v1_gpu{gpu}.json'
    if state_path.exists() and read(state_path).get('status') in ('running','waiting_base'):
        prior=read(state_path)
        try:
            os.kill(prior['pid'],0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('A worker with this GPU allocation is still alive')
    status=dict(status='running',pid=os.getpid(),gpu=gpu,started_utc=now(),freeze_sha256=sha(args.freeze))
    write(state_path,status)
    try:
        base=next(s for s in states if s['arm']=='base')
        u_cases=Path(freeze['u_cases']);g_cases=Path(freeze['g_cases']);noise=Path(freeze['base_noise'])
        # Both GPUs first create distinct base shards. Each paired observation is
        # generated once, then its exact saved latent is reused by all trained arms.
        for task,script,manifest in (('understanding','run_understanding.py',u_cases),('generation','generate_images.py',g_cases)):
            status.update(stage='base_'+task,updated_utc=now());write(state_path,status)
            output=Path(base[task])/f'shard_{gpu}'
            arguments=['--cases' if task=='understanding' else '--manifest',manifest,
                '--output',output,'--shard',gpu,'--shards',2]
            if task=='generation':
                arguments.extend(['--save-base-noise',noise])
            execute(script,arguments,output,gpu,f'base_{task}_{gpu}')
        status.update(status='waiting_base',updated_utc=now());write(state_path,status)
        while not all(completed(Path(base[task])/f'shard_{g}'/'status.json')
                for task in ('understanding','generation') for g in (0,1)):
            other=STUDY/'results'/f'formal_worker_v1_gpu{1-gpu}.json'
            if other.exists() and read(other)['status']=='failed':
                raise RuntimeError('Other base worker failed')
            time.sleep(15)
        assigned=[s for s in states if s['worker']==gpu]
        for state in assigned:
            validate_freeze(freeze)
            status.update(status='running',state=state['id'],stage='training',updated_utc=now());write(state_path,status)
            if state['arm']!='base':
                output=Path(state['training'])
                execute('train_arm_v2.py',['--arm',state['arm'],'--seed',state['seed'],
                    '--train',freeze['train'],'--dev',freeze['dev'],'--output',output],output,gpu,state['id']+'_training')
                adapter=read(output/'status.json')['final_adapter']
                for task,script,manifest in (('understanding','run_understanding.py',u_cases),('generation','generate_images.py',g_cases)):
                    status.update(stage=task,updated_utc=now());write(state_path,status)
                    output=Path(state[task]);arguments=['--cases' if task=='understanding' else '--manifest',manifest,
                        '--adapter',adapter,'--output',output]
                    if task=='generation':
                        arguments.extend(['--replay-noise',noise])
                    execute(script,arguments,output,gpu,state['id']+'_'+task)
            if freeze.get('grading_location')=='remote_gpu':
                write(Path(state['aggregate']).parent/'native_ready.json',dict(state=state['id'],
                    completed_utc=now(),freeze_sha256=sha(args.freeze),
                    grading='Assigned to the separately qualified third GPU; native BAGEL work continues.'))
                continue
            for judge in ('qwen','ministral'):
                python=PYTHON if judge=='qwen' else STUDY/'environment_judge/bin/python'
                for task in ('understanding','depiction','appearance','content'):
                    status.update(stage=f'grading_{judge}_{task}',updated_utc=now());write(state_path,status)
                    output=Path(state['grades'])/judge/task
                    execute('score_outputs_v2.py',['--judge',judge,'--task',task,
                        '--manifest',u_cases if task=='understanding' else g_cases,
                        '--measured',state['understanding' if task=='understanding' else 'generation'],
                        '--output',output],output,gpu,state['id']+'_'+judge+'_'+task,python)
            status.update(stage='aggregate',updated_utc=now());write(state_path,status)
            from aggregate_outputs import aggregate_state
            if not (Path(state['aggregate'])/'summary.json').exists():
                aggregate_state(state,rows(u_cases),rows(g_cases),rows(freeze['u_pairs']))
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(state_path,status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True);parser.add_argument('--gpu',type=int,choices=(0,1),required=True)
    main(parser.parse_args())
