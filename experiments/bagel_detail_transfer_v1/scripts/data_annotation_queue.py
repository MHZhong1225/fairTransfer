"""Qualify the annotation format and judges, then build the actual fact bank."""
import os
import subprocess
import time
import traceback

from common import STUDY,PYTHON,now,read,rows,sha,write


def wait_completed(path):
    while True:
        if path.exists():
            status=read(path)['status']
            if status in ('completed','passed'):
                return
            if status.startswith('failed'):
                raise RuntimeError('Prerequisite failed: '+str(path))
        time.sleep(10)


def launch(name,script,args,device):
    environment=os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(device),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    command=[str(PYTHON),str(STUDY/'scripts'/script),*args]
    with (STUDY/'logs'/(name+'.log')).open('x') as stream:
        child=subprocess.Popen(command,env=environment,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
    return child,command


def main():
    path=STUDY/'results/data_annotation_queue_v1.json'
    status=dict(status='waiting_technical_v2',started_utc=now(),completed=[])
    write(path,status)
    try:
        wait_completed(STUDY/'results/fact_annotation_technical_v2/verify/status.json')
        for name,script,args in [
            ('fact_technical_v3_propose','propose_facts_batched.py',['technical','--limit','4']),
            ('fact_technical_v3_verify','annotate_facts_v3.py',['technical','verify','--limit','4']),
            ('fact_controls_qwen','qualify_fact_judges.py',['qwen']),
            ('fact_controls_phi','qualify_fact_judges.py',['phi'])]:
            child,command=launch(name,script,args,1)
            status.update(status='running_technical',active=name,pid=child.pid,command=command,updated_utc=now());write(path,status)
            if child.wait():
                raise RuntimeError('Annotation qualification failed: '+name)
            status['completed'].append(name);write(path,status)
        for task in ('propose','verify'):
            observations=rows(STUDY/f'results/fact_annotation_technical_v3/{task}/observations.jsonl')
            if len(observations)!=4 or any(r['status']!='completed' for r in observations):
                raise ValueError('Batched annotation interface has invalid technical outputs')
        sources=[STUDY/'scripts'/name for name in ('annotate_facts_v3.py','propose_facts_batched.py','qualify_fact_judges.py','common.py')]
        write(STUDY/'configs/annotation_freeze_v3.json',dict(created_utc=now(),sources={str(p):sha(p) for p in sources},
            technical_checks={str(p):sha(p) for p in [STUDY/'results/fact_judge_controls_v1'/judge/'status.json' for judge in ('qwen','phi')]},
            limits='Format and obvious visible-content controls only; this is not human gold or proof of accuracy for subtle age/skin/action judgments.'))
        status.update(status='waiting_backend_qualification',active=None,updated_utc=now());write(path,status)
        wait_completed(STUDY/'results/qualification_queue.json')
        wait_completed(STUDY/'data/scene_supplement_v2/status.json')
        for stage in ('candidate','scene'):
            active={}
            for device,task,script,args in [(1,'propose','propose_facts_batched.py',[stage]),
                                          (0,'verify','annotate_facts_v3.py',[stage,'verify'])]:
                name=f'fact_{stage}_v3_{task}'
                active[task]=launch(name,script,args,device)[0]
            status.update(status='running_facts',stage=stage,active={task:p.pid for task,p in active.items()},updated_utc=now());write(path,status)
            while active:
                for task,child in list(active.items()):
                    code=child.poll()
                    if code is None:
                        continue
                    if code:
                        for other in active.values():
                            if other.poll() is None:
                                other.terminate()
                        raise RuntimeError(f'Fact annotation failed: {stage}/{task}')
                    status['completed'].append(stage+'/'+task);del active[task]
                status['active']={task:p.pid for task,p in active.items()};write(path,status)
                if active:
                    time.sleep(10)
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc(),updated_utc=now());raise
    finally:
        write(path,status)


if __name__=='__main__':
    main()
