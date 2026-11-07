"""Use the producer GPU after annotation to complete real pipeline qualification."""
import os
from pathlib import Path
import subprocess
import time
import traceback
from common import STUDY,PYTHON,now,read,sha,write


def execute(script,args,label):
    process=subprocess.Popen([str(PYTHON),str(STUDY/'scripts'/script),*map(str,args)],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',HF_HUB_DISABLE_PROGRESS_BARS='1'),
        stdin=subprocess.DEVNULL,stdout=(STUDY/'logs'/label).open('x'),stderr=subprocess.STDOUT,start_new_session=True)
    if process.wait():
        raise RuntimeError(f'{script} failed; inspect {label}, original output retained')


def main():
    path=STUDY/'results/post_annotation_qualification_queue_v1.json'
    status=dict(status='waiting_producer',started_utc=now(),source_sha256=sha(__file__));write(path,status)
    try:
        producer=STUDY/'results/fact_annotation_full_v5/propose/status.json'
        while True:
            state=read(producer)['status']
            if state=='failed':
                raise RuntimeError('Source proposal worker failed')
            if state=='completed':
                break
            time.sleep(10)
        status['status']='task_branch_qualification';write(path,status)
        execute('qualify_task_branches.py',[],'task_branch_qualification_v1.log')
        status['status']='native_base_quality';write(path,status)
        quality=STUDY/'results/base_quality_v1'
        execute('generate_images.py',['--manifest',STUDY/'configs/generation_plan_v1/quality_check.jsonl',
            '--output',quality,'--save-base-noise',quality/'noise_reference'],'base_quality_v1.log')
        # This only produces the quality panel; the supervisor must inspect it before formal G.
        mini=STUDY/'data/mini_pipeline_v1'
        for arm in ('T0','T1','T2','T3'):
            status.update(status='mini_pipeline',arm=arm);write(path,status)
            output=STUDY/'results/mini_pipeline_v1'/arm
            execute('train_arm.py',['--arm',arm,'--seed','7100','--train',mini/'train.jsonl',
                '--dev',mini/'dev.jsonl','--output',output/'training','--technical'],f'mini_{arm}_train.log')
            adapter=read(output/'training/status.json')['final_adapter']
            execute('run_understanding.py',['--cases',mini/'u_cases.jsonl','--adapter',adapter,
                '--output',output/'understanding'],f'mini_{arm}_understanding.log')
            execute('generate_images.py',['--manifest',mini/'g_cases.jsonl','--adapter',adapter,
                '--output',output/'generation','--replay-noise',quality/'noise_reference'],f'mini_{arm}_generation.log')
        status.update(status='completed',completed_utc=now(),remaining='Grade the mini outputs and inspect base images before freezing formal evaluation and training.')
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(path,status)


if __name__=='__main__':
    main()
