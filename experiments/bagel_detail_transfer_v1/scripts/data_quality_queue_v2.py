"""Finish source checks and then qualifying work on the released GPU only."""
import argparse
import os
import subprocess
import time
import traceback
from common import STUDY,PYTHON,now,read,sha,write


def execute(script,args,python,gpu,label):
    process=subprocess.Popen([str(python),str(STUDY/'scripts'/script),*map(str,args)],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_DISABLE_PROGRESS_BARS='1'),
        stdin=subprocess.DEVNULL,stdout=(STUDY/'logs'/label).open('x'),stderr=subprocess.STDOUT,start_new_session=True)
    if process.wait():
        raise RuntimeError(f'{script} failed; inspect {label}, outputs retained')


def main(judge):
    gpu=1 if judge=='qwen' else 0;role='propose' if judge=='qwen' else 'verify'
    python=PYTHON if judge=='qwen' else STUDY/'environment_judge/bin/python'
    path=STUDY/'results'/f'data_quality_queue_v2_{judge}.json'
    status=dict(status='waiting_annotation',started_utc=now(),source_sha256=sha(__file__));write(path,status)
    try:
        while True:
            state=read(STUDY/f'results/fact_annotation_full_v5/{role}/status.json')['status']
            parent=read(STUDY/'results/annotation_queue_v5.json')['status']
            if state=='failed' or parent=='failed':
                raise RuntimeError('Source annotation failed')
            if state=='completed':
                break
            time.sleep(10)
        status['status']='depiction_screen';write(path,status)
        execute('check_depictions.py',[judge],python,gpu,f'depiction_check_v1_{judge}.log')
        if judge=='ministral':
            for scorer in ('qwen','ministral'):
                status.update(status='scoring_qualification',scorer=scorer);write(path,status)
                executable=PYTHON if scorer=='qwen' else STUDY/'environment_judge/bin/python'
                execute('qualify_scoring.py',[scorer],executable,gpu,f'scoring_qualification_{scorer}.log')
        else:
            status['status']='task_branch_qualification';write(path,status)
            execute('qualify_task_branches.py',[],PYTHON,gpu,'task_branch_qualification_v1.log')
            status['status']='native_base_quality';write(path,status)
            quality=STUDY/'results/base_quality_v1'
            execute('generate_images.py',['--manifest',STUDY/'configs/generation_plan_v1/quality_check.jsonl',
                '--output',quality,'--save-base-noise',quality/'noise_reference'],PYTHON,gpu,'base_quality_v1.log')
            while True:
                other_path=STUDY/'results/depiction_check_v1/ministral/status.json'
                if not other_path.exists():
                    time.sleep(10)
                    continue
                other=read(other_path)['status']
                if other=='failed':
                    raise RuntimeError('Independent depiction screen failed')
                if other=='completed' and read(STUDY/'results/annotation_queue_v5.json')['status']=='completed':
                    break
                time.sleep(10)
            execute('build_fact_bank_v2.py',['--technical'],PYTHON,gpu,'technical_fact_bank_v2.log')
            execute('build_fact_bank_v2.py',[],PYTHON,gpu,'formal_fact_bank_v2.log')
            execute('prepare_mini_pipeline_v2.py',[],PYTHON,gpu,'prepare_mini_pipeline_v2.log')
            mini=STUDY/'data/mini_pipeline_v2'
            for arm in ('T0','T1','T2','T3'):
                status.update(status='mini_pipeline',arm=arm);write(path,status)
                output=STUDY/'results/mini_pipeline_v2'/arm
                execute('train_arm_v2.py',['--arm',arm,'--seed','7100','--train',mini/'train.jsonl','--dev',mini/'dev.jsonl',
                    '--output',output/'training','--technical'],PYTHON,gpu,f'mini_{arm}_train.log')
                adapter=read(output/'training/status.json')['final_adapter']
                execute('run_understanding.py',['--cases',mini/'u_cases.jsonl','--adapter',adapter,'--output',output/'understanding'],
                    PYTHON,gpu,f'mini_{arm}_understanding.log')
                execute('generate_images.py',['--manifest',mini/'g_cases.jsonl','--adapter',adapter,'--output',output/'generation',
                    '--replay-noise',quality/'noise_reference'],PYTHON,gpu,f'mini_{arm}_generation.log')
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(path,status)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('judge',choices=('qwen','ministral'))
    main(parser.parse_args().judge)
