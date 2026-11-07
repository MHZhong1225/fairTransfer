"""Finish all five technical states with the same revised scorer on two free GPUs."""
import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback
from common import STUDY,PYTHON,now,read,rows,sha,write,write_rows
from aggregate_outputs import aggregate_state


def states():
    root=STUDY/'results/mini_pipeline_v2'
    return [dict(id='mini_'+arm,arm=arm,seed=None if arm=='base' else 7100,
        understanding=str(root/arm/'understanding'),
        generation=str(STUDY/'results/base_quality_v1' if arm=='base' else root/arm/'generation'),
        grades=str(root/arm/'grades_revision_v5'),aggregate=str(root/arm/'aggregate_revision_v5'))
        for arm in ('base','T0','T1','T2','T3')]


def check_grades(output):
    status=read(output/'status.json');contract=read(output/'contract.json')
    if status['status']!='completed':
        raise RuntimeError('A previous grading process is incomplete; reconcile its evidence first')
    if any(sha(STUDY/'scripts'/name)!=digest for name,digest in contract['source_hashes'].items()):
        raise RuntimeError('Existing technical grades used different scoring code')
    values=[read(p) for p in (output/'observations').glob('*.json')]
    if len(values)!=status['planned'] or sum(v['status']=='completed' for v in values)<.9*len(values):
        raise RuntimeError(f'Technical grading completeness below 90%: {output}')


def worker(gpu):
    path=STUDY/f'results/mini_scoring_worker_v4_gpu{gpu}.json'
    status=dict(status='running',pid=os.getpid(),gpu=gpu,started_utc=now());write(path,status)
    try:
        if gpu==0:
            predecessor=read(STUDY/'results/scoring_revision_v5_launch_attempt2.json')['pid']
            while True:
                process=Path(f'/proc/{predecessor}/cmdline')
                if not process.exists() or b'grades_revision_v5' not in process.read_bytes():
                    break
                time.sleep(10)
            for judge in ('qwen','ministral'):
                check_grades(STUDY/'results/mini_pipeline_v2/base/grades_revision_v5'/judge/'understanding')
        mini=STUDY/'data/mini_pipeline_v2'
        assigned=('base','T0','T2') if gpu==0 else ('T1','T3')
        for state in [s for s in states() if s['arm'] in assigned]:
            for judge in ('qwen','ministral'):
                python=PYTHON if judge=='qwen' else STUDY/'environment_judge/bin/python'
                for task in ('understanding','depiction','appearance','content'):
                    status.update(state=state['id'],judge=judge,task=task,updated_utc=now());write(path,status)
                    output=Path(state['grades'])/judge/task
                    if not (output/'status.json').exists():
                        manifest=mini/('u_cases.jsonl' if task=='understanding' else 'g_cases.jsonl')
                        measured=state['understanding' if task=='understanding' else 'generation']
                        log=STUDY/'logs'/f'mini_scoring_v4_{state["arm"]}_{judge}_{task}.log'
                        with log.open('x') as stream:
                            result=subprocess.run([str(python),str(STUDY/'scripts/score_outputs_v5.py'),
                                '--judge',judge,'--task',task,'--manifest',str(manifest),'--measured',measured,'--output',str(output)],
                                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_DISABLE_PROGRESS_BARS='1'),
                                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                        if result.returncode:
                            raise RuntimeError(f'Technical grading failed; retain {log}')
                    check_grades(output)
            aggregate_state(state,rows(mini/'u_cases.jsonl'),rows(mini/'g_cases.jsonl'),rows(mini/'u_pairs.jsonl'))
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(path,status)


def main():
    path=STUDY/'results/mini_scoring_queue_v4.json'
    if path.exists():
        raise RuntimeError('Existing queue needs reconciliation, not replacement')
    status=dict(status='running',pid=os.getpid(),started_utc=now(),source_sha256=sha(__file__))
    write(path,status)
    try:
        if read(STUDY/'results/data_quality_queue_v2_qwen.json')['status']!='completed':
            raise RuntimeError('Original native technical pipeline must finish first')
        write_rows(STUDY/'results/mini_scoring_v4_states.jsonl',states())
        children=[]
        for gpu in (0,1):
            log=STUDY/'logs'/f'mini_scoring_queue_v4_gpu{gpu}.log'
            with log.open('x') as stream:
                child=subprocess.Popen([str(PYTHON),__file__,'--gpu',str(gpu)],stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(child)
        status.update(worker_pids=[c.pid for c in children]);write(path,status)
        codes=[c.wait() for c in children]
        if any(codes):
            raise RuntimeError('A technical scoring worker failed; keep every raw judgment')
        status.update(status='completed',completed_utc=now(),states=5,
            scope='All fixed technical cases and controls; no formal hypothesis conclusion.')
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(path,status)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,choices=(0,1))
    args=parser.parse_args()
    main() if args.gpu is None else worker(args.gpu)
