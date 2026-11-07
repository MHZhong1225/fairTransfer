"""Complete the excluded technical end-to-end runs on the released judge GPU."""
from pathlib import Path
import time
import traceback
from common import STUDY,PYTHON,now,read,rows,sha,write,write_rows
from data_quality_queue_v2 import execute


def wait_status(path,accepted,parents=()):
    while True:
        if path.exists():
            value=read(path)
            if value['status'] in accepted:
                return value
            if value['status']=='failed':
                raise RuntimeError(f'Dependency failed: {path}')
        for parent in parents:
            if parent.exists() and read(parent)['status']=='failed':
                raise RuntimeError(f'Parent queue failed: {parent}')
        time.sleep(10)


def main():
    status_path=STUDY/'results/mini_scoring_queue_v2.json'
    status=dict(status='waiting_qualifications',started_utc=now(),source_sha256=sha(__file__))
    write(status_path,status)
    try:
        queues=[STUDY/'results/data_quality_queue_v2_qwen.json',STUDY/'results/data_quality_queue_v3_ministral.json']
        wait_status(queues[1],{'completed'},queues)
        for judge in ('qwen','ministral'):
            wait_status(STUDY/'results/scoring_qualification_v2'/judge/'status.json',{'passed'},queues)
        mini=STUDY/'data/mini_pipeline_v2'
        while not (mini/'contract.json').exists():
            if any(read(p)['status']=='failed' for p in queues):
                raise RuntimeError('Mini-data preparation dependency failed')
            time.sleep(10)
        root=STUDY/'results/mini_pipeline_v2';states=[]
        for arm in ('base','T0','T1','T2','T3'):
            state=dict(id='mini_'+arm,arm=arm,seed=None if arm=='base' else 7100,
                understanding=str(root/arm/'understanding'),
                generation=str(STUDY/'results/base_quality_v1' if arm=='base' else root/arm/'generation'),
                grades=str(root/arm/'grades'),aggregate=str(root/arm/'aggregate'))
            if arm=='base':
                status.update(status='base_understanding',arm=arm);write(status_path,status)
                execute('run_understanding.py',['--cases',mini/'u_cases.jsonl','--output',state['understanding']],
                    PYTHON,0,'mini_base_understanding.log')
            else:
                status.update(status='waiting_mini_output',arm=arm);write(status_path,status)
                wait_status(Path(state['generation'])/'status.json',{'completed'},queues)
            status.update(status='grading',arm=arm);write(status_path,status)
            for judge in ('qwen','ministral'):
                python=PYTHON if judge=='qwen' else STUDY/'environment_judge/bin/python'
                for task in ('understanding','depiction','appearance','content'):
                    output=Path(state['grades'])/judge/task
                    execute('score_outputs_v3.py',['--judge',judge,'--task',task,
                        '--manifest',mini/('u_cases.jsonl' if task=='understanding' else 'g_cases.jsonl'),
                        '--measured',state['understanding' if task=='understanding' else 'generation'],
                        '--output',output],python,0,f'mini_{arm}_{judge}_{task}.log')
                    records=[read(p) for p in (output/'observations').glob('*.json')]
                    if not records or sum(r['status']=='completed' for r in records)/len(records)<.9:
                        raise RuntimeError(f'Technical grading completeness below 90%: {arm}/{judge}/{task}')
            from aggregate_outputs import aggregate_state
            aggregate_state(state,rows(mini/'u_cases.jsonl'),rows(mini/'g_cases.jsonl'),rows(mini/'u_image_pairs.jsonl'))
            states.append(state)
        write_rows(mini/'states.jsonl',states)
        status.update(status='completed',completed_utc=now(),states=len(states),
            scope='Technical pipeline qualification only; no formal research conclusion.')
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(status_path,status)


if __name__=='__main__':
    main()
