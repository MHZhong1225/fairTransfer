"""Grade completed native states on server 2 while both BAGEL GPUs keep working."""
from pathlib import Path
import os
import time
import traceback
from common import STUDY,now,read,rows,sha,write
from aggregate_outputs import aggregate_state
from remote_grading import grade_outputs


def main():
    path=STUDY/'results/remote_formal_grading_queue_v1.json'
    status=dict(status='running',stage='waiting_formal_freeze',pid=os.getpid(),started_utc=now(),source_sha256=sha(__file__))
    write(path,status)
    try:
        config=STUDY/'configs/formal_v1/freeze.json'
        while not config.exists():
            parent=read(STUDY/'results/full_study_queue_v1.json')
            if parent['status']=='failed':
                raise RuntimeError('Pre-formal parent failed; inspect preserved evidence')
            time.sleep(30)
        freeze=read(config)
        if freeze.get('grading_location')!='remote_gpu':
            status.update(status='completed',stage='local_grading_configured',enabled=False,completed_utc=now())
            return
        states=rows(freeze['states_manifest']);finished=[]
        while len(finished)<len(states):
            available=[s for s in states if s['id'] not in finished and
                (Path(s['aggregate']).parent/'native_ready.json').exists()]
            if not available:
                workers=[STUDY/f'results/formal_worker_v1_gpu{gpu}.json' for gpu in (0,1)]
                if any(p.exists() and read(p)['status']=='failed' for p in workers):
                    raise RuntimeError('A native-output worker failed before remaining planned states were ready')
                status.update(stage='waiting_native_outputs',finished=finished,updated_utc=now());write(path,status)
                time.sleep(30)
                continue
            state=available[0]
            marker=read(Path(state['aggregate']).parent/'native_ready.json')
            if marker['freeze_sha256']!=sha(config):
                raise ValueError('Native state belongs to a different frozen experiment')
            status.update(stage='remote_grading',state=state['id'],finished=finished,updated_utc=now());write(path,status)
            grade_outputs('formal_'+state['id'],Path(freeze['g_cases']),Path(state['generation']),Path(state['grades']),
                Path(freeze['u_cases']),Path(state['understanding']))
            aggregate_state(state,rows(freeze['u_cases']),rows(freeze['g_cases']),rows(freeze['u_pairs']))
            finished.append(state['id'])
        status.update(status='completed',stage='all_states_graded',finished=finished,completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc(),failed_utc=now());raise
    finally:
        write(path,status)


if __name__=='__main__':
    main()
