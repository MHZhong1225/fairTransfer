"""Resume formal remote grading with the recovered third-card workers."""
from pathlib import Path
import os
import time
import traceback
from common import STUDY,now,read,rows,sha,write
from remote_grading_v4 import grade_outputs


def main():
    path=STUDY/'results/remote_formal_grading_queue_v7.json'
    status=dict(status='running',stage='waiting_native_outputs',pid=os.getpid(),
                started_utc=now(),source_sha256=sha(__file__),finished=[])
    write(path,status)
    try:
        config=STUDY/'configs/formal_v1/freeze.json'
        freeze=read(config)
        states=rows(freeze['states_manifest'])
        for state in states:
            if state['id'] in status['finished']:
                continue
            ready=Path(state['aggregate']).parent/'native_ready.json'
            while not ready.exists():
                status.update(stage='waiting_native_outputs',state=state['id'],updated_utc=now())
                write(path,status)
                time.sleep(30)
            marker=read(ready)
            if marker['freeze_sha256']!=sha(config):
                raise ValueError('Native state belongs to a different frozen experiment')
            status.update(stage='remote_grading',state=state['id'],updated_utc=now())
            write(path,status)
            destination=Path(state['aggregate']).parent/'grades_remote_v7'
            if destination.exists():
                status['finished'].append(state['id'])
                continue
            grade_outputs('formal_'+state['id']+'_v7',Path(freeze['g_cases']),
                Path(state['generation']),destination,
                Path(freeze['u_cases']),Path(state['understanding']))
            status['finished'].append(state['id'])
            write(path,dict(status,updated_utc=now()))
        status.update(status='completed',stage='all_states_graded',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc(),failed_utc=now())
        raise
    finally:
        write(path,status)


if __name__=='__main__':
    main()
