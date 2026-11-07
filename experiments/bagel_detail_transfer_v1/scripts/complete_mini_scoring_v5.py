"""Join the resumed GPU0 and original GPU1 work without repeating model runs."""
from pathlib import Path
import os
import time
import traceback
from common import STUDY,now,read,sha,write
from mini_scoring_queue_v4 import states,check_grades


def main():
    path=STUDY/'results/mini_scoring_queue_v5.json'
    if path.exists():
        raise RuntimeError('Existing completion audit must not be overwritten')
    status=dict(status='running',pid=os.getpid(),stage='waiting_existing_workers',started_utc=now(),source_sha256=sha(__file__))
    write(path,status)
    try:
        while True:
            workers=[read(STUDY/f'results/mini_scoring_worker_v4_gpu{gpu}.json') for gpu in (0,1)]
            if any(w['status']=='failed' for w in workers):
                raise RuntimeError('A technical scoring worker failed; evidence retained')
            if all(w['status']=='completed' for w in workers):
                break
            time.sleep(15)
        for state in states():
            for judge in ('qwen','ministral'):
                for task in ('understanding','depiction','appearance','content'):
                    check_grades(Path(state['grades'])/judge/task)
            if not (Path(state['aggregate'])/'summary.json').exists():
                raise RuntimeError('A measured technical state has not been aggregated')
        status.update(status='completed',completed_utc=now(),states=5,
            recovery='Same worker implementation and all original cases. The pair-file path alias and preserved failure are in results/mini_scoring_v4_path_recovery.',
            scope='Real technical pipeline only; not formal fairness evidence.')
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(path,status)


if __name__=='__main__':
    main()
