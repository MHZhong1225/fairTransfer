"""Use the now-free third GPU for fixed scoring controls before base images exist."""
import os
import time
import traceback
from common import STUDY,now,read,sha,write
from remote_grading import deploy_scoring_sources,execute_bundle,synchronize,HOST,REMOTE


def main():
    path=STUDY/'results/remote_scoring_controls_queue_v1.json'
    if path.exists():
        raise RuntimeError('An earlier controls queue exists; inspect it instead of duplicating work')
    status=dict(status='running',stage='waiting_source_screens',pid=os.getpid(),
        started_utc=now(),source_sha256=sha(__file__))
    write(path,status)
    try:
        while True:
            screens=[read(STUDY/f'results/remote_{judge}_preparation_queue_v1.json')
                for judge in ('qwen','ministral')]
            if any(item['status']=='failed' for item in screens):
                raise RuntimeError('A source screening queue failed; retain it and inspect first')
            if all(item['status']=='completed' for item in screens):
                break
            time.sleep(30)
        if not all(item.get('imported_source_screen') for item in screens):
            status.update(status='completed',stage='deferred_until_formal_preflight',completed_utc=now())
            return
        status.update(stage='remote_controls',updated_utc=now());write(path,status)
        deploy_scoring_sources()
        staging=STUDY/'data/remote_grading_staging_v1/qualification'
        staging.mkdir(parents=True,exist_ok=False)
        remote_job=execute_bundle('qualification',dict(mode='qualify',tasks=[]),staging)
        destination=STUDY/'results/remote_scoring_controls_v1'
        destination.mkdir(exist_ok=False)
        synchronize(HOST+':'+remote_job+'/model_verification.json',destination/'model_verification.json')
        synchronize(HOST+':'+REMOTE+'/results/scoring_qualification_remote_v1/',str(destination/'qualification')+'/')
        for judge in ('qwen','ministral'):
            if read(destination/'qualification'/judge/'status.json')['status']!='passed':
                raise ValueError('The fixed remote scoring controls did not pass')
        status.update(status='completed',stage='controls_passed',completed_utc=now(),
            request_sha256=sha(staging/'request.json'),
            scope='Only the fixed scoring controls and model hashes. Actual official-size base-image grading remains required before formal freeze.')
    except BaseException:
        status.update(status='failed',error=traceback.format_exc(),failed_utc=now());raise
    finally:
        write(path,status)


if __name__=='__main__':
    main()
