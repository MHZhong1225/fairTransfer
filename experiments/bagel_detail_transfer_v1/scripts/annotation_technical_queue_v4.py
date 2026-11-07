"""Run the excluded-data annotation qualification after actual GPU jobs finish."""
import os
from pathlib import Path
import subprocess
import time
import traceback
from common import STUDY,PYTHON,now,read,sha,write


def launch(script,python,gpu):
    log=(STUDY/'logs'/f'{script}_technical_v4.log').open('x')
    return subprocess.Popen([str(python),str(STUDY/'scripts'/script),'technical'],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_DISABLE_PROGRESS_BARS='1'),
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)


def main():
    path=STUDY/'results/annotation_technical_queue_v4.json'
    status=dict(status='waiting_backend_qualification',started_utc=now(),source_sha256=sha(__file__));write(path,status)
    try:
        proposal=launch('propose_facts_v4.py',PYTHON,1)
        status['proposal_pid']=proposal.pid;write(path,status)
        while True:
            qualification=read(STUDY/'results/backend_qualification_v3/status.json')
            if qualification['status']=='failed':
                raise RuntimeError('Backend qualification failed; keep proposal output for diagnosis only')
            if qualification['status']=='passed':
                break
            time.sleep(5)
        if read(STUDY/'results/fact_judge_controls_v3/ministral_batch/status.json')['status']!='passed':
            raise RuntimeError('Verifier controls failed')
        verifier=launch('verify_facts_v4.py',STUDY/'environment_judge/bin/python',0)
        status.update(status='running_technical',verifier_pid=verifier.pid);write(path,status)
        a=proposal.wait();b=verifier.wait()
        if a or b:
            raise RuntimeError(f'Technical annotation exited {a}, {b}')
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(path,status)


if __name__=='__main__':
    main()
