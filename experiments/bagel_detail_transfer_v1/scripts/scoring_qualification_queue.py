"""Qualify scoring on the verifier GPU after its factual labels are complete."""
import os
import subprocess
import time
import traceback
from common import STUDY,PYTHON,now,read,sha,write


def main():
    path=STUDY/'results/scoring_qualification_queue_v1.json'
    status=dict(status='waiting_verifier',started_utc=now(),source_sha256=sha(__file__));write(path,status)
    try:
        while True:
            state=read(STUDY/'results/fact_annotation_full_v5/verify/status.json')['status']
            if state=='failed':
                raise RuntimeError('Source verifier failed')
            if state=='completed':
                break
            time.sleep(10)
        for judge in ('qwen','ministral'):
            status.update(status='qualifying',judge=judge);write(path,status)
            python=PYTHON if judge=='qwen' else STUDY/'environment_judge/bin/python'
            process=subprocess.Popen([str(python),str(STUDY/'scripts/qualify_scoring.py'),judge],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_DISABLE_PROGRESS_BARS='1'),
                stdin=subprocess.DEVNULL,stdout=(STUDY/'logs'/f'scoring_qualification_{judge}.log').open('x'),
                stderr=subprocess.STDOUT,start_new_session=True)
            if process.wait():
                raise RuntimeError(f'{judge} scoring controls failed')
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(path,status)


if __name__=='__main__':
    main()
