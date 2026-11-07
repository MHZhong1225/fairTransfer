"""Requalify the revised scorer on the released local GPU; preserve v1 failures."""
import os
import traceback
from common import STUDY,PYTHON,now,read,sha,write
from data_quality_queue_v2 import execute


def main():
    path=STUDY/'results/data_quality_queue_v3_ministral.json'
    if path.exists():
        raise RuntimeError('Existing revised queue must be inspected, not overwritten')
    status=dict(status='running',pid=os.getpid(),started_utc=now(),source_sha256=sha(__file__))
    write(path,status)
    try:
        assert read(STUDY/'results/fact_annotation_full_v5/verify/status.json')['status']=='completed'
        assert read(STUDY/'results/depiction_check_v1/ministral/status.json')['status']=='completed'
        for judge in ('qwen','ministral'):
            status.update(stage='scoring_qualification',scorer=judge);write(path,status)
            python=PYTHON if judge=='qwen' else STUDY/'environment_judge/bin/python'
            execute('qualify_scoring_v2.py',[judge],python,0,f'scoring_qualification_v2_{judge}.log')
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(path,status)


if __name__=='__main__':
    main()
