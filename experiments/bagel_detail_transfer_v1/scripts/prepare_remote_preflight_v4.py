"""Run actual official-size remote scoring now that native base images exist."""
import os
import time
import traceback
from common import STUDY,now,read,sha,write
from remote_grading_v4 import qualify_remote_scoring


def main():
    path=STUDY/'results/remote_preflight_queue_v4.json'
    if path.exists():
        raise RuntimeError('Existing preflight queue must be inspected, not overwritten')
    status=dict(status='running',stage='waiting_base_quality',pid=os.getpid(),started_utc=now(),source_sha256=sha(__file__))
    write(path,status)
    try:
        while read(STUDY/'results/base_quality_v1/status.json')['status']!='completed':
            if read(STUDY/'results/data_quality_queue_v2_qwen.json')['status']=='failed':
                raise RuntimeError('Native base-quality generation failed')
            time.sleep(15)
        status.update(stage='remote_full_preflight');write(path,status)
        qualify_remote_scoring()
        status.update(status='completed',stage='remote_preflight_passed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(path,status)


if __name__=='__main__':
    main()
