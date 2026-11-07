"""Wait for the owned annotation job to release GPU 0, then verify BAGEL."""
import os
import subprocess
import time
import traceback

from common import STUDY,PYTHON,now,read,write


def main():
    status_path = STUDY/'results/qualification_queue.json'
    status = dict(status='waiting_for_annotation_gpu0',started_utc=now())
    write(status_path,status)
    try:
        source = STUDY/'results/fact_annotation_technical_v2/propose/status.json'
        while True:
            observed = read(source)
            if observed['status'] == 'failed':
                raise RuntimeError('Technical annotation producer failed')
            if observed['status'] == 'completed':
                free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True).strip())
                if free > 44000:
                    break
            time.sleep(10)
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        command = [str(PYTHON),str(STUDY/'scripts/qualify_backend.py')]
        with (STUDY/'logs/backend_qualification_v1.log').open('x') as stream:
            child = subprocess.Popen(command,env=environment,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            status.update(status='running_backend_qualification',pid=child.pid,command=command,updated_utc=now())
            write(status_path,status)
            result = child.wait()
        if result:
            raise RuntimeError(f'BAGEL qualification exited with code {result}')
        if read(STUDY/'results/backend_qualification_v1/status.json')['status']!='passed':
            raise ValueError('Qualification did not pass')
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc(),updated_utc=now());raise
    finally:
        write(status_path,status)


if __name__=='__main__':
    main()
