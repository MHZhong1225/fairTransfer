"""Second independent Qwen API formal grading queue with resumable outputs."""
import os
import subprocess
import time
import traceback
from pathlib import Path
from common import STUDY, now, read, rows, sha, write


def main():
    path = STUDY / 'results/qwen_api_alt_formal_queue_v2.json'
    status = dict(status='running', stage='waiting_native_outputs', pid=os.getpid(),
                  started_utc=now(), model='qwen3-omni-flash',
                  source_sha256=sha(__file__), finished=[])
    write(path, status)
    try:
        freeze = read(STUDY / 'configs/formal_v1/freeze.json')
        for state in rows(freeze['states_manifest']):
            root = Path(state['aggregate']).parent
            ready = root / 'native_ready.json'
            while not ready.exists():
                status.update(stage='waiting_native_outputs', state=state['id'], updated_utc=now())
                write(path, status)
                time.sleep(30)
            output_root = root / 'grades_api_qwen_alt_v3'
            output_root.mkdir(exist_ok=True)
            for task, manifest, measured in (
                ('understanding', Path(freeze['u_cases']), Path(state['understanding'])),
                ('depiction', Path(freeze['g_cases']), Path(state['generation'])),
                ('appearance', Path(freeze['g_cases']), Path(state['generation'])),
                ('content', Path(freeze['g_cases']), Path(state['generation'])),
            ):
                output = output_root / task
                if (output / 'status.json').exists() and read(output / 'status.json').get('status') == 'completed':
                    continue
                status.update(stage='api_grading', state=state['id'], task=task, updated_utc=now())
                write(path, status)
                env = dict(os.environ, QWEN_API_MODE='1', QWEN_API_MODEL='qwen3-omni-flash')
                subprocess.run([
                    os.environ.get('PYTHON', '/home/ubuntu/anaconda3/envs/ucp/bin/python'),
                    str(STUDY / 'scripts/api_score_outputs_v1.py'),
                    '--judge', 'ministral', '--task', task,
                    '--manifest', str(manifest), '--measured', str(measured),
                    '--output', str(output),
                ], cwd=STUDY, env=env, check=True)
            status['finished'].append(state['id'])
            write(path, dict(status, updated_utc=now()))
        status.update(status='completed', stage='all_states_qwen_api_alt_graded', completed_utc=now())
    except BaseException:
        status.update(status='failed', error=traceback.format_exc(), failed_utc=now())
        raise
    finally:
        write(path, status)


if __name__ == '__main__':
    main()
