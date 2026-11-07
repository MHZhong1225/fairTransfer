"""Score formal native outputs with an independent second Qwen API model."""
import os
import subprocess
import time
import traceback
from pathlib import Path
from common import STUDY, now, read, rows, sha, write


def run_task(task, manifest, measured, output):
    command = [
        os.environ.get('PYTHON', '/home/ubuntu/anaconda3/envs/ucp/bin/python'),
        str(STUDY / 'scripts/score_outputs_remote_v4.py'),
        '--judge', 'ministral',
        '--task', task,
        '--manifest', str(manifest),
        '--measured', str(measured),
        '--output', str(output),
    ]
    env = dict(os.environ, QWEN_API_MODE='1', QWEN_API_MODEL='qwen3.5-omni-plus')
    subprocess.run(command, cwd=STUDY, env=env, check=True)


def main():
    path = STUDY / 'results/qwen_api_alt_formal_queue_v1.json'
    status = dict(status='running', stage='waiting_native_outputs', pid=os.getpid(),
                  started_utc=now(), model='qwen3.5-omni-plus',
                  source_sha256=sha(__file__), finished=[])
    write(path, status)
    try:
        freeze = read(STUDY / 'configs/formal_v1/freeze.json')
        states = rows(freeze['states_manifest'])
        for state in states:
            root = Path(state['aggregate']).parent
            ready = root / 'native_ready.json'
            while not ready.exists():
                status.update(stage='waiting_native_outputs', state=state['id'], updated_utc=now())
                write(path, status)
                time.sleep(30)
            output_root = root / 'grades_api_qwen_alt'
            output_root.mkdir(exist_ok=True)
            tasks = [
                ('understanding', Path(freeze['u_cases']), Path(state['understanding'])),
                ('depiction', Path(freeze['g_cases']), Path(state['generation'])),
                ('appearance', Path(freeze['g_cases']), Path(state['generation'])),
                ('content', Path(freeze['g_cases']), Path(state['generation'])),
            ]
            for task, manifest, measured in tasks:
                output = output_root / task
                if (output / 'status.json').exists() and read(output / 'status.json').get('status') == 'completed':
                    continue
                status.update(stage='api_grading', state=state['id'], task=task, updated_utc=now())
                write(path, status)
                run_task(task, manifest, measured, output)
            status['finished'].append(state['id'])
            status['updated_utc'] = now()
            write(path, status)
        status.update(status='completed', stage='all_states_qwen_api_alt_graded', completed_utc=now())
    except BaseException:
        status.update(status='failed', error=traceback.format_exc(), failed_utc=now())
        raise
    finally:
        write(path, status)


if __name__ == '__main__':
    main()
