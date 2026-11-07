"""Choose the already-ready server without delaying the local source pipeline."""
import json
import fcntl
import time
from common import STUDY,now,read,write


DECISION=STUDY/'results/remote_qwen_source_decision_v1.json'


def choose(execution,judge='qwen'):
    decision=STUDY/'results'/f'remote_{judge}_source_decision_v1.json'
    with decision.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not decision.exists():
            write(decision,dict(execution=execution,decided_utc=now()))
        return read(decision)['execution']


def use_remote_if_running(judge):
    config=STUDY/'configs/remote_node_v1.json'
    if not config.exists() or judge not in read(config).get('assigned_judges',['qwen']):
        return False
    # The remote controller claims the task only after model files and imports
    # are ready. If local GPU work reaches this point earlier, use it normally.
    if choose('local',judge)=='local':
        return False
    path=STUDY/'results'/f'remote_{judge}_preparation_queue_v1.json'
    while True:
        status=read(path)
        if status['status']=='failed':
            raise RuntimeError('Remote source screen failed; inspect its preserved controls and evidence')
        if status['status']=='completed' and status.get('imported_source_screen'):
            if read(STUDY/f'results/depiction_check_v1/{judge}/status.json')['status']!='completed':
                raise RuntimeError('Remote source screen import is incomplete')
            return True
        time.sleep(10)
