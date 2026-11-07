"""Ready, run and retrieve the third GPU's finite source-quality workload."""
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import traceback
from common import STUDY,now,read,rows,sha,write
from remote_source_handoff import choose

HOST='icdm@218.194.61.160';PORT='51888';REMOTE='/home/icdm/umm_bias_bagel_detail_v1'


def ssh_python(code):
    while True:
        try:
            result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p',PORT,HOST,'python3','-'],
                input=code,text=True,capture_output=True,timeout=45)
            if result.returncode==255:
                raise ConnectionError(result.stderr[-1000:])
        except (subprocess.TimeoutExpired,ConnectionError) as error:
            write(STUDY/'results/remote_node_connection_v1.json',dict(status='observation_unavailable',
                observed_utc=now(),error=str(error)[-1200:],action='Retry observation of the same remote state; do not restart a possibly live job.'))
            time.sleep(15)
            continue
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        return json.loads(result.stdout)


def resume_other_transfer():
    path=STUDY/'results/remote_ministral_transfer_priority_v1.json'
    if not path.exists():
        return
    value=read(path);cmd=Path(f'/proc/{value["pid"]}/cmdline')
    if cmd.exists() and cmd.read_bytes().split(b'\0')[:-1]==[a.encode() for a in value['argv']]:
        os.kill(value['pid'],signal.SIGCONT)
    value['resumed_utc']=now();write(path,value)


def main():
    path=STUDY/'results/remote_qwen_preparation_queue_v1.json'
    status=dict(status='running',stage='waiting_assets',pid=os.getpid(),started_utc=now(),
        source_sha256=sha(__file__),server=HOST,port=PORT,planned=4247)
    write(path,status)
    try:
        while True:
            # Inspect only the newly created project directory, never user data.
            ready=ssh_python(f'''from pathlib import Path
import json
r=Path({REMOTE!r});p=json.loads((r/'data/qwen_model_provenance.json').read_text())
model=r/'models/Qwen3-VL-8B-Instruct'
missing=[name for name,d in p['files'].items() if not (model/name).exists() or (model/name).stat().st_size!=d['bytes']]
print(json.dumps(dict(missing=missing,bytes_present=sum(x.stat().st_size for x in model.iterdir() if x.is_file()))))
''')
            status.update(assets=ready,updated_utc=now());write(path,status)
            if not ready['missing']:
                break
            if (STUDY/'results/remote_qwen_source_decision_v1.json').exists() and read(STUDY/'results/remote_qwen_source_decision_v1.json')['execution']=='local':
                resume_other_transfer()
                status.update(status='completed',stage='local_source_won_race_remote_assets_continue',imported_source_screen=False,completed_utc=now())
                return
            time.sleep(30)
        resume_other_transfer()
        # Loader imports must succeed before the remote host can claim work.
        subprocess.run(['ssh','-o','BatchMode=yes','-p',PORT,HOST,
            f'{REMOTE}/envs/qwen/bin/python -c '+shlex.quote('from transformers import Qwen3VLForConditionalGeneration,AutoProcessor; import torch; assert torch.cuda.is_available()')],check=True,capture_output=True,text=True)
        if choose('remote')!='remote':
            status.update(status='completed',stage='assets_ready_local_source_already_running',imported_source_screen=False,completed_utc=now())
            return
        remote_output=REMOTE+'/results/source_depiction_v1/qwen'
        command=[REMOTE+'/envs/qwen/bin/python',REMOTE+'/scripts/remote_depiction_worker.py','--judge','qwen',
            '--manifest',REMOTE+'/data/manifest.jsonl','--controls',REMOTE+'/data/controls.json',
            '--provenance',REMOTE+'/data/qwen_model_provenance.json','--output',remote_output]
        launch=ssh_python(f'''import subprocess,os,json
command={command!r}
with open({(REMOTE+'/logs/source_depiction_qwen.log')!r},'x') as log:
 p=subprocess.Popen(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',UMM_JUDGE_MODEL_ROOT={(REMOTE+'/models')!r},HF_HUB_DISABLE_PROGRESS_BARS='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid)))
''')
        status.update(stage='remote_source_screen',remote_pid=launch['pid'],remote_output=remote_output);write(path,status)
        while True:
            observed=ssh_python(f'''from pathlib import Path
import json
p=Path({(remote_output+'/status.json')!r});state=json.loads(p.read_text()) if p.exists() else dict(status='starting')
proc=Path('/proc/{launch['pid']}/cmdline');state['process_alive']=bool(proc.exists() and proc.read_bytes())
print(json.dumps(state))
''')
            status.update(remote=observed,updated_utc=now());write(path,status)
            if observed['status']=='failed' or (not observed['process_alive'] and observed['status']!='completed'):
                raise RuntimeError('Remote depiction process failed: '+str(observed))
            if observed['status']=='completed':
                break
            time.sleep(30)
        incoming=STUDY/'results/remote_qwen_source_incoming_v1';incoming.mkdir()
        subprocess.run(['rsync','-a','-e',f'ssh -o BatchMode=yes -p {PORT}',
            HOST+':'+remote_output+'/',str(incoming)+'/'],check=True,capture_output=True,text=True)
        expected={r['id']:r['image_sha256'] for r in rows(STUDY/'data/remote_node_v1/manifest.jsonl')}
        actual={p.stem:read(p) for p in (incoming/'observations').glob('*.json')}
        if set(actual)!=set(expected) or any(r['image_sha256']!=expected[key] for key,r in actual.items()):
            raise ValueError('Remote screen does not match the complete original image pool')
        controls=read(incoming/'controls.json')
        if len(controls)!=7 or any(r['expected']!=r['parsed']['category'] for r in controls):
            raise ValueError('Remote qualification controls did not pass')
        write(incoming/'import_lineage.json',dict(imported_utc=now(),server=HOST,remote_output=remote_output,
            images=len(actual),original_blind_manifest_sha256=sha(STUDY/'data/remote_node_v1/manifest.jsonl'),
            imported_hashes={str(p.relative_to(incoming)):sha(p) for p in incoming.rglob('*.json')},
            source_sha256=sha(__file__)))
        destination=STUDY/'results/depiction_check_v1/qwen';destination.parent.mkdir(exist_ok=True)
        if destination.exists():
            raise ValueError('A local source result already exists; never overwrite it')
        incoming.rename(destination)
        status.update(status='completed',stage='source_screen_imported',imported_source_screen=True,completed_utc=now())
    except BaseException:
        resume_other_transfer()
        status.update(status='failed',error=traceback.format_exc(),failed_utc=now());raise
    finally:
        write(path,status)


if __name__=='__main__':
    main()
