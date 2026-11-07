"""Transfer identical scoring inputs, execute one remote bundle, retrieve evidence."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from common import STUDY,now,read,rows,sha,write,write_rows
from aggregate_outputs import indexed
from remote_qwen_preparation_queue import HOST,PORT,REMOTE,ssh_python

SOURCES=('remote_grade_bundle_v2.py','score_outputs_remote_v2.py','qualify_scoring_remote_v2.py',
    'understanding_grading_v4.py','remote_judge_runtime.py','judge_runtime.py','ministral_judge.py','ministral_batch.py',
    'grading.py','depiction.py','annotate_facts_v3.py','common.py')


def synchronize(source,destination):
    result=subprocess.run(['rsync','-a','--partial','-e',f'ssh -o BatchMode=yes -o ConnectTimeout=15 -p {PORT}',
        str(source),str(destination)],capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError('Scoring transfer failed, original files retained: '+result.stderr[-1500:])


def execute_bundle(name,request,staging):
    remote_job=REMOTE+'/jobs/'+name
    request=dict(request,remote_root=REMOTE,
        source_hashes={name:sha(STUDY/'scripts'/name) for name in SOURCES})
    write(staging/'request.json',request)
    subprocess.run(['ssh','-o','BatchMode=yes','-p',PORT,HOST,'mkdir','-p',remote_job],check=True,capture_output=True)
    synchronize(str(staging)+'/',HOST+':'+remote_job+'/')
    launch=ssh_python(f'''from pathlib import Path
import subprocess,os,json,fcntl
job=Path({remote_job!r});marker=job/'launch.json'
with (job/'launch.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 if marker.exists():
  result=json.loads(marker.read_text())
 else:
  if (job/'bundle.log').exists():
   raise RuntimeError('Unreconciled previous launch; never create a duplicate GPU worker')
  with (job/'bundle.log').open('x') as log:
   p=subprocess.Popen(['python3',{(REMOTE+'/scripts/remote_grade_bundle_v2.py')!r},'--job',str(job)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  result=dict(pid=p.pid);tmp=job/'launch.json.tmp';tmp.write_text(json.dumps(result));tmp.replace(marker)
 print(json.dumps(result))
''')
    write(staging/'remote_launch.json',dict(launch,remote_job=remote_job,started_utc=now()))
    while True:
        status=ssh_python(f'''from pathlib import Path
import json
p=Path({(remote_job+'/status.json')!r});state=json.loads(p.read_text()) if p.exists() else dict(status='starting')
proc=Path('/proc/{launch['pid']}/cmdline');state['process_alive']=bool(proc.exists() and proc.read_bytes())
print(json.dumps(state))
''')
        write(staging/'remote_status.json',status)
        if status['status']=='failed' or (not status['process_alive'] and status['status']!='completed'):
            raise RuntimeError('Remote scoring bundle failed: '+str(status))
        if status['status']=='completed':
            return remote_job
        time.sleep(30)


def deploy_scoring_sources():
    for name in SOURCES:
        synchronize(STUDY/'scripts'/name,HOST+':'+REMOTE+'/scripts/'+name)
    scene=rows(STUDY/'data/scene_technical_v1/manifest.jsonl')
    mapping={r['id']:r['image_path'] for r in rows(STUDY/'data/remote_node_v1/manifest.jsonl')}
    directory=STUDY/'data/remote_grading_control_v1';directory.mkdir(exist_ok=True)
    target=directory/'manifest.jsonl'
    if not target.exists():
        write_rows(target,[dict(row,image_path=mapping[row['id']]) for row in scene])
    subprocess.run(['ssh','-o','BatchMode=yes','-p',PORT,HOST,'mkdir','-p',REMOTE+'/data/scene_technical_v1'],check=True,capture_output=True)
    synchronize(target,HOST+':'+REMOTE+'/data/scene_technical_v1/manifest.jsonl')


def grade_outputs(name,g_manifest,measured_g,destination,u_manifest=None,measured_u=None):
    staging=STUDY/'data/remote_grading_staging_v2'/name
    staging.mkdir(parents=True,exist_ok=False);inputs=staging/'inputs';inputs.mkdir()
    remote_job=REMOTE+'/jobs/'+name
    source_images={r['id']:r['image_path'] for r in rows(STUDY/'data/remote_node_v1/manifest.jsonl')}
    original_hashes={};original_observations={};planned={}
    for task,manifest,measured in (('generation',g_manifest,measured_g),('understanding',u_manifest,measured_u)):
        if manifest is None:
            continue
        cases=rows(manifest);actual=indexed(measured);expected={r['id'] for r in cases}
        if not expected <= set(actual):
            raise ValueError('Cannot remotely grade incomplete native output')
        (inputs/task/'observations').mkdir(parents=True)
        original_observations[task]={str(p):sha(p) for p in Path(measured).rglob('observations/*.json')
            if p.stem in expected}
        translated=[]
        for item in cases:
            copy=dict(item)
            if task=='understanding' and item.get('image_path'):
                copy['image_path']=source_images[item['image_id']]
            if {k:v for k,v in copy.items() if k!='image_path'}!={k:v for k,v in item.items() if k!='image_path'}:
                raise ValueError('Remote question or reference stimulus changed')
            translated.append(copy)
            observed=dict(actual[item['id']]);original={k:v for k,v in observed.items()}
            if task=='generation':
                image_path=Path(observed['image_path'])
                if sha(image_path)!=observed['image_sha256']:
                    raise ValueError('Measured image changed before transfer')
                images=inputs/'images';images.mkdir(exist_ok=True)
                target=images/(item['id']+'.png')
                try:
                    os.link(image_path,target)
                except OSError:
                    shutil.copyfile(image_path,target)
                observed['image_path']=remote_job+'/inputs/images/'+target.name
            if {k:v for k,v in observed.items() if k!='image_path'}!={k:v for k,v in original.items() if k!='image_path'}:
                raise ValueError('Remote model response or recorded outcome changed')
            write(inputs/task/'observations'/(item['id']+'.json'),observed)
        manifest_name='g_cases.jsonl' if task=='generation' else 'u_cases.jsonl'
        write_rows(inputs/manifest_name,translated);planned[task]=expected
        original_hashes[str(manifest)]=sha(manifest)
    tasks=['understanding','depiction','appearance','content'] if u_manifest else ['depiction','appearance','content']
    write(staging/'input_lineage.json',dict(created_utc=now(),source_manifests=original_hashes,source_observations=original_observations,
        location_translation='Only image_path is translated. Questions, reference facts, model responses, image SHA, and planned IDs are unchanged.',
        staged_hashes={str(p.relative_to(inputs)):sha(p) for p in inputs.rglob('*.json*')},
        remote_source_image_manifest_sha256=sha(STUDY/'data/remote_node_v1/manifest.jsonl')))
    remote_job=execute_bundle(name,dict(mode='score',tasks=tasks),staging)
    incoming=destination.with_name(destination.name+'_incoming');incoming.mkdir(parents=True,exist_ok=False)
    synchronize(HOST+':'+remote_job+'/grades/',str(incoming)+'/')
    for judge in ('qwen','ministral'):
        for task in tasks:
            records=indexed(incoming/judge/task)
            source='understanding' if task=='understanding' else 'generation'
            if set(records)!=planned[source]:
                raise ValueError('Remote grades omit or add planned cases')
            for identifier,grade in records.items():
                expected_sha=sha(inputs/source/'observations'/(identifier+'.json'))
                if grade.get('measured_sha256')!=expected_sha:
                    raise ValueError('Remote grade does not refer to the staged measured output')
    write(incoming/'remote_execution.json',dict(completed_utc=now(),remote_job=remote_job,
        request_sha256=sha(staging/'request.json'),input_lineage_sha256=sha(staging/'input_lineage.json'),
        source_sha256=sha(__file__),host=HOST))
    if destination.exists():
        raise RuntimeError('Never overwrite existing scientific grades')
    incoming.rename(destination)


def qualify_remote_scoring():
    directory=STUDY/'results/remote_grading_preflight_v2'
    if (directory/'status.json').exists():
        previous=read(directory/'status.json')
        request=read(STUDY/'data/remote_grading_staging_v2/qualification_v2/request.json')
        if previous['status']=='passed' and previous.get('source_sha256')==sha(__file__) and all(
                sha(STUDY/'scripts'/name)==digest for name,digest in request['source_hashes'].items()):
            return
        raise RuntimeError('Previous remote preflight cannot be silently replaced or reinterpreted')
    directory.mkdir(exist_ok=False);write(directory/'status.json',dict(status='running',started_utc=now()))
    try:
        deploy_scoring_sources()
        staging=STUDY/'data/remote_grading_staging_v2/qualification_v2'
        if staging.exists():
            expected=dict(mode='qualify',tasks=[],remote_root=REMOTE,
                source_hashes={name:sha(STUDY/'scripts'/name) for name in SOURCES})
            if read(staging/'request.json')!=expected:
                raise RuntimeError('Earlier remote controls used a different contract; do not silently reuse them')
        else:
            staging.mkdir(parents=True)
        remote_job=execute_bundle('qualification_v2',dict(mode='qualify',tasks=[]),staging)
        synchronize(HOST+':'+remote_job+'/model_verification.json',directory/'model_verification.json')
        synchronize(HOST+':'+REMOTE+'/results/scoring_qualification_remote_v2/',str(directory/'qualification')+'/')
        for judge in ('qwen','ministral'):
            if read(directory/'qualification'/judge/'status.json')['status']!='passed':
                raise ValueError('Remote scorer did not pass the actual controls')
        grade_outputs('base_quality',STUDY/'configs/generation_plan_v1/quality_check.jsonl',
            STUDY/'results/base_quality_v1',directory/'quality_grades')
        for judge in ('qwen','ministral'):
            for task in ('depiction','appearance','content'):
                values=indexed(directory/'quality_grades'/judge/task)
                if sum(v['status']=='completed' for v in values.values())<.9*len(values):
                    raise ValueError('Remote official-size quality grading is not structurally reliable')
        write(directory/'status.json',dict(status='passed',completed_utc=now(),source_sha256=sha(__file__),
            scope='Same positive/contradictory/empty controls plus real 1024-pixel official base images on the third GPU. No resolution, precision, decoding or parser change.',remote_job=remote_job))
    except BaseException as error:
        write(directory/'status.json',dict(status='failed',failed_utc=now(),error=str(error)));raise
