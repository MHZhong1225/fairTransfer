"""Qualify mixed-size scenes, then annotate the frozen full source pool."""
import os
from pathlib import Path
import subprocess
import time
import traceback
from common import STUDY,PYTHON,now,read,rows,sha,write,write_rows


def launch(name,python,gpu,args,log_name):
    return subprocess.Popen([str(python),str(STUDY/'scripts'/name),*map(str,args)],
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_DISABLE_PROGRESS_BARS='1'),
        stdin=subprocess.DEVNULL,stdout=(STUDY/'logs'/log_name).open('x'),stderr=subprocess.STDOUT,start_new_session=True)


def wait_pair(a,b):
    codes=[a.wait(),b.wait()]
    if any(codes):
        raise RuntimeError(f'Workers exited {codes}; partial outputs retained')


def annotate(manifest,output,label):
    arguments=['--manifest',manifest,'--output',output]
    a=launch('annotation_worker.py',PYTHON,1,arguments+['--task','propose'],label+'_propose.log')
    b=launch('annotation_worker.py',STUDY/'environment_judge/bin/python',0,arguments+['--task','verify'],label+'_verify.log')
    wait_pair(a,b)
    for task in ('propose','verify'):
        records=rows(output/task/'observations.jsonl')
        completed=sum(r['status']=='completed' for r in records)
        if completed<len(records)*.9:
            raise RuntimeError(f'{label} {task}: fewer than 90% structurally usable records; inspect without changing frozen outputs')


def main():
    path=STUDY/'results/annotation_queue_v5.json'
    status=dict(status='waiting_technical',started_utc=now(),source_sha256=sha(__file__));write(path,status)
    try:
        while True:
            state=read(STUDY/'results/annotation_technical_queue_v4.json')['status']
            if state=='failed':
                raise RuntimeError('Prior technical annotation failed')
            if state=='completed':
                break
            time.sleep(5)
        for task in ('propose','verify'):
            observations=rows(STUDY/f'results/fact_annotation_technical_v4/{task}/observations.jsonl')
            if sum(r['status']=='completed' for r in observations)<.9*len(observations):
                raise RuntimeError('Portrait technical annotations are not structurally reliable')
        status['status']='scene_controls';write(path,status)
        a=launch('qualify_scene_judges.py',PYTHON,1,['qwen'],'scene_controls_qwen.log')
        b=launch('qualify_scene_judges.py',STUDY/'environment_judge/bin/python',0,['ministral'],'scene_controls_ministral.log')
        wait_pair(a,b)
        status['status']='scene_technical_annotation';write(path,status)
        annotate(STUDY/'data/scene_technical_v1/blind_manifest.jsonl',STUDY/'results/fact_annotation_scene_technical_v5','scene_technical_v5')
        source_paths=[STUDY/'data'/name for name in ('bpo_pool_v2/manifest.jsonl','scene_supplement_v2/manifest.jsonl')]
        reserved={r['id'] for r in rows(STUDY/'data/scene_technical_v1/manifest.jsonl')}
        plan=sorted([r for p in source_paths for r in rows(p) if r['id'] not in reserved],key=lambda r:r['id'])
        directory=STUDY/'data/combined_sources_v1';directory.mkdir(exist_ok=False)
        write_rows(directory/'manifest.jsonl',plan)
        write_rows(directory/'blind_manifest.jsonl',[{k:r[k] for k in ('id','image_path','image_sha256')} for r in plan])
        write(directory/'contract.json',dict(created_utc=now(),sources={str(p):sha(p) for p in source_paths},
            reserved_technical=sorted(reserved),images=len(plan),blind_manifest_sha256=sha(directory/'blind_manifest.jsonl'),
            annotation_source_hashes={name:sha(STUDY/'scripts'/name) for name in
                ('annotation_worker.py','annotate_facts_v3.py','ministral_judge.py','ministral_batch.py')},
            purpose='Frozen source annotation pool before BAGEL outcome inspection; final training facts and weights are not yet frozen.'))
        status.update(status='full_annotation',planned=len(plan));write(path,status)
        annotate(directory/'blind_manifest.jsonl',STUDY/'results/fact_annotation_full_v5','full_annotation_v5')
        status.update(status='completed',completed_utc=now())
    except BaseException:
        status.update(status='failed',error=traceback.format_exc());raise
    finally:
        write(path,status)


if __name__=='__main__':
    main()
