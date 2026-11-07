"""Durable end-to-end scheduling, independent of continued chat generation."""
import os
from pathlib import Path
import subprocess
import time
import traceback
from common import STUDY,PYTHON,now,read,sha,write


def launch(script,args,label,python=PYTHON,gpu=None):
    logdir=STUDY/'logs/full_study_queue_v4';logdir.mkdir(exist_ok=True)
    env=dict(os.environ,HF_HUB_DISABLE_PROGRESS_BARS='1')
    if gpu is not None:
        env['CUDA_VISIBLE_DEVICES']=str(gpu)
    with (logdir/(label+'.log')).open('x') as log:
        process=subprocess.Popen([str(python),str(STUDY/'scripts'/script),*map(str,args)],
            env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(logdir/(label+'.launch.json'),dict(pid=process.pid,started_utc=now(),script=script,
        args=list(map(str,args)),python=str(python),gpu=gpu,source_sha256=sha(STUDY/'scripts'/script)))
    return process


def finish(process,label):
    code=process.wait()
    if code:
        raise RuntimeError(f'{label} exited {code}; original logs and outputs retained')


def wait_technical(status,status_path):
    dependencies=[STUDY/'results'/p for p in ('data_quality_queue_v2_qwen.json',
        'mini_scoring_queue_v5.json')]
    while True:
        values={str(p):read(p) for p in dependencies}
        if any(v['status']=='failed' for v in values.values()):
            raise RuntimeError('Technical prerequisite failed: '+str({p:v.get('error') for p,v in values.items() if v['status']=='failed'}))
        status.update(stage='waiting_technical_pipeline',dependencies={p:v['status'] for p,v in values.items()},updated_utc=now())
        write(status_path,status)
        if all(v['status']=='completed' for v in values.values()):
            return
        time.sleep(30)


def main():
    status_path=STUDY/'results/full_study_queue_v4.json'
    if status_path.exists():
        raise RuntimeError('Overall queue already exists; inspect rather than duplicate it')
    status=dict(status='running',pid=os.getpid(),started_utc=now(),stage='waiting_technical_pipeline',
        source_sha256=sha(__file__),
        authorized_workload=dict(training_runs=12,training_arms=['T0','T1','T2','T3'],seeds=[7101,7102,7103],
            evaluation_states=13,g_images_per_state=480,total_formal_g_images=6240,
            u_cohort='All eligible held-out images; no test-image cap',judges=['qwen','ministral']),
        policy='Start each dependent stage automatically after its actual qualification. Preserve errors and halt on failed gates; never fabricate a completion or silently change scientific settings.')
    write(status_path,status)
    try:
        wait_technical(status,status_path)
        status.update(stage='base_quality_grading',updated_utc=now());write(status_path,status)
        quality=STUDY/'results/base_quality_v1'
        preflight=STUDY/'results/remote_preflight_queue_v4.json'
        while read(preflight)['status']!='completed':
            if read(preflight)['status']=='failed':
                raise RuntimeError('Official-size remote scoring qualification failed')
            time.sleep(15)
        source=STUDY/'results/remote_grading_preflight_v4/quality_grades'
        target=quality/'quality_grades'
        if target.is_symlink():
            if target.resolve()!=source.resolve():
                raise RuntimeError('Existing quality grades refer to a different execution')
        elif target.exists():
            raise RuntimeError('Do not overwrite existing quality grades')
        else:
            target.symlink_to(source,target_is_directory=True)
        write(quality/'quality_grades_origin.json',dict(source=str(source),
            remote_execution_sha256=sha(source/'remote_execution.json'),
            rationale='Reuse all ten already measured and hash-verified remote base-quality grades. No image selection or relabeling.'))
        finish(launch('review_base_quality.py',[],'review_base_quality'),'base quality verification')
        status.update(stage='cpu_checks_and_formal_freeze',updated_utc=now());write(status_path,status)
        log=STUDY/'logs/full_study_queue_v4/cpu_checks.log'
        with log.open('x') as stream:
            checks=subprocess.run([str(PYTHON),'-m','unittest','discover','-s',str(STUDY/'scripts'),'-p','test_*.py'],stdout=stream,stderr=subprocess.STDOUT)
        if checks.returncode:
            raise RuntimeError('Pre-formal arithmetic/interface tests failed')
        finish(launch('freeze_formal_v4.py',[],'freeze_formal'),'formal freeze')
        freeze=STUDY/'configs/formal_v1/freeze.json'
        status.update(stage='formal_training_and_evaluation',freeze=str(freeze),updated_utc=now());write(status_path,status)
        workers=[launch('run_formal_worker.py',['--freeze',freeze,'--gpu',gpu],f'formal_gpu{gpu}') for gpu in (0,1)]
        status['formal_worker_pids']=[p.pid for p in workers];write(status_path,status)
        for gpu,worker in enumerate(workers):
            finish(worker,f'formal GPU {gpu}')
        status.update(stage='integrity_and_paired_analysis',updated_utc=now());write(status_path,status)
        finish(launch('validate_final_outputs.py',[],'final_integrity'),'final integrity')
        finish(launch('compare_states.py',['--states',STUDY/'configs/formal_v1/states.jsonl',
            '--output',STUDY/'results/formal_v1/comparisons'],'paired_comparisons'),'paired comparisons')
        finish(launch('write_actual_reports.py',[],'actual_reports'),'actual result reports')
        status.update(status='completed',stage='reports_ready',completed_utc=now(),
            report=str(STUDY/'reports/main_results.md'),manual_interpretation_review='pending_agent_review')
    except BaseException:
        status.update(status='failed',error=traceback.format_exc(),failed_utc=now());raise
    finally:
        write(status_path,status)


if __name__=='__main__':
    main()
