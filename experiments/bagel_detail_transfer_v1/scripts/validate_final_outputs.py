"""Verify complete paired native outputs before declaring the core experiment done."""
from pathlib import Path
import time
from common import STUDY,now,read,rows,sha,write
from aggregate_outputs import indexed
from run_formal_worker import validate_freeze


def verify_generation_artifacts(value,base,trained):
    image=Path(value['image_path']);noise=image.with_suffix('.noise.npy')
    if sha(image)!=value['image_sha256'] or sha(noise)!=value['noise_file_sha256']:
        raise ValueError('A generated image or saved actual noise file changed')
    if (value['noise_file_sha256']!=base['noise_file_sha256'] or
            value['noise_tensor_sha256']!=base['noise_tensor_sha256']):
        raise ValueError('Generated comparisons do not share the actual initial noise')
    if trained and not value['actual_noise_replayed']:
        raise ValueError('A trained state did not replay the saved base noise')
    if value['generation_settings']!=base['generation_settings']:
        raise ValueError('Paired generation settings differ')


def verify_training_endpoint(state,freeze):
    directory=Path(state['training']);contract=read(directory/'contract.json')
    status=read(directory/'status.json');latest=read(directory/'latest.json')
    adapter=Path(status['final_adapter']);complete=read(adapter.parent/'complete.json')
    expected=contract['epochs']*((contract['train_images']+contract['batch_images']-1)//contract['batch_images'])
    if (status['status']!='completed' or status['step']!=expected or latest!=complete or
            complete['step']!=expected or complete['next_epoch']!=contract['epochs'] or complete['next_offset']!=0):
        raise ValueError('Training did not reach and save its fixed final endpoint')
    if contract['arm']!=state['arm'] or contract['seed']!=state['seed'] or contract['technical']:
        raise ValueError('Wrong formal training arm or seed')
    if (contract['train_sha256']!=sha(freeze['train']) or contract['dev_sha256']!=sha(freeze['dev']) or
            status['training_contract_sha256']!=sha(directory/'contract.json')):
        raise ValueError('Training inputs or contract changed')
    if sha(adapter/'adapter.safetensors')!=read(adapter/'adapter.json')['sha256']:
        raise ValueError('Final trained adapter changed')
    if not status['frozen']['unchanged'] or not status['frozen']['gradients_absent']:
        raise ValueError('Base model freezing was not verified')


def main():
    config=STUDY/'configs/formal_v1';freeze=read(config/'freeze.json');states=rows(config/'states.jsonl')
    if freeze.get('grading_location')=='remote_gpu':
        path=Path(freeze['remote_grading_queue_status'])
        while True:
            status=read(path)
            if status['status']=='failed':
                raise RuntimeError('Remote formal grading failed; native output completion is not full completion')
            if status['status']=='completed':
                break
            time.sleep(30)
    validate_freeze(freeze)
    planned_u={r['id'] for r in rows(freeze['u_cases'])};planned_g={r['id'] for r in rows(freeze['g_cases'])}
    base=next(s for s in states if s['arm']=='base');base_g=indexed(base['generation']);audit=[]
    for state in states:
        u=indexed(state['understanding']);g=indexed(state['generation'])
        if set(u)!=planned_u or set(g)!=planned_g:
            raise ValueError(f'Incomplete native outputs for {state["id"]}')
        if state['arm']!='base':
            verify_training_endpoint(state,freeze)
        for identifier,value in g.items():
            verify_generation_artifacts(value,base_g[identifier],state['arm']!='base')
        grading={}
        for judge in ('qwen','ministral'):
            grading[judge]={}
            for task in ('understanding','depiction','appearance','content'):
                data=indexed(Path(state['grades'])/judge/task)
                expected=planned_u if task=='understanding' else planned_g
                if set(data)!=expected:
                    raise ValueError('Missing or unexpected grading attempts')
                grading[judge][task]=dict(planned=len(expected),parsed=sum(r['status']=='completed' for r in data.values()),
                    failed_or_missing=sum(r['status']!='completed' for r in data.values()))
        audit.append(dict(state=state['id'],u_cases=len(u),g_images=len(g),grading=grading))
    write(STUDY/'results/formal_v1/final_integrity.json',dict(status='passed',completed_utc=now(),
        states=audit,total_generated_images=sum(r['g_images'] for r in audit),source_sha256=sha(__file__),
        interpretation='Complete native outputs and attempted grades; parse failures remain failures and are reported, not relabeled successes.'))


if __name__=='__main__':
    main()
